"""Phase 3 / RQ3 -- does conditioning on INDETERMINACY beat conditioning on a
scalar confidence, at equal expansion budget?

Pre-registered in PREREGISTRATION.md. This is the experiment that decides
whether the narrowed claim survives.

Three conditions, identical retriever, identical reader, identical expansion
schedule. Only the TRIGGER differs:

  (a) FIXED-K      never expands.
  (b) SCALAR-GATE  expands when a scalar confidence falls below a threshold.
  (c) I-GATE       expands when I dominates the (T, I, F) triple.

The ablation is exact. (b) and (c) are computed from the SAME per-passage NLI
outputs at each stage -- the same model calls, cached and shared -- so the only
thing that varies between them is the decision function applied to those
outputs. Nothing else in the pipeline differs.

EQUAL BUDGET. (b)'s threshold is not chosen by hand. It is set to the quantile
of the scalar distribution that reproduces (c)'s expansion rate on the same
data, which equalises the budgets by construction. That is calibration. Once
set it is not touched again; adjusting it afterwards to move either arm's score
would be tuning and is forbidden by the pre-registration.

    python experiments/run_hotpot_gate.py --limit 40      # pilot
    python experiments/run_hotpot_gate.py                 # full, N=400

Scalar variants. The pre-registered scalar is the NLI head's own maximum class
probability, `max(e, n, c)`. That is reported as the primary (b). A difficulty
with it is noted in RESULTS.md and tested here as a robustness check: a
*confidently neutral* passage set scores HIGH on max(e,n,c), so this scalar is
structurally blind to the very state the I-gate fires on. Two stronger scalars
are therefore also calibrated and reported, so the verdict does not rest on the
weakest available baseline:

  max_enc   max(e, n, c)   -- pre-registered
  one_minus_n  1 - n       -- fires when neutrality is high
  entail    e              -- fires when support is weak
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neutrorag.core import TIF, clamp  # noqa: E402
from neutrorag.hotpot import (  # noqa: E402
    DIVERSITY_TARGET, SUPPORT_TARGET, HotpotExample, exact_match,
    load_hotpot_hf, stratified_sample, token_f1, type_counts,
)
from neutrorag.reader import ExtractiveReader  # noqa: E402
from neutrorag.retrieval import DenseRetriever, Document  # noqa: E402
from neutrorag.scoring import Evidence, HFNLIBackend, NLIRetrievalScorer  # noqa: E402

#: Base retrieval depth, and the schedule an expansion follows. Fixed here so
#: every condition expands the same way and only the trigger varies.
K_BASE = 3
K_SCHEDULE = (6, 10)

SAMPLE_SEED = 20261019
SAMPLE_N = 400
SCALARS = ("max_enc", "one_minus_n", "entail")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=SAMPLE_N)
    p.add_argument("--seed", type=int, default=SAMPLE_SEED)
    p.add_argument("--limit", type=int, default=None, help="pilot: first N of the sample")
    p.add_argument("--k-base", type=int, default=K_BASE)
    p.add_argument("--outdir", default="results")
    p.add_argument("--cache", default="results/hotpot_gate_cache.jsonl")
    # -- Addendum 4 (H1) --------------------------------------------------
    p.add_argument("--qtype", default=None,
                   help="restrict the pool to one question type")
    p.add_argument("--exclude-csv", default=None,
                   help="exclude every qid in this CSV and abort if the "
                        "resulting sample is not disjoint from it")
    p.add_argument("--out-prefix", default="hotpot_gate",
                   help="basename for the emitted CSV and meta JSON")
    p.add_argument("--no-analysis", action="store_true",
                   help="write the CSV and print only counts and elapsed "
                        "time; no accuracy, no tests, no thresholds")
    return p.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


# ---------------------------------------------------------------------------
# One stage of evidence for one question: the NLI outputs and everything
# derivable from them. Computed once, shared by both gates.
# ---------------------------------------------------------------------------
@dataclass
class Stage:
    k: int
    titles: List[str]
    texts: List[str]
    relevances: List[float]
    answer: str
    reader_score: float
    reader_null: float
    passage_index: Optional[int]
    enc: List[Tuple[float, float, float]] = field(default_factory=list)
    tif: Optional[TIF] = None
    r: float = 0.0

    # -- the two decision variables, from identical inputs -----------------
    @property
    def pooled(self) -> Tuple[float, float, float]:
        w = [clamp(x) for x in self.relevances]
        wsum = sum(w) or 1.0
        e = sum(wi * x[0] for wi, x in zip(w, self.enc)) / wsum
        n = sum(wi * x[1] for wi, x in zip(w, self.enc)) / wsum
        c = sum(wi * x[2] for wi, x in zip(w, self.enc)) / wsum
        return e, n, c

    def scalar(self, which: str) -> float:
        e, n, c = self.pooled
        if which == "max_enc":
            return max(e, n, c)
        if which == "one_minus_n":
            return 1.0 - n
        if which == "entail":
            return e
        raise ValueError(which)

    @property
    def i_dominates(self) -> bool:
        return self.tif is not None and self.tif.dominant == "I"


def score_stage(ex: HotpotExample, k: int, retr: DenseRetriever,
                reader: ExtractiveReader, nli, scorer: NLIRetrievalScorer,
                title_of: Dict[str, str]) -> Stage:
    """Retrieve at depth k, read, and score. One NLI pass over the passages."""
    got = retr.retrieve(ex.question, k=k)
    titles = [title_of[e.text] for e in got]
    texts = [e.text for e in got]
    rels = [e.relevance for e in got]

    out = reader.read(ex.question, texts)
    # Hypothesis = question + the reader's own answer. A question alone is not a
    # valid NLI hypothesis (the lesson of the first CONFLICTS run), so the gate
    # scores the answer it is about to commit to against the evidence it has.
    hyp = f"{ex.question} {out.answer}".strip()

    # Re-attach sources: DenseRetriever.retrieve drops the Document id, which
    # would silently zero the diversity component of r.
    ev = [Evidence(t, rel, source=ti) for t, rel, ti in zip(texts, rels, titles)]
    enc = [nli.predict(t, hyp) for t in texts]

    class _Replay:
        """Serve recorded NLI outputs BY PREMISE TEXT, not by call order.

        scorer.score drops blank passages before calling the backend, so a
        positional replay can mis-align without raising. Keying by premise
        makes a mis-alignment impossible, and an unexpected premise raises
        rather than returning a wrong triple.
        """

        def __init__(self, texts, rows):
            self.by_premise = dict(zip(texts, rows))
            self.calls = 0

        def predict(self, premise, hypothesis):
            self.calls += 1
            try:
                return self.by_premise[premise]
            except KeyError:  # pragma: no cover - guard, must never fire
                raise AssertionError(
                    "NLI replay desync: scorer.score asked for a premise that "
                    f"was not scored ({premise[:60]!r})")

    replay = _Replay(texts, enc)
    scorer.nli = replay
    tif = scorer.score(hyp, ev)
    r = scorer.sufficiency(hyp, ev).r
    scorer.nli = nli
    assert replay.calls == len(texts), (
        f"replay served {replay.calls} calls for {len(texts)} passages")

    return Stage(k=k, titles=titles, texts=texts, relevances=rels,
                 answer=out.answer, reader_score=out.score,
                 reader_null=out.null_score, passage_index=out.passage_index,
                 enc=enc, tif=tif, r=r)


def wald(p: float, n: int):
    se = math.sqrt(p * (1 - p) / n) if n else 0.0
    return se, max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def mcnemar(a_ok: Sequence[bool], b_ok: Sequence[bool]):
    """Exact two-sided McNemar. b = A right where B wrong, c = the reverse."""
    b = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    c = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    m = b + c
    if m == 0:
        return b, c, 1.0
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(min(b, c) + 1)) / (2 ** m))
    return b, c, p


def quantile_threshold(values: Sequence[float], rate: float) -> float:
    """Threshold t such that P(value < t) ~= rate.

    This is what equalises expansion budgets: the scalar gate fires on exactly
    the lowest-confidence `rate` fraction of questions.
    """
    if not values or rate <= 0:
        return float("-inf")
    if rate >= 1:
        return float("inf")
    v = sorted(values)
    idx = min(len(v) - 1, max(0, int(round(rate * len(v)))))
    return v[idx]


def main() -> None:
    args = parse_args()
    t_start = time.time()

    print("=" * 78)
    print("PHASE 3 / RQ3 -- I-gate vs scalar-gate at equal expansion budget")
    print("=" * 78)

    allex = load_hotpot_hf()

    # ---- Addendum 4: restrict the pool, then enforce disjointness -------
    excluded: set = set()
    if args.exclude_csv:
        with open(REPO / args.exclude_csv, encoding="utf-8") as fh:
            excluded = {r["qid"] for r in csv.DictReader(fh)}
        print(f"excluding  : {len(excluded)} qids from {args.exclude_csv}")
    pool = [e for e in allex
            if (args.qtype is None or e.qtype == args.qtype)
            and e.qid not in excluded]
    if args.qtype or excluded:
        print(f"pool       : {len(pool)} of {len(allex)} after qtype="
              f"{args.qtype!r} and exclusion")
        if len(pool) < args.n:
            raise SystemExit(
                f"ABORT: pool has {len(pool)} questions, need {args.n}")
    sample = stratified_sample(pool, args.n, seed=args.seed)
    if args.limit:
        sample = sample[:args.limit]

    # Assert disjointness LOUDLY rather than trusting the filter above.
    if excluded:
        overlap = {e.qid for e in sample} & excluded
        if overlap:
            raise SystemExit(
                f"ABORT: sample is NOT disjoint from {args.exclude_csv} -- "
                f"{len(overlap)} overlapping qid(s), first {sorted(overlap)[:5]}")
        print(f"disjoint   : CONFIRMED, 0 of {len(sample)} sampled qids appear "
              f"in {args.exclude_csv}")
    if args.qtype:
        wrong = {e.qtype for e in sample} - {args.qtype}
        if wrong:
            raise SystemExit(f"ABORT: sample contains qtype(s) {wrong}")
    print(f"sample     : {len(sample)}  seed={args.seed}  {type_counts(sample)}")
    print(f"k_base     : {args.k_base}   expansion schedule: {list(K_SCHEDULE)}")
    print(f"commit     : {git_commit()}")

    reader = ExtractiveReader()
    nli = HFNLIBackend()
    nli.predict("warm.", "warm.")
    # HotpotQA's own depth-derived targets (Decision 2). NOT the library
    # defaults, which are FEVER's and would pin r near 0.9 at every depth --
    # see PREREGISTRATION.md Addendum 1.
    scorer = NLIRetrievalScorer(nli=nli, support_target=SUPPORT_TARGET,
                                diversity_target=DIVERSITY_TARGET)

    # ---- pass 1: base stage for every question -------------------------
    print(f"\npass 1/2: scoring {len(sample)} questions at k={args.k_base} ...")
    base: Dict[str, Stage] = {}
    retrievers: Dict[str, Tuple[DenseRetriever, Dict[str, str]]] = {}
    recoverable: List[str] = []
    sufficient: List[str] = []

    for i, ex in enumerate(sample):
        r = DenseRetriever()
        r.index([Document(p.title, p.text) for p in ex.paragraphs])
        title_of = {p.text: p.title for p in ex.paragraphs}
        retrievers[ex.qid] = (r, title_of)
        st = score_stage(ex, args.k_base, r, reader, nli, scorer, title_of)
        base[ex.qid] = st
        if ex.gold_titles.issubset(set(st.titles)):
            sufficient.append(ex.qid)
        else:
            recoverable.append(ex.qid)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sample)}  ({time.time() - t_start:.0f}s)")

    n = len(sample)
    print(f"\nrecoverable-miss subset : {len(recoverable)}/{n} "
          f"({100 * len(recoverable) / n:.1f}%)")
    print(f"top-{args.k_base} already sufficient : {len(sufficient)}/{n} "
          f"({100 * len(sufficient) / n:.1f}%)")
    # (subset sizes are sample construction, not an outcome)
    if len(recoverable) < 40:
        print("\n*** STOP CONDITION: recoverable-miss subset is under 40 "
              "questions. The comparison has no power. ***")

    # ---- calibrate: (c)'s rate, then (b)'s thresholds -------------------
    i_fire = {q: base[q].i_dominates for q in base}
    rate_c = sum(i_fire.values()) / n
    if not args.no_analysis:
        print(f"\n(c) I-GATE expansion rate : {rate_c:.4f} "
              f"({sum(i_fire.values())}/{n})")

    thresholds: Dict[str, float] = {}
    scalar_fire: Dict[str, Dict[str, bool]] = {}
    for which in SCALARS:
        vals = [base[q].scalar(which) for q in base]
        t = quantile_threshold(vals, rate_c)
        thresholds[which] = t
        fires = {q: base[q].scalar(which) < t for q in base}
        scalar_fire[which] = fires
        if not args.no_analysis:
            rate_b = sum(fires.values()) / n
            flag = "MATCHED" if abs(rate_b - rate_c) <= 0.02 else "MISMATCH"
            print(f"(b) {which:12s} threshold {t:.4f} -> rate {rate_b:.4f} "
                  f"({sum(fires.values())}/{n})  [{flag}]")

    # ---- pass 2: expand where a gate fired ------------------------------
    need = {q for q in base if i_fire[q] or any(scalar_fire[w][q] for w in SCALARS)}
    print(f"\npass 2/2: expanding {len(need)} questions that at least one gate "
          f"fired on ...")
    expanded: Dict[str, List[Stage]] = {}
    for i, qid in enumerate(sorted(need)):
        ex = next(e for e in sample if e.qid == qid)
        r, title_of = retrievers[qid]
        stages = []
        for k in K_SCHEDULE:
            if k > len(ex.paragraphs):
                break
            stages.append(score_stage(ex, k, r, reader, nli, scorer, title_of))
        expanded[qid] = stages
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(need)}  ({time.time() - t_start:.0f}s)")

    # ---- resolve each condition ----------------------------------------
    def resolve(qid: str, fires: Optional[Dict[str, bool]],
                gate: str) -> Tuple[Stage, int]:
        """Final stage for a condition, plus expansions used.

        A gated condition re-scores after each expansion and stops as soon as
        its own trigger stops firing, or when the schedule is exhausted.
        """
        st = base[qid]
        if fires is None or not fires[qid]:
            return st, 0
        used = 0
        for nxt in expanded.get(qid, []):
            st, used = nxt, used + 1
            still = (nxt.i_dominates if gate == "i"
                     else nxt.scalar(gate) < thresholds[gate])
            if not still:
                break
        return st, used

    conditions: Dict[str, Dict[str, Tuple[Stage, int]]] = {
        "a_fixed_k": {q: resolve(q, None, "none") for q in base},
        "c_i_gate": {q: resolve(q, i_fire, "i") for q in base},
    }
    for which in SCALARS:
        conditions[f"b_scalar_{which}"] = {
            q: resolve(q, scalar_fire[which], which) for q in base}

    gold = {e.qid: e.answer for e in sample}
    qtype = {e.qid: e.qtype for e in sample}
    gold_titles = {e.qid: e.gold_titles for e in sample}

    rows = []
    for cond, res in conditions.items():
        for q, (st, used) in res.items():
            rows.append({
                "condition": cond, "qid": q, "qtype": qtype[q],
                "in_recoverable": q in recoverable,
                "expansions_used": used, "final_k": st.k,
                "answer": st.answer, "gold": gold[q],
                "em": int(exact_match(st.answer, gold[q])),
                "f1": round(token_f1(st.answer, gold[q]), 6),
                "gold_in_context": int(gold_titles[q].issubset(set(st.titles))),
                "reader_score": round(st.reader_score, 6),
                "T": round(st.tif.T, 6), "I": round(st.tif.I, 6),
                "F": round(st.tif.F, 6), "r": round(st.r, 6),
                "scalar_max_enc": round(st.scalar("max_enc"), 6),
            })

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{args.out_prefix}.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    meta = {"commit": git_commit(), "n": n, "k_base": args.k_base,
            "k_schedule": list(K_SCHEDULE), "seed": args.seed,
            "rate_c": rate_c, "thresholds": thresholds,
            "n_recoverable": len(recoverable), "n_sufficient": len(sufficient),
            "recoverable_qids": recoverable}
    (outdir / f"{args.out_prefix}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    if args.no_analysis:
        print(f"\nno-analysis mode: wrote the CSV and computed no accuracy.")
        print(f"wall-clock: {(time.time() - t_start) / 60:.1f} min")
        return

    # ---- report ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    hdr = (f"{'condition':22s} {'exp.rate':>9s} {'recov':>8s} {'unnec':>8s} "
           f"{'EM':>8s} {'F1':>8s}")
    print(hdr)
    print("-" * 78)
    summary = {}
    for cond, res in conditions.items():
        rate = statistics.mean(1 if u else 0 for _s, u in res.values())
        rec = [res[q] for q in recoverable]
        suf = [res[q] for q in sufficient]
        recov = statistics.mean(
            gold_titles[q].issubset(set(res[q][0].titles)) for q in recoverable
        ) if recoverable else 0.0
        unnec = statistics.mean(1 if res[q][1] else 0 for q in sufficient) if sufficient else 0.0
        em = statistics.mean(exact_match(s.answer, gold[q])
                             for q, (s, _u) in res.items())
        f1 = statistics.mean(token_f1(s.answer, gold[q])
                             for q, (s, _u) in res.items())
        em_rec = statistics.mean(exact_match(res[q][0].answer, gold[q])
                                 for q in recoverable) if recoverable else 0.0
        summary[cond] = {"rate": rate, "recov": recov, "unnec": unnec,
                         "em": em, "f1": f1, "em_rec": em_rec}
        print(f"{cond:22s} {rate:>9.4f} {recov:>8.4f} {unnec:>8.4f} "
              f"{em:>8.4f} {f1:>8.4f}")
    print("-" * 78)
    print("  exp.rate = expansion rate | recov = gold recovered into context on")
    print("  the recoverable-miss subset | unnec = expanded although top-k was")
    print("  already sufficient | EM/F1 = end to end, whole sample")

    print(f"\n{'condition':22s} {'EM on recoverable-miss subset':>32s}")
    for cond in conditions:
        e = summary[cond]["em_rec"]
        _se, lo, hi = wald(e, len(recoverable))
        print(f"{cond:22s} {e:>12.4f}  95% CI [{lo:.3f}, {hi:.3f}]")

    # ---- the gate: (c) vs (a) and (c) vs (b) ---------------------------
    print("\n" + "=" * 78)
    print("EXACT McNEMAR -- end-to-end EM, whole sample")
    print("=" * 78)
    ok = {cond: [exact_match(res[q][0].answer, gold[q]) for q in base]
          for cond, res in conditions.items()}
    for other in ["a_fixed_k"] + [f"b_scalar_{w}" for w in SCALARS]:
        b, c, p = mcnemar(ok["c_i_gate"], ok[other])
        verdict = ("c_i_gate WINS" if b > c and p < 0.05 else
                   "c_i_gate LOSES" if c > b and p < 0.05 else
                   "no significant difference")
        print(f"  c_i_gate vs {other:20s} b={b:3d} c={c:3d} p={p:.4f}  {verdict}")
        print(f"      (b = I-gate right where {other} wrong; c = the reverse)")

    print("\n  Same, restricted to the recoverable-miss subset:")
    okr = {cond: [exact_match(res[q][0].answer, gold[q]) for q in recoverable]
           for cond, res in conditions.items()}
    for other in ["a_fixed_k"] + [f"b_scalar_{w}" for w in SCALARS]:
        b, c, p = mcnemar(okr["c_i_gate"], okr[other])
        print(f"  c_i_gate vs {other:20s} b={b:3d} c={c:3d} p={p:.4f}")

    # ---- by question type ----------------------------------------------
    print("\n" + "=" * 78)
    print("BY QUESTION TYPE -- end-to-end EM")
    print("=" * 78)
    types = sorted({qtype[q] for q in base})
    print(f"{'condition':22s} " + " ".join(f"{t:>14s}" for t in types))
    for cond, res in conditions.items():
        cells = []
        for t in types:
            qs = [q for q in base if qtype[q] == t]
            cells.append(statistics.mean(
                exact_match(res[q][0].answer, gold[q]) for q in qs))
        print(f"{cond:22s} " + " ".join(f"{c:>14.4f}" for c in cells))

    print(f"\nwrote {path}")
    print(f"wall-clock: {(time.time() - t_start) / 60:.1f} min")

    # ---- the pre-registered verdict ------------------------------------
    print("\n" + "=" * 78)
    print("PRE-REGISTERED GATE")
    print("=" * 78)
    prim = "b_scalar_max_enc"
    b, c, p = mcnemar(ok["c_i_gate"], ok[prim])
    print(f"  (c) I-GATE EM          : {summary['c_i_gate']['em']:.4f}")
    print(f"  (b) SCALAR-GATE EM     : {summary[prim]['em']:.4f}  "
          f"[pre-registered scalar: max(e,n,c)]")
    print(f"  (a) FIXED-K EM         : {summary['a_fixed_k']['em']:.4f}")
    print(f"  expansion rates        : c={summary['c_i_gate']['rate']:.4f}  "
          f"b={summary[prim]['rate']:.4f}")
    print(f"  McNemar (c) vs (b)     : b={b} c={c} p={p:.4f}")
    beat = summary["c_i_gate"]["em"] > summary[prim]["em"]
    print(f"\n  Did (c) beat (b)? {'YES' if beat else 'NO'}"
          f"{' (significantly)' if beat and p < 0.05 else ''}")
    if not beat:
        print("  -> STOP CONDITION FIRED. Conditioning on I adds nothing over")
        print("     conditioning on low confidence generally. Do not proceed to")
        print("     Phases 4-5; do not adjust thresholds.")


if __name__ == "__main__":
    main()
