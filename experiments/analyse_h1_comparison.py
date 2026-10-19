"""Addendum 4 / H1 -- the single pre-registered analysis pass. Run ONCE.

H1, registered before the sample was drawn (PREREGISTRATION.md Addendum 4):

    On HotpotQA comparison questions, fixed k=3 retrieval achieves HIGHER
    exact match than indeterminacy-gated adaptive expansion.

Directional, so the primary test is ONE-SIDED in the registered direction. The
direction was registered because it had already been observed in Phase 3's
exploratory by-type split; this is a confirmation test of a stated direction.

Everything reported here was specified in advance:

  primary      exact one-sided McNemar, a_fixed_k vs c_i_gate
  secondary    the other three gated conditions vs a_fixed_k, Holm-corrected
               across those three at family alpha = 0.05
  robustness   the primary test restricted to span-answerable questions
  uniformity   direction count and pooled b/c, DESCRIPTIVE, no p-value, because
               the four conditions share a reader, a retriever and most of
               their decisions and are therefore heavily dependent
  diagnostics  the four firing rates, the base-depth r distribution, and
               whether r = 0.3333*coverage + 0.2000 still holds -- replication
               of the Phase 3 diagnosis on fresh data, not hypothesis tests

    python experiments/analyse_h1_comparison.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CSV = REPO / "results" / "hotpot_gate_h1.csv"
META = REPO / "results" / "hotpot_gate_h1_meta.json"
PHASE3 = REPO / "results" / "hotpot_gate.csv"
BASE = "a_fixed_k"
PRIMARY = "c_i_gate"
#: r = w*coverage + offset, the Phase 3 reduction being replicated.
AFFINE_W, AFFINE_OFFSET = 1.0 / 3.0, 0.2


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def mcnemar_counts(gated_ok, base_ok):
    """b = gated right where fixed-k wrong; c = fixed-k right where gated wrong.

    H1 predicts c > b.
    """
    b = sum(1 for g, f in zip(gated_ok, base_ok) if g and not f)
    c = sum(1 for g, f in zip(gated_ok, base_ok) if f and not g)
    return b, c


def mcnemar_one_sided(b: int, c: int) -> float:
    """P(X <= b) under X ~ Binomial(b+c, 0.5): the chance of seeing this few
    gated wins if the two conditions were equally good. Small p supports H1."""
    m = b + c
    if m == 0:
        return 1.0
    return min(1.0, sum(math.comb(m, i) for i in range(b + 1)) / 2 ** m)


def mcnemar_two_sided(b: int, c: int) -> float:
    m = b + c
    if m == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(m, i)
                            for i in range(min(b, c) + 1)) / 2 ** m)


def wald(p: float, n: int):
    se = math.sqrt(p * (1 - p) / n) if n else 0.0
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def holm(pairs):
    """Holm-Bonferroni. pairs = [(label, p), ...]. Returns in input order."""
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    m = len(pairs)
    out = {}
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pairs[i][1])
        running = max(running, adj)      # enforce monotonicity
        out[pairs[i][0]] = running
    return out


def is_yes_no(gold: str) -> bool:
    return gold.strip().lower() in ("yes", "no")


def main() -> None:
    if not CSV.exists():
        print(f"missing {CSV}")
        return
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    by = defaultdict(dict)
    for r in rows:
        by[r["condition"]][r["qid"]] = r
    conds = sorted(by)
    gated = [c for c in conds if c != BASE]
    qids = sorted(by[BASE])
    n = len(qids)

    print("=" * 78)
    print("ADDENDUM 4 / H1 -- pre-registered confirmation test")
    print("=" * 78)
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    print(f"realised n   : {n}   seed {meta.get('seed')}   "
          f"k_base {meta.get('k_base')}   ladder {meta.get('k_schedule')}")
    print(f"analysis at  : commit {commit()}")

    # disjointness, re-verified here rather than trusted from the run
    if PHASE3.exists():
        p3 = {r["qid"] for r in csv.DictReader(open(PHASE3, encoding="utf-8"))}
        overlap = set(qids) & p3
        print(f"disjoint from Phase 3 : {'CONFIRMED' if not overlap else 'VIOLATED'}"
              f" ({len(overlap)} overlapping qid(s) of {len(p3)} Phase 3 qids)")
    types = {r["qtype"] for r in rows}
    print(f"question types in sample : {sorted(types)}")

    ok = {c: [int(by[c][q]["em"]) == 1 for q in qids] for c in conds}
    em = {c: statistics.mean(int(by[c][q]["em"]) for q in qids) for c in conds}
    f1 = {c: statistics.mean(float(by[c][q]["f1"]) for q in qids) for c in conds}

    # ---------------------------------------------------------------- EM
    print("\n" + "-" * 78)
    print("Exact match and token-F1, all five conditions")
    print("-" * 78)
    print(f"  {'condition':24} {'EM':>8} {'95% CI':>16} {'F1':>8} {'d vs fixed-k':>13}")
    for c in [BASE] + gated:
        lo, hi = wald(em[c], n)
        d = "" if c == BASE else f"{em[c] - em[BASE]:+.4f}"
        print(f"  {c:24} {em[c]:>8.4f} {f'[{lo:.3f}, {hi:.3f}]':>16} "
              f"{f1[c]:>8.4f} {d:>13}")

    # ------------------------------------------------------------ PRIMARY
    print("\n" + "=" * 78)
    print("PRIMARY TEST (pre-registered): a_fixed_k vs c_i_gate, ONE-SIDED")
    print("=" * 78)
    b, c = mcnemar_counts(ok[PRIMARY], ok[BASE])
    p1 = mcnemar_one_sided(b, c)
    p2 = mcnemar_two_sided(b, c)
    print(f"  fixed k=3 EM   {em[BASE]:.4f}")
    print(f"  I-gate EM      {em[PRIMARY]:.4f}   difference "
          f"{em[PRIMARY] - em[BASE]:+.4f}")
    print(f"  discordant pairs {b + c}/{n} ({100 * (b + c) / n:.2f}%)   "
          f"b={b} (I-gate only right)   c={c} (fixed-k only right)")
    print(f"  one-sided exact McNemar p = {p1:.4f}      "
          f"(two-sided, for reference, {p2:.4f})")
    supported = em[BASE] > em[PRIMARY] and p1 < 0.05
    direction_held = em[BASE] > em[PRIMARY]
    print()
    if supported:
        print("  >>> H1 SUPPORTED: fixed k=3 beats the I-gate, p < 0.05 one-sided.")
    elif direction_held:
        print("  >>> H1 NOT SUPPORTED: direction held but the test is NULL "
              f"(p = {p1:.4f}).")
    else:
        print("  >>> H1 FALSIFIED: the direction REVERSED -- the I-gate scored "
              "at or above fixed k=3.")

    # ---------------------------------------------------------- SECONDARY
    print("\n" + "=" * 78)
    print("SECONDARY TESTS (pre-registered): the other three, Holm-corrected")
    print("=" * 78)
    sec = [c for c in gated if c != PRIMARY]
    raw = []
    counts = {}
    for cond in sec:
        bb, cc = mcnemar_counts(ok[cond], ok[BASE])
        counts[cond] = (bb, cc)
        raw.append((cond, mcnemar_one_sided(bb, cc)))
    adj = holm(raw)
    print(f"  {'condition':24} {'EM':>8} {'b':>4} {'c':>4} {'p (1-sided)':>12} "
          f"{'Holm-adj':>10} {'sig@.05':>8}")
    for cond, p in raw:
        bb, cc = counts[cond]
        print(f"  {cond:24} {em[cond]:>8.4f} {bb:>4} {cc:>4} {p:>12.4f} "
              f"{adj[cond]:>10.4f} {'yes' if adj[cond] < 0.05 else 'no':>8}")

    # -------------------------------------------------------- ROBUSTNESS
    print("\n" + "=" * 78)
    print("ROBUSTNESS (pre-registered): span-answerable questions only")
    print("=" * 78)
    span = [q for q in qids if not is_yes_no(by[BASE][q]["gold"])]
    yn = n - len(span)
    print(f"  yes/no answers excluded : {yn}/{n} ({100 * yn / n:.1f}%)")
    print(f"  span-answerable n       : {len(span)}")
    if span:
        sok = {c: [int(by[c][q]["em"]) == 1 for q in span] for c in conds}
        sem = {c: statistics.mean(int(by[c][q]["em"]) for q in span) for c in conds}
        sb, sc = mcnemar_counts(sok[PRIMARY], sok[BASE])
        sp = mcnemar_one_sided(sb, sc)
        print(f"\n  fixed k=3 EM {sem[BASE]:.4f}   I-gate EM {sem[PRIMARY]:.4f}   "
              f"difference {sem[PRIMARY] - sem[BASE]:+.4f}")
        print(f"  discordant {sb + sc}/{len(span)}   b={sb} c={sc}   "
              f"one-sided p = {sp:.4f}")
        agree = (sem[BASE] > sem[PRIMARY]) == direction_held
        print(f"  direction agrees with the primary test: {agree}")
        print("\n  Yes/no answers are unreachable by a span reader, so they can "
              "only\n  contribute concordant ties; excluding them concentrates "
              "the\n  discordant pairs that carry the signal.")

    # --------------------------------------------------------- UNIFORMITY
    print("\n" + "=" * 78)
    print("UNIFORMITY (pre-registered, DESCRIPTIVE -- no p-value)")
    print("=" * 78)
    below = [c for c in gated if em[c] < em[BASE]]
    tb = tc = 0
    for cond in gated:
        bb, cc = mcnemar_counts(ok[cond], ok[BASE])
        tb += bb
        tc += cc
    print(f"  gated conditions scoring BELOW fixed k=3 : {len(below)}/4")
    for cond in gated:
        mark = "below" if em[cond] < em[BASE] else "at or above"
        print(f"    {cond:24} {em[cond]:.4f}  ({mark})")
    print(f"\n  pooled across the four: b={tb} (gated only right)  "
          f"c={tc} (fixed-k only right)")
    print("  Reported as a direction count, NOT as a test: the four conditions")
    print("  share a reader, a retriever and most of their decisions, so a")
    print("  combined p-value would overstate the evidence.")

    # -------------------------------------------------------- DIAGNOSTICS
    print("\n" + "=" * 78)
    print("DIAGNOSTIC REPLICATION ON FRESH DATA (not hypothesis tests)")
    print("=" * 78)
    print("  firing rates")
    for cond in gated:
        rate = statistics.mean(
            1 if int(by[cond][q]["expansions_used"]) > 0 else 0 for q in qids)
        mk = statistics.mean(int(by[cond][q]["final_k"]) for q in qids)
        print(f"    {cond:24} {rate:.4f}   mean final k {mk:.2f}")
    print(f"  (Phase 3 comparison-arm I-gate rate was 0.7975)")

    rb = [float(by[BASE][q]["r"]) for q in qids]
    over = sum(1 for v in rb if v > 0.5)
    print(f"\n  base-depth r: min {min(rb):.4f} max {max(rb):.4f} "
          f"mean {statistics.mean(rb):.4f} var {statistics.pvariance(rb):.6f}")
    print(f"    r > 0.5 on {over}/{n} ({100 * over / n:.1f}%)   "
          f"distinct values {len({round(v, 9) for v in rb})}")
    print(f"  (Phase 3 whole-sample: mean 0.4546, r>0.5 on 59/400 = 14.8%)")

    # the affine law, with coverage computed independently (NLI-free)
    print(f"\n  replicating r = {AFFINE_W:.4f}*coverage + {AFFINE_OFFSET:.4f} ...")
    from neutrorag.hotpot import load_hotpot_hf
    from neutrorag.retrieval import DenseRetriever, Document
    from neutrorag.sufficiency import retrieval_sufficiency
    from neutrorag.hotpot import DIVERSITY_TARGET, SUPPORT_TARGET

    ex = {e.qid: e for e in load_hotpot_hf() if e.qid in set(qids)}
    worst, checked, const = 0.0, 0, defaultdict(set)
    for q in qids:
        e = ex.get(q)
        if e is None:
            continue
        rt = DenseRetriever()
        rt.index([Document(p.title, p.text) for p in e.paragraphs])
        got = rt.retrieve(e.question, k=int(meta.get("k_base", 3)))
        tof = {p.text: p.title for p in e.paragraphs}
        hyp = f"{e.question} {by[BASE][q]['answer']}".strip()
        bd = retrieval_sufficiency(
            hyp, passages=[g.text for g in got],
            relevances=[g.relevance for g in got],
            sources=[tof[g.text] for g in got],
            support_target=SUPPORT_TARGET, diversity_target=DIVERSITY_TARGET)
        for k, v in bd.components.items():
            const[k].add(round(v, 9))
        pred = AFFINE_W * bd.components.get("coverage", 0.0) + AFFINE_OFFSET
        worst = max(worst, abs(pred - bd.r))
        checked += 1
    print(f"    checked {checked} questions; max |predicted - actual r| = {worst:.2e}")
    for k in sorted(const):
        vals = const[k]
        print(f"    {k:16} {len(vals)} distinct value(s)"
              + (f"  (pinned at {next(iter(vals)):.4f})" if len(vals) == 1 else ""))
    print(f"    holds to machine precision: {worst < 1e-9}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  H1: fixed k=3 > I-gate on comparison-question exact match")
    print(f"  direction held : {direction_held}")
    print(f"  one-sided p    : {p1:.4f}")
    print(f"  H1 {'SUPPORTED' if supported else 'NOT SUPPORTED'}"
          f"{' (direction reversed -- FALSIFIED)' if not direction_held else ''}")


if __name__ == "__main__":
    main()
