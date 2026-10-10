"""The policy test on CONFLICTS in pairwise mode -- Task 4.

This is the experiment the project has never run. Everything before it graded
the SCORER's label attribution; the actual claim is that different failure modes
get different ACTIONS, and CONFLICTS' conflict-type annotations are the first
external ground truth for which action is right.

Replays the per-pair NLI cached by score_conflicts_pairwise.py, so nothing but
the named variable changes between conditions. `r` is recomputed from the FULL
retrieved set via the loader (the cache holds only the compared top-5), which is
what pairwise_breakdown does in production.

    python experiments/score_conflicts_pairwise.py
    python experiments/run_conflicts_pairwise_eval.py

Misinformation (5 instances) is excluded from every per-type claim, as are the
three types with no expected action, which are reported for context only.
T is CORROBORATION throughout and is labelled as such.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from make_results_table import wald_ci  # noqa: E402
from neutrorag.conflicts import (  # noqa: E402
    DIVERSITY_TARGET, SUPPORT_TARGET, TOO_FEW_TO_EVALUATE, TYPE_TO_EXPECTED_ACTION,
    load_conflicts,
)
from neutrorag.policy import Action, NeutroPolicy  # noqa: E402
from neutrorag.scoring import NLIRetrievalScorer  # noqa: E402

ACTIONS = [a.value for a in Action]


class _PairReplay:
    """Returns the cached (e, n, c) for a given (premise, hypothesis) text pair."""

    def __init__(self, table):
        self._t = table
        self.misses = 0

    def predict(self, premise, hypothesis):
        key = (premise, hypothesis)
        if key not in self._t:
            self.misses += 1
            return (0.0, 1.0, 0.0)
        return self._t[key]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="data/conflicts_pairwise.jsonl")
    p.add_argument("--data", default="data/conflicts.jsonl")
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def mann_whitney(a, b):
    """Two-sided Mann-Whitney U with a normal approximation; no scipy."""
    comb = sorted([(x, 0) for x in a] + [(x, 1) for x in b])
    ranks, i = {}, 0
    while i < len(comb):
        j = i
        while j + 1 < len(comb) and comb[j + 1][0] == comb[i][0]:
            j += 1
        rr = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = rr
        i = j + 1
    ra = sum(ranks[k] for k, (_, g) in enumerate(comb) if g == 0)
    na, nb = len(a), len(b)
    u = ra - na * (na + 1) / 2
    sd = math.sqrt(na * nb * (na + nb + 1) / 12)
    z = (u - na * nb / 2) / sd if sd else 0.0
    return u, z, math.erfc(abs(z) / math.sqrt(2))


def prf(pairs, labels):
    out = {}
    for lab in labels:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        out[lab] = (pr, rc, f1, tp + fn)
    return out


def main() -> None:
    args = parse_args()
    cache = REPO / args.cache
    if not cache.exists():
        print(f"missing {cache}; run experiments/score_conflicts_pairwise.py first")
        raise SystemExit(1)

    recs, header = [], None
    for line in open(cache, encoding="utf-8"):
        r = json.loads(line)
        if r.get("_header"):
            header = r
        else:
            recs.append(r)

    by_question = {x.question: x for x in load_conflicts(str(REPO / args.data))}

    scorer_kw = dict(support_target=SUPPORT_TARGET, diversity_target=DIVERSITY_TARGET)
    policy = NeutroPolicy()
    L = []

    def emit(s=""):
        print(s)
        L.append(s)

    emit("# CONFLICTS policy test -- PAIRWISE mode (Decision 1)")
    emit()
    emit(f"Cache `{args.cache}` | model `{header['model']}` | "
         f"max_passages {header['max_passages']} | sampling `{json.dumps(header['sampling'])}`")
    emit(f"`python experiments/run_conflicts_pairwise_eval.py` | commit `{git_commit()}`")
    emit()
    emit("**T is CORROBORATION in this mode** -- mutual agreement between sources, "
         "not support for a claim.")
    emit()

    rows, misses = [], 0
    for rec in recs:
        ex = by_question.get(rec["question"])
        if ex is None:
            continue
        table = {}
        ps = rec["passages"]
        for pr in rec["pairs"]:
            table[(ps[pr["i"]]["text"], ps[pr["j"]]["text"])] = (pr["e"], pr["n"], pr["c"])
        backend = _PairReplay(table)
        sc = NLIRetrievalScorer(nli=backend, **scorer_kw)
        bd = sc.pairwise_breakdown(ex.question, ex.evidence,
                                   max_passages=header["max_passages"])
        misses += backend.misses
        rows.append({"type": rec["conflict_type"], "bd": bd,
                     "action": policy.decide_once(bd.tif)[0].value})

    emit(f"N = {len(rows)} instances. Pair-table lookup misses: {misses} (expect 0).")
    emit()
    emit("| conflict type | n |")
    emit("|---|---:|")
    for t, k in Counter(r["type"] for r in rows).most_common():
        emit(f"| {t} | {k} |")
    emit()

    # ---------------- r ---------------------------------------------------
    emit("## r under the new targets, pairwise mode")
    emit()
    rs = sorted(r["bd"].r for r in rows)
    emit(f"- min {rs[0]:.3f}, median {rs[len(rs)//2]:.3f}, max {rs[-1]:.3f}; "
         f"instances at r <= 0.5: **{sum(1 for v in rs if v <= 0.5)}/{len(rs)}**")
    emit(f"- for comparison: FEVER claim mode median 0.750 (was 0.417); "
         f"CONFLICTS claim mode under old targets was median 1.000, pstdev 0.0297")
    emit()

    # ---------------- 4a: contradiction by type ---------------------------
    emit("## 4a. Inter-source contradiction by annotated conflict type")
    emit()
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["bd"])
    emit("| conflict type | n | median contradiction | mean | median corroboration | median I |")
    emit("|---|---:|---:|---:|---:|---:|")
    for t, bds in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        con = sorted(b.contradiction for b in bds)
        cor = sorted(b.corroboration for b in bds)
        ii = sorted(b.tif.I for b in bds)
        emit(f"| {t} | {len(bds)} | **{con[len(con)//2]:.3f}** | "
             f"{sum(con)/len(con):.3f} | {cor[len(cor)//2]:.3f} | {ii[len(ii)//2]:.3f} |")
    emit()

    a = [b.contradiction for b in by_type.get("Conflicting opinions and research outcomes", [])]
    b_ = [b.contradiction for b in by_type.get("No conflict", [])]
    if a and b_:
        u, z, p = mann_whitney(a, b_)
        ma, mb = sorted(a)[len(a)//2], sorted(b_)[len(b_)//2]
        emit(f"**Conflicting opinions (median {ma:.3f}, n={len(a)}) vs "
             f"No conflict (median {mb:.3f}, n={len(b_)})**: "
             f"Mann-Whitney U={u:.0f}, z={z:.2f}, two-sided p={p:.3e}. "
             f"Difference of medians {ma - mb:+.3f}.")
        emit()
        direction = "HIGHER on Conflicting opinions (as predicted)" if ma > mb else \
                    "HIGHER on No conflict (the REVERSE of the prediction)"
        sig = "significant" if p < 0.05 else "NOT significant"
        emit(f"**Verdict: contradiction is {direction}; the difference is {sig} at alpha=0.05.**")
        emit(f"Claim mode on the same corpus gave the reverse: 0.222 vs 0.556, p=2.14e-03.")
        emit()

    # ---------------- 4b: the policy test ---------------------------------
    emit("## 4b. The policy test -- actions vs external ground truth")
    emit()
    emit(f"Excluded: {sorted(TOO_FEW_TO_EVALUATE)} (5 instances, too few for any claim). "
         "Complementary information and Outdated information carry no expected action "
         "in the five-way space and are reported for context only.")
    emit()
    emit("### Action x conflict type")
    emit()
    emit("| conflict type | " + " | ".join(ACTIONS) + " | n |")
    emit("|---" * (len(ACTIONS) + 2) + "|")
    act_by_type = defaultdict(Counter)
    for r in rows:
        act_by_type[r["type"]][r["action"]] += 1
    for t, cnt in sorted(act_by_type.items(), key=lambda kv: -sum(kv[1].values())):
        emit(f"| {t} | " + " | ".join(str(cnt.get(a_, 0)) for a_ in ACTIONS) +
             f" | {sum(cnt.values())} |")
    emit()

    pairs = [(TYPE_TO_EXPECTED_ACTION[r["type"]], r["action"])
             for r in rows if r["type"] in TYPE_TO_EXPECTED_ACTION]
    if pairs:
        labels = ["surface_contradiction", "answer"]
        tbl = prf(pairs, labels)
        acc = sum(1 for g, p in pairs if g == p) / len(pairs)
        se, lo, hi = wald_ci(acc, len(pairs))
        emit(f"### Per-action, n={len(pairs)}")
        emit()
        emit(f"Exact-match accuracy **{acc:.3f}** (95% CI [{lo:.3f}, {hi:.3f}])")
        emit()
        emit("| expected action | precision | recall | F1 | support |")
        emit("|---|---:|---:|---:|---:|")
        for lab in labels:
            pr_, rc_, f1_, sup = tbl[lab]
            emit(f"| {lab} | {pr_:.3f} | {rc_:.3f} | {f1_:.3f} | {sup} |")
        emit()
        sc_conf = act_by_type["Conflicting opinions and research outcomes"]["surface_contradiction"]
        n_conf = sum(act_by_type["Conflicting opinions and research outcomes"].values())
        sc_noc = act_by_type["No conflict"]["surface_contradiction"]
        n_noc = sum(act_by_type["No conflict"].values())
        rate_c = sc_conf / n_conf if n_conf else 0.0
        rate_n = sc_noc / n_noc if n_noc else 0.0
        emit(f"**SURFACE_CONTRADICTION fires on {sc_conf}/{n_conf} ({rate_c:.1%}) of "
             f"Conflicting opinions vs {sc_noc}/{n_noc} ({rate_n:.1%}) of No conflict.**")
        emit()
        emit(f"**Verdict: it {'DOES' if rate_c > rate_n else 'does NOT'} fire "
             f"preferentially on Conflicting opinions.** "
             f"{'The central claim survives this test.' if rate_c > rate_n else 'The policy central claim is UNSUPPORTED on this evidence.'}")
        emit()

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "conflicts_pairwise_eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(outdir / "conflicts_pairwise.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["conflict_type", "expected_action", "action", "r",
                    "corroboration", "contradiction", "neutrality", "T", "I", "F"])
        for r in rows:
            b = r["bd"]
            w.writerow([r["type"], TYPE_TO_EXPECTED_ACTION.get(r["type"], ""), r["action"],
                        f"{b.r:.4f}", f"{b.corroboration:.4f}", f"{b.contradiction:.4f}",
                        f"{b.neutrality:.4f}", f"{b.tif.T:.4f}", f"{b.tif.I:.4f}",
                        f"{b.tif.F:.4f}"])
    print(f"\nwrote {outdir/'conflicts_pairwise_eval.md'} and {outdir/'conflicts_pairwise.csv'}")


if __name__ == "__main__":
    main()
