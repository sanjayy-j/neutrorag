"""Addendum 6 conditional secondary: the paired AUROC comparison. One pass.

Same 400 questions, same retriever, same reader, same r formula, same base
depth. One thing changed: whether retrieval draws from the pooled
19,294-passage index or from the question's own 10 supplied paragraphs.

Reports AUROC of legacy r for em under both settings and a DeLong test for the
two correlated ROC curves ON THE SAME CASES, which is what makes this paired and
what the unpaired reference comparison could not do.

    python experiments/analyse_h3_paired.py
"""
from __future__ import annotations

import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

RES = REPO / "results"
POOLED = RES / "hotpot_gate_h3.csv"
PERQ = RES / "h3_paired_perquestion.csv"
BASE = "a_fixed_k"


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def main() -> None:
    from analyse_rq2_auroc import auc_ci, delong_test

    if not PERQ.exists():
        raise SystemExit(f"ABORT: {PERQ} missing; run "
                         "experiments/run_h3_paired_perquestion.py first")

    pooled = {r["qid"]: r for r in csv.DictReader(open(POOLED, encoding="utf-8"))
              if r["condition"] == BASE}
    perq = {r["qid"]: r for r in csv.DictReader(open(PERQ, encoding="utf-8"))}
    qids = sorted(set(pooled) & set(perq))

    print("=" * 78)
    print("ADDENDUM 6 conditional secondary -- PAIRED comparison")
    print("=" * 78)
    print(f"commit {commit()}")
    print(f"questions in both settings: {len(qids)}   "
          f"(pooled {len(pooled)}, per-question {len(perq)})")
    if len(qids) != len(pooled) or len(qids) != len(perq):
        print("  WARNING: the two sets are not identical; only the "
              "intersection is used")

    print("\n" + "-" * 78)
    print("How the two settings differ")
    print("-" * 78)
    print(f"  {'setting':20} {'EM':>8} {'gold@k=3':>10} {'mean r':>9} "
          f"{'r var':>9}")
    for label, src, keys in (
        ("pooled 19,294", pooled, ("em", "gold_in_context", "r")),
        ("per-question 10", perq, ("em", "gold_in_context", "r")),
    ):
        em = statistics.mean(int(src[q][keys[0]]) for q in qids)
        gic = statistics.mean(int(src[q][keys[1]]) for q in qids)
        rs = [float(src[q][keys[2]]) for q in qids]
        print(f"  {label:20} {em:>8.4f} {gic:>10.4f} "
              f"{statistics.mean(rs):>9.4f} {statistics.pvariance(rs):>9.6f}")
    flips = sum(1 for q in qids
                if int(pooled[q]["em"]) != int(perq[q]["em"]))
    print(f"\n  questions whose em differs between settings: {flips}/{len(qids)}")

    print("\n" + "=" * 78)
    print("PAIRED AUROC of legacy r for ANSWER CORRECTNESS (em)")
    print("=" * 78)
    results = {}
    for label, src in (("pooled (hard retrieval)", pooled),
                       ("per-question (easy retrieval)", perq)):
        y = [int(src[q]["em"]) for q in qids]
        s = [float(src[q]["r"]) for q in qids]
        if not (0 < sum(y) < len(y)):
            print(f"  {label}: DEGENERATE, one class empty")
            continue
        a, se, lo, hi, _m, _n = auc_ci(s, y)
        results[label] = (a, s, y)
        verdict = ("above 0.5" if lo > 0.5 else
                   "INVERTED" if hi < 0.5 else "not distinguishable from 0.5")
        print(f"  {label:30} AUROC {a:.4f}  SE {se:.4f}  "
              f"95% CI [{lo:.4f}, {hi:.4f}]   {verdict}")
        print(f"  {'':30} ({sum(y)} positives of {len(y)})")

    if len(results) == 2:
        (la, (aa, sa, ya)), (lb, (ab, sb, _yb)) = list(results.items())
        # same cases, same label vector -- the labels DIFFER between settings
        # because em differs, so a paired DeLong on one shared label is not
        # available; test each predictor against the other on ITS OWN label is
        # not paired either. Report the honest version: DeLong on the shared
        # subset where the label agrees, plus the raw difference.
        print("\n" + "-" * 78)
        print("DeLong, two correlated ROC curves")
        print("-" * 78)
        print(f"  raw difference in AUROC: {aa - ab:+.4f} "
              f"({la} minus {lb})")
        agree = [q for q in qids
                 if int(pooled[q]["em"]) == int(perq[q]["em"])]
        ya2 = [int(pooled[q]["em"]) for q in agree]
        if 0 < sum(ya2) < len(ya2):
            z, pv = delong_test([float(pooled[q]["r"]) for q in agree],
                                [float(perq[q]["r"]) for q in agree], ya2)
            a1, *_ = auc_ci([float(pooled[q]["r"]) for q in agree], ya2)
            a2, *_ = auc_ci([float(perq[q]["r"]) for q in agree], ya2)
            sig = "SIGNIFICANT" if pv == pv and pv < 0.05 else "not significant"
            print(f"\n  On the {len(agree)} questions where em AGREES between")
            print("  settings, so a single shared label exists and the two")
            print("  curves are genuinely on the same cases:")
            print(f"    pooled r AUROC {a1:.4f}   per-question r AUROC {a2:.4f}")
            print(f"    difference {a1 - a2:+.4f}   z = {z:.3f}   p = {pv:.4f}"
                  f"   [{sig}]")
            print(f"\n  The {flips} questions whose em flips between settings are")
            print("  excluded from that test by necessity: they have no single")
            print("  label, since answer correctness is what the retrieval")
            print("  change moved. Their exclusion is stated, not silent.")
        else:
            print("\n  DeLong not computable: the agreeing subset is "
                  "single-class.")

    print("\n" + "=" * 78)
    print("READ THIS WITH THE RESULT")
    print("=" * 78)
    print("  This comparison is paired on QUESTIONS, which removes the sample")
    print("  variation that confounded the unpaired references (0.5483 Phase 3,")
    print("  0.4689 H1, both different samples). It is NOT paired on the label:")
    print("  em itself changes with retrieval, which is the intended effect, so")
    print("  the two AUROCs are computed against different label vectors and")
    print("  the headline difference is descriptive. The DeLong test above is")
    print("  restricted to the questions where the label is shared.")


if __name__ == "__main__":
    main()
