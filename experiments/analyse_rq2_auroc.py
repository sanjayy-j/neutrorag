"""RQ2 as registered at the proposal: AUROC of r for answerability.

Offline. Reads committed CSVs only -- no NLI model, no reader, no retriever, no
new run.

The proposal registered "AUROC of r for answerability". RESULTS.md has until now
substituted a Spearman rank correlation against Recall@3 and said so. This
computes the registered quantity.

LABELS (binary, per question, taken from the a_fixed_k rows at k=3, which never
expand so their rows ARE the base depth). They answer different questions and
are reported separately, never pooled:

  gold_in_context   retrieval success -- were BOTH gold paragraphs inside the
                    top 3? Verified identical to (Recall@3 == 1.0) on 400/400
                    Phase 3 rows.
  em                reader success -- did the extractive reader produce the
                    exact gold answer from that context? A strictly harder
                    label: it can fail with perfect retrieval.

PREDICTORS at base depth:

  r               retrieval sufficiency as the SYSTEM computed it, i.e. on the
                  hypothesis (question + reader answer). This is the quantity
                  the gate actually read.
  r_question      the same formula with coverage computed on the question
                  alone. Non-circular -- the hypothesis contains the answer,
                  which is a span taken from the retrieved passages. Available
                  for the Phase 3 sample only.
  max_relevance   the retriever's own top cosine score. The reference r is
                  supposed to improve on. Available for the Phase 3 sample
                  only; recovering it for the H1 sample would require running
                  the embedder, which this analysis does not do.

Because r is a strictly increasing affine function of coverage, AUROC(r) equals
AUROC(coverage) exactly -- AUROC depends only on ranks.

SAMPLES, run separately and never pooled:
  Phase 3   n=400, mixed bridge/comparison
  H1        n=600, comparison only, disjoint from Phase 3

    python experiments/analyse_rq2_auroc.py
"""
from __future__ import annotations

import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RES = REPO / "results"
BASE = "a_fixed_k"


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


# ---------------------------------------------------------------------------
# DeLong: AUROC, its variance, and the test for two correlated curves
# ---------------------------------------------------------------------------
def _psi(x: float, y: float) -> float:
    """Placement kernel: 1 if the positive outranks the negative, 0.5 on a tie."""
    if x > y:
        return 1.0
    return 0.5 if x == y else 0.0


def delong_components(scores: Sequence[float], labels: Sequence[int]
                      ) -> Tuple[float, List[float], List[float]]:
    """Return (AUC, V10 over positives, V01 over negatives).

    V10_i is the fraction of negatives that positive i outranks; V01_j is the
    fraction of positives that negative j is outranked by. Both average to the
    AUC, and their variances give DeLong's variance estimate.
    """
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        raise ValueError("need both classes present")
    v10 = [sum(_psi(x, y) for y in neg) / n for x in pos]
    v01 = [sum(_psi(x, y) for x in pos) / m for y in neg]
    return sum(v10) / m, v10, v01


def auc_ci(scores, labels, alpha: float = 0.05):
    """AUROC with a DeLong Wald CI, clipped to [0, 1]."""
    auc, v10, v01 = delong_components(scores, labels)
    m, n = len(v10), len(v01)
    s10 = statistics.variance(v10) if m > 1 else 0.0
    s01 = statistics.variance(v01) if n > 1 else 0.0
    var = s10 / m + s01 / n
    se = math.sqrt(var)
    z = 1.959963984540054
    return auc, se, max(0.0, auc - z * se), min(1.0, auc + z * se), m, n


def delong_test(a_scores, b_scores, labels):
    """Two correlated ROC curves on the SAME cases. Returns (z, two-sided p)."""
    auc_a, v10a, v01a = delong_components(a_scores, labels)
    auc_b, v10b, v01b = delong_components(b_scores, labels)
    m, n = len(v10a), len(v01a)
    if m < 2 or n < 2:
        return float("nan"), float("nan")
    s10a, s10b = statistics.variance(v10a), statistics.variance(v10b)
    s01a, s01b = statistics.variance(v01a), statistics.variance(v01b)
    s10ab = statistics.covariance(v10a, v10b)
    s01ab = statistics.covariance(v01a, v01b)
    var_a = s10a / m + s01a / n
    var_b = s10b / m + s01b / n
    cov = s10ab / m + s01ab / n
    denom = var_a + var_b - 2 * cov
    if denom <= 0:
        return float("nan"), float("nan")
    z = (auc_a - auc_b) / math.sqrt(denom)
    from scipy import stats
    return z, float(2 * (1 - stats.norm.cdf(abs(z))))


# ---------------------------------------------------------------------------
def load_sample(gate_csv: Path, diag_csv: Path | None):
    """Per-question predictors and labels from the base-depth rows."""
    base = {r["qid"]: r for r in csv.DictReader(open(gate_csv, encoding="utf-8"))
            if r["condition"] == BASE}
    diag = {}
    if diag_csv and diag_csv.exists():
        diag = {r["qid"]: r
                for r in csv.DictReader(open(diag_csv, encoding="utf-8"))}
    rows = []
    for q, r in sorted(base.items()):
        rec = {"qid": q, "qtype": r["qtype"],
               "r": float(r["r"]),
               "gold_in_context": int(r["gold_in_context"]),
               "em": int(r["em"])}
        if q in diag:
            rec["r_question"] = float(diag[q]["r"])
            rec["max_relevance"] = float(diag[q]["max_relevance"])
        rows.append(rec)
    return rows


def report(name: str, rows, predictors: Sequence[str], labels: Sequence[str]):
    print("\n" + "=" * 78)
    print(f"{name}   n = {len(rows)}")
    print("=" * 78)
    avail = [p for p in predictors if p in rows[0]]
    missing = [p for p in predictors if p not in rows[0]]
    if missing:
        print(f"  predictors UNAVAILABLE offline: {missing}")
        print("    (max_relevance is the retriever's cosine score; recovering it")
        print("     for this sample would require running the embedder, which")
        print("     this analysis does not do)")

    for lab in labels:
        y = [r[lab] for r in rows]
        npos, nneg = sum(y), len(y) - sum(y)
        print(f"\n  label: {lab}   positives {npos}/{len(y)} "
              f"({100 * npos / len(y):.1f}%)   negatives {nneg}")
        if npos == 0 or nneg == 0:
            print("    DEGENERATE -- one class is empty, AUROC undefined")
            continue
        print(f"    {'predictor':16} {'AUROC':>7} {'SE':>7} {'95% CI':>18} "
              f"{'better than chance?':>20}")
        cache = {}
        for p in avail:
            s = [r[p] for r in rows]
            auc, se, lo, hi, _m, _n = auc_ci(s, y)
            cache[p] = s
            verdict = "yes" if lo > 0.5 else ("no" if hi >= 0.5 else "INVERTED")
            print(f"    {p:16} {auc:>7.4f} {se:>7.4f} "
                  f"{f'[{lo:.4f}, {hi:.4f}]':>18} {verdict:>20}")
        if "max_relevance" in cache:
            print("\n    DeLong, two correlated ROC curves on the same cases:")
            for p in [x for x in avail if x != "max_relevance"]:
                z, pv = delong_test(cache[p], cache["max_relevance"], y)
                a1, *_ = auc_ci(cache[p], y)
                a2, *_ = auc_ci(cache["max_relevance"], y)
                sig = "SIGNIFICANT" if pv == pv and pv < 0.05 else "not significant"
                print(f"      {p} ({a1:.4f}) vs max_relevance ({a2:.4f}): "
                      f"difference {a1 - a2:+.4f}, z = {z:.3f}, "
                      f"p = {pv:.4f}  [{sig}]")
        else:
            print("\n    DeLong comparison NOT COMPUTABLE offline for this "
                  "sample:\n      max_relevance is not in any committed artefact "
                  "for it.")


def main() -> None:
    print("=" * 78)
    print("RQ2 as registered at the proposal: AUROC of r for answerability")
    print("=" * 78)
    print(f"offline; committed CSVs only; analysis at commit {commit()}")
    print("\nSamples are reported SEPARATELY. The H1 sample is disjoint from the")
    print("Phase 3 sample, so it is independent confirmation, not a replication")
    print("of the same cases -- and the two are never pooled.")

    p3 = load_sample(RES / "hotpot_gate.csv", RES / "r_component_diagnostics.csv")
    h1 = load_sample(RES / "hotpot_gate_h1.csv", None)
    overlap = {r["qid"] for r in p3} & {r["qid"] for r in h1}
    print(f"\ndisjointness re-verified here: {len(overlap)} overlapping qid(s)")

    preds = ["r", "r_question", "max_relevance"]
    labs = ["gold_in_context", "em"]
    report("PHASE 3 SAMPLE (mixed bridge/comparison)", p3, preds, labs)
    report("H1 SAMPLE (comparison only, disjoint)", h1, preds, labs)

    print("\n" + "=" * 78)
    print("NOTES")
    print("=" * 78)
    print("  * r is a strictly increasing affine function of coverage, so")
    print("    AUROC(r) == AUROC(coverage) exactly. AUROC depends only on ranks.")
    print("  * r uses coverage computed on the HYPOTHESIS (question + the")
    print("    reader's answer), which is partly circular: the answer is a span")
    print("    taken from the retrieved passages. r_question is the")
    print("    non-circular variant and is the fairer test of the signal.")
    print("  * gold_in_context asks whether retrieval succeeded; em asks whether")
    print("    the reader then succeeded. The second is strictly harder and can")
    print("    fail with perfect retrieval.")


if __name__ == "__main__":
    main()
