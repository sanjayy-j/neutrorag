"""Diagnostics for r on HotpotQA: is the problem composition, or inputs?

DIAGNOSTICS ONLY. Nothing here recalibrates r, changes a weight or a target, or
recomputes any accuracy figure. It measures what r and its components do at the
base retrieval depth and how each correlates with actual retrieval success.

    python experiments/diagnose_r_components.py

Cost: retrieval embeddings only. retrieval_sufficiency is NLI-free by
construction ("no argument is, or can be derived from, an NLI output"), so no
NLI model and no reader is loaded.

Ground truth for "retrieval success" is per-question Recall@3 against the gold
supporting paragraphs, taken from results/hotpot_retrieval.csv (Phase 2, same
sample and seed). Recall@3 is the right comparator because the gate runs at
k = 3.

The question being answered: r has no dynamic range (observed 0.311-0.533) and
no apparent relation to retrieval quality (0.008 spread across a 0.214 Recall@3
gap between question types). Either one good component is being diluted by flat
ones -- a composition problem, cheap to fix -- or no component tracks retrieval
at all, which is an inputs problem and is not.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neutrorag.hotpot import (  # noqa: E402
    DIVERSITY_TARGET, SUPPORT_TARGET, load_hotpot_hf, stratified_sample,
)
from neutrorag.retrieval import DenseRetriever, Document  # noqa: E402
from neutrorag.sufficiency import retrieval_sufficiency  # noqa: E402

K_BASE = 3
SAMPLE_SEED = 20261019
SAMPLE_N = 400
RETRIEVAL_CSV = REPO / "results" / "hotpot_retrieval.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=SAMPLE_N)
    p.add_argument("--seed", type=int, default=SAMPLE_SEED)
    p.add_argument("--k", type=int, default=K_BASE)
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def spearman(x: List[float], y: List[float]):
    """Spearman rho with a two-sided p-value. Returns (rho, p, note)."""
    from scipy import stats
    if len({*x}) < 2:
        return float("nan"), float("nan"), "x constant"
    if len({*y}) < 2:
        return float("nan"), float("nan"), "y constant"
    res = stats.spearmanr(x, y)
    return float(res.statistic), float(res.pvalue), ""


def describe(name: str, vals: List[float]) -> dict:
    n = len(vals)
    at0 = sum(1 for v in vals if v <= 1e-12)
    at1 = sum(1 for v in vals if v >= 1 - 1e-12)
    return {
        "component": name, "n": n,
        "min": min(vals), "max": max(vals),
        "range": max(vals) - min(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "variance": statistics.pvariance(vals),
        "stdev": statistics.pstdev(vals),
        "distinct": len({round(v, 9) for v in vals}),
        "at_0": at0, "at_1": at1,
        "sat_frac": (at0 + at1) / n,
    }


def main() -> None:
    args = parse_args()
    print("=" * 78)
    print("r COMPONENT DIAGNOSTICS -- no recalibration, no accuracy recomputed")
    print("=" * 78)
    print(f"k = {args.k}   targets: support={SUPPORT_TARGET} "
          f"diversity={DIVERSITY_TARGET}   commit {git_commit()}")

    if not RETRIEVAL_CSV.exists():
        print(f"missing {RETRIEVAL_CSV}; run run_hotpot_retrieval.py first")
        return
    recall = {r["qid"]: float(r[f"recall@{args.k}"])
              for r in csv.DictReader(open(RETRIEVAL_CSV, encoding="utf-8"))}

    sample = stratified_sample(load_hotpot_hf(), args.n, seed=args.seed)
    print(f"sample: {len(sample)}   joined on Recall@{args.k} from Phase 2\n")

    rows: List[dict] = []
    for ex in sample:
        if ex.qid not in recall:
            continue
        r = DenseRetriever()
        r.index([Document(p.title, p.text) for p in ex.paragraphs])
        got = r.retrieve(ex.question, k=args.k)
        title_of = {p.text: p.title for p in ex.paragraphs}
        bd = retrieval_sufficiency(
            ex.question,
            passages=[g.text for g in got],
            relevances=[g.relevance for g in got],
            sources=[title_of[g.text] for g in got],
            support_target=SUPPORT_TARGET,
            diversity_target=DIVERSITY_TARGET,
        )
        row = {"qid": ex.qid, "qtype": ex.qtype, "r": bd.r,
               "recall_at_k": recall[ex.qid],
               "max_relevance": max((g.relevance for g in got), default=0.0)}
        row.update(bd.components)
        rows.append(row)

    present = [c for c in ("coverage", "support_count", "connectivity", "diversity")
               if c in rows[0]]
    missing = [c for c in ("coverage", "support_count", "connectivity", "diversity")
               if c not in rows[0]]
    print(f"components PRESENT : {present}")
    print(f"components DROPPED : {missing}")
    print("  (sufficiency.py drops an unavailable component and renormalises the")
    print("   remaining weights, rather than scoring it 0.)\n")

    # ---- distributions ---------------------------------------------------
    print("-" * 78)
    print("DISTRIBUTION of each component at base depth")
    print("-" * 78)
    print(f"  {'component':16} {'min':>7} {'max':>7} {'range':>7} {'mean':>7} "
          f"{'var':>9} {'distinct':>9} {'at 0':>6} {'at 1':>6}")
    for name in present + ["r"]:
        d = describe(name, [x[name] for x in rows])
        print(f"  {d['component']:16} {d['min']:>7.4f} {d['max']:>7.4f} "
              f"{d['range']:>7.4f} {d['mean']:>7.4f} {d['variance']:>9.6f} "
              f"{d['distinct']:>9} {d['at_0']:>6} {d['at_1']:>6}")
    d = describe("recall_at_k", [x["recall_at_k"] for x in rows])
    print(f"  {'[Recall@' + str(args.k) + ']':16} {d['min']:>7.4f} {d['max']:>7.4f} "
          f"{d['range']:>7.4f} {d['mean']:>7.4f} {d['variance']:>9.6f} "
          f"{d['distinct']:>9} {d['at_0']:>6} {d['at_1']:>6}")

    # ---- correlations ----------------------------------------------------
    types = sorted({x["qtype"] for x in rows})
    print("\n" + "-" * 78)
    print(f"SPEARMAN rho vs per-question Recall@{args.k}")
    print("-" * 78)
    groups = [("pooled", rows)] + [(t, [x for x in rows if x["qtype"] == t])
                                   for t in types]
    header = f"  {'signal':16}" + "".join(f" {g[0] + ' rho':>14} {'p':>9}"
                                          for g in groups)
    print(header)
    for name in present + ["r", "max_relevance"]:
        line = f"  {name:16}"
        for _lbl, sub in groups:
            rho, p, note = spearman([x[name] for x in sub],
                                    [x["recall_at_k"] for x in sub])
            cell = f"{rho:>14.4f}" if rho == rho else f"{note:>14}"
            pc = f"{p:>9.4f}" if p == p else f"{'-':>9}"
            line += cell + pc
        print(line)
    print("\n  max_relevance is the raw retriever score, included as the")
    print("  reference signal r is supposed to improve on.")

    # ---- the composition question ---------------------------------------
    print("\n" + "-" * 78)
    print("COMPOSITION: what does r reduce to here?")
    print("-" * 78)
    const = [c for c in present
             if describe(c, [x[c] for x in rows])["distinct"] == 1]
    varying = [c for c in present if c not in const]
    print(f"  constant components : {const or 'none'}")
    print(f"  varying components  : {varying or 'none'}")
    if len(varying) == 1 and const:
        v = varying[0]
        w = 1.0 / len(present)
        offset = sum(rows[0][c] for c in const) * w
        pred = [w * x[v] + offset for x in rows]
        err = max(abs(a - x["r"]) for a, x in zip(pred, rows))
        print(f"\n  With {len(const)} of {len(present)} components pinned, r is an AFFINE")
        print(f"  function of {v} alone:  r = {w:.4f}*{v} + {offset:.4f}")
        print(f"  max |predicted - actual r| over {len(rows)} questions = {err:.2e}")
        lo, hi = w * 0 + offset, w * 1 + offset
        print(f"  so r is confined to [{lo:.4f}, {hi:.4f}] whatever {v} does,")
        print(f"  and can never reach the 0.5 boundary unless {v} > "
              f"{(0.5 - offset) / w:.4f}.")

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "r_component_diagnostics.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
