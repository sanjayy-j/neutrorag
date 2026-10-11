"""Why does inter-source contradiction come out HIGHER on "No conflict"?

Pairwise mode passed its sanity gate -- plainly contradicting text gives
contradiction 1.000, plainly agreeing text gives corroboration 0.997 -- so
unlike the claim-mode attempt this is NOT an ill-posed measurement. The
reversal therefore needs a different explanation, and this script looks for it
in the cached pairs. No model, no network.

    python experiments/diagnose_pairwise_reversal.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CONFLICTING = "Conflicting opinions and research outcomes"
NO_CONFLICT = "No conflict"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="data/conflicts_pairwise.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    recs = [json.loads(l) for l in open(REPO / args.cache, encoding="utf-8")]
    recs = [r for r in recs if not r.get("_header")]

    def contradiction(r):
        return max(p["c"] for p in r["pairs"]) if r["pairs"] else 0.0

    def corroboration(r):
        e = {(p["i"], p["j"]): p["e"] for p in r["pairs"]}
        ks = {p["i"] for p in r["pairs"]} | {p["j"] for p in r["pairs"]}
        vals = [min(e[(i, j)], e[(j, i)]) for i in ks for j in ks
                if i < j and (i, j) in e and (j, i) in e]
        return max(vals) if vals else 0.0

    by = defaultdict(list)
    for r in recs:
        by[r["conflict_type"]].append(r)

    print("1. Are the max-over-pairs statistics saturating?")
    print(f"   {'type':>44} {'n':>3} {'median max-c':>13} {'frac c>0.9':>11} "
          f"{'median max-corrob':>18} {'frac corrob>0.9':>16}")
    for t, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        cs = [contradiction(r) for r in rs]
        co = [corroboration(r) for r in rs]
        print(f"   {t:>44} {len(rs):>3} {statistics.median(cs):>13.3f} "
              f"{sum(1 for x in cs if x > 0.9)/len(cs):>10.1%} "
              f"{statistics.median(co):>18.3f} "
              f"{sum(1 for x in co if x > 0.9)/len(co):>15.1%}")

    print("\n2. Per-PAIR contradiction, pooled over all 2000 pairs")
    allc = [p["c"] for r in recs for p in r["pairs"]]
    alle = [p["e"] for r in recs for p in r["pairs"]]
    alln = [p["n"] for r in recs for p in r["pairs"]]
    for nm, v in (("entailment", alle), ("neutral", alln), ("contradiction", allc)):
        print(f"   {nm:>14}: median {statistics.median(v):.3f}  mean {statistics.mean(v):.3f}  "
              f"frac>0.5 {sum(1 for x in v if x > 0.5)/len(v):.1%}")
    print("   -> a single extreme pair sets the max; with 20 pairs per instance the")
    print("      max is an order statistic, not a measure of how conflicted the set is.")

    print("\n3. Does the MEAN over pairs separate the two types better than the max?")
    for stat_name, fn in (("max  (current)", lambda r: contradiction(r)),
                          ("mean over pairs", lambda r: statistics.mean(p["c"] for p in r["pairs"]))):
        a = [fn(r) for r in by[CONFLICTING]]
        b = [fn(r) for r in by[NO_CONFLICT]]
        print(f"   {stat_name:>16}: Conflicting {statistics.median(a):.3f}  "
              f"No-conflict {statistics.median(b):.3f}  "
              f"diff {statistics.median(a)-statistics.median(b):+.3f}")
    print("   (reported as a diagnostic only -- NOT adopted. Changing the statistic")
    print("    after seeing which way it separates would be exactly the post-hoc")
    print("    tuning the stop conditions forbid.)")

    print("\n4. The driving cases")
    nc = sorted(by[NO_CONFLICT], key=contradiction, reverse=True)[0]
    best = max(nc["pairs"], key=lambda p: p["c"])
    print(f"   highest-contradiction 'No conflict' instance (max-c {contradiction(nc):.3f}):")
    print(f"     question: {nc['question']!r}")
    print(f"     pair ({best['i']},{best['j']}) c={best['c']:.3f}")
    print(f"       A: {nc['passages'][best['i']]['text'][:150]!r}")
    print(f"       B: {nc['passages'][best['j']]['text'][:150]!r}")

    co_ = sorted(by[CONFLICTING], key=contradiction)[0]
    print(f"\n   lowest-contradiction 'Conflicting opinions' instance "
          f"(max-c {contradiction(co_):.3f}):")
    print(f"     question: {co_['question']!r}")
    print(f"     max entailment across its pairs: "
          f"{max(p['e'] for p in co_['pairs']):.3f}; "
          f"max neutral: {max(p['n'] for p in co_['pairs']):.3f}")
    print(f"       A: {co_['passages'][0]['text'][:150]!r}")
    print(f"       B: {co_['passages'][1]['text'][:150]!r}")

    print("\n5. Could the top-5 relevance cap be excluding the dissenting source?")
    print("   The cap keeps the 5 passages most similar to the QUESTION. A passage")
    print("   that restates the question ranks above one that disputes the premise,")
    print("   so the cap is not neutral with respect to conflict. Untestable from")
    print("   this cache, which stores only the 5 that were kept -- it would need a")
    print("   re-run at a larger cap. Stated as an unresolved confound.")


if __name__ == "__main__":
    main()
