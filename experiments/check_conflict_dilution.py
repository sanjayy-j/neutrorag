"""Memo F6/F8: does adding neutral filler still dilute the conflict signal?

The memo's test: take the drug-trial case (one passage reporting a 30% risk
reduction, one reporting no significant reduction) and add neutral passages one
at a time. Under the pooled formula F decayed 1.000 -> 0.752 -> 0.631 -> 0.545
-> 0.499 even though the opposed pair never changed.

Those published numbers came from the pre-Gate-B code (old r, old I, and a
non-normalised mock backend), so they are not directly comparable to anything
produced now. This script therefore runs BOTH disagree formulas under otherwise
identical current code, which isolates the reformulation itself.

Note that F = r*c + lambda*disagree still contains a pooled term: `c` is the
relevance-weighted mean contradiction, so it falls as filler is added regardless
of what `disagree` does. The prediction "F should now be flat" can only hold for
the disagree component; the report separates them.

    python experiments/check_conflict_dilution.py            # mock NLI (offline)
    python experiments/check_conflict_dilution.py --nli real # real DeBERTa NLI
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neutrorag.core import clamp  # noqa: E402
from neutrorag.policy import NeutroPolicy, PolicyConfig  # noqa: E402
from neutrorag.scoring import (  # noqa: E402
    Evidence, HFNLIBackend, MockNLIBackend, NLIRetrievalScorer,
)

CLAIM = "Drug X reduces the risk of Condition Y in high-risk patients."

OPPOSED = [
    Evidence("In a randomized controlled trial, Drug X reduced the risk of Condition Y "
             "by 30 percent in high-risk patients.", 0.95, source="trialA"),
    Evidence("A large meta-analysis found no significant reduction in Condition Y risk "
             "among high-risk patients taking Drug X.", 0.90, source="metaM"),
]

FILLER = [
    Evidence("Drug X was approved for cardiovascular risk reduction over a decade ago.",
             0.60, source="regulatory"),
    Evidence("Condition Y prevalence increases significantly with age.", 0.55, source="epi"),
    Evidence("Drug X is commonly prescribed to high-risk patients for a range of conditions.",
             0.50, source="label"),
    Evidence("Clinical guidelines for Condition Y were last updated three years ago.",
             0.45, source="guideline"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nli", choices=["mock", "real"], default="mock")
    return p.parse_args()


def pooled_and_pairwise(scorer, claim, evidence):
    """Recompute both disagree variants from the same per-passage NLI outputs."""
    kept = [e for e in evidence if e.text.strip()]
    preds = [scorer.nli.predict(ev.text, claim) for ev in kept]
    weights = [clamp(ev.relevance) for ev in kept]
    wsum = sum(weights) or 1.0
    e = sum(w * p[0] for w, p in zip(weights, preds)) / wsum
    c = sum(w * p[2] for w, p in zip(weights, preds)) / wsum
    pooled = clamp(2.0 * min(e, c))
    raw = 2.0 * max(min(p[0], q[2]) for p in preds for q in preds)
    return e, c, pooled, clamp(raw), raw


def main() -> None:
    args = parse_args()
    nli = MockNLIBackend() if args.nli == "mock" else HFNLIBackend()
    scorer = NLIRetrievalScorer(nli=nli)
    policy = NeutroPolicy()
    tau_f = PolicyConfig().tau_F

    print(f"Conflict dilution test -- {args.nli} NLI")
    print(f"claim: {CLAIM}")
    print(f"opposed pair held fixed; neutral filler added one passage at a time\n")
    print(f"{'filler':>6} {'r':>6} {'c(pooled)':>10} {'disagree_OLD':>13} "
          f"{'disagree_NEW':>13} {'(unclamped)':>12} {'F':>7} {'F_if_old':>9}")
    print("-" * 92)

    news, olds, fs = [], [], []
    for k in range(len(FILLER) + 1):
        evidence = OPPOSED + FILLER[:k]
        # the scorer uses a fresh backend call sequence each time; rebuild to keep
        # any stateful backend honest
        scorer = NLIRetrievalScorer(nli=nli)
        tif = scorer.score(CLAIM, evidence)
        r = scorer.sufficiency(CLAIM, evidence).r
        e, c, pooled, pairwise, raw = pooled_and_pairwise(scorer, CLAIM, evidence)
        f_if_old = clamp(r * c + scorer.lam * pooled)
        action, _ = policy.decide_once(tif)
        news.append(pairwise); olds.append(pooled); fs.append(tif.F)
        print(f"{k:>6} {r:>6.3f} {c:>10.3f} {pooled:>13.3f} {pairwise:>13.3f} "
              f"{raw:>12.3f} {tif.F:>7.3f} {f_if_old:>9.3f}")

    def spread(v):
        return max(v) - min(v)

    print("\nspread (max - min) across the filler sweep:")
    print(f"  disagree OLD (pooled)   : {spread(olds):.3f}   {olds[0]:.3f} -> {olds[-1]:.3f}")
    print(f"  disagree NEW (pairwise) : {spread(news):.3f}   {news[0]:.3f} -> {news[-1]:.3f}")
    print(f"  F (new)                 : {spread(fs):.3f}   {fs[0]:.3f} -> {fs[-1]:.3f}")
    print("\n  F still moves because F = r*c + lambda*disagree and BOTH r and the")
    print("  pooled c fall as filler is added. Only the disagree term is expected flat.")

    print(f"\nF8 -- does the flagship case fire SURFACE_CONTRADICTION (F >= tau_F = {tau_f})?")
    for k in (0, len(FILLER)):
        evidence = OPPOSED + FILLER[:k]
        scorer = NLIRetrievalScorer(nli=nli)
        tif = scorer.score(CLAIM, evidence)
        action, rationale = policy.decide_once(tif)
        verdict = "YES" if tif.F >= tau_f else "NO"
        print(f"  {k} filler: F={tif.F:.3f} -> {verdict};  action={action.value}  ({rationale})")


if __name__ == "__main__":
    main()
