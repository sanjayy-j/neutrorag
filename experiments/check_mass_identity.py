"""Gate B prediction test: does T + I + F = 1 + lambda*disagree still hold?

The Gate B proposal committed, in advance, to this prediction:

    "The mass identity breaks. T + I + F = 1 + lambda*disagree currently holds
     because I = (1-r) + r*n and the r terms telescope against T and F. Once r
     carries retrieval information orthogonal to the NLI simplex, the sum is no
     longer a function of disagree alone."

This script tests that empirically the same way the memo's Appendix A did:
over random draws, and over a real run. It reports the maximum absolute
deviation and states plainly whether the identity broke or held.

    python experiments/check_mass_identity.py
    python experiments/check_mass_identity.py --draws 200000

The real-run check replays the NLI outputs recorded in
fever_run_200_current.csv
through the current scorer, so the model is not re-run and the e/n/c values are
bit-identical to the Gate A run; only r differs.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neutrorag.core import clamp  # noqa: E402
from neutrorag.graph import Fact, KnowledgeGraph  # noqa: E402
from neutrorag.scoring import Evidence, MockNLIBackend, NLIRetrievalScorer  # noqa: E402
from neutrorag.sufficiency import retrieval_sufficiency  # noqa: E402

LAM = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--draws", type=int, default=200_000)
    p.add_argument("--csv", default="fever_run_200_current.csv")
    p.add_argument("--seed", type=int, default=20260913)
    return p.parse_args()


class _ReplayNLI:
    """Replays recorded (e, n, c) values in order -- no model, identical numbers."""

    def __init__(self, rows):
        self._rows = list(rows)
        self._i = 0

    def predict(self, premise, hypothesis):
        out = self._rows[self._i]
        self._i += 1
        return out


def deviation(T: float, I: float, F: float, e: float, c: float, lam: float = LAM) -> float:
    """|(T + I + F) - (1 + lambda*disagree)|."""
    disagree = clamp(2.0 * min(e, c))
    return abs((T + I + F) - (1.0 + lam * disagree))


def channels(r: float, e: float, n: float, c: float, lam: float = LAM):
    """Apply the REAL scorer to a given r and e/n/c.

    This previously held a hand-copied duplicate of the three channel formulas.
    That was a latent trap and it sprang: when I was changed to max(1-r, n) the
    duplicate kept the old form, so the random-draw checks reported the identity
    still holding (4.4e-16) while the real runs -- which go through
    NLIRetrievalScorer.score -- correctly reported it broken. A check that can
    disagree with the code it is checking is worse than no check.

    Now routed through the real scorer: one passage, r pinned via r_override, and
    the NLI output replayed. With a single passage the relevance-weighted means
    reduce to exactly (e, n, c), so this is the production formula, not a copy.
    """
    scorer = NLIRetrievalScorer(nli=_ReplayNLI([(e, n, c)]), lam=lam, r_override=r)
    tif = scorer.score("claim", [Evidence("passage", 1.0)])
    return tif.T, tif.I, tif.F


def random_simplex(rng: random.Random):
    a, b, c_ = rng.random(), rng.random(), rng.random()
    total = a + b + c_ or 1.0
    return a / total, b / total, c_ / total


def _report_masses(masses, label: str) -> None:
    """Distribution of T+I+F. Under the old I this was 1 + lam*disagree exactly,
    so 'mass > 1' merely restated 'disagree > 0' (memo F5). If the mass now
    carries independent information, this distribution should be non-degenerate
    and should straddle 1 in both directions."""
    if not masses:
        return
    srt = sorted(masses)
    n = len(srt)
    above = sum(1 for m in masses if m > 1.0 + 1e-12)
    below = sum(1 for m in masses if m < 1.0 - 1e-12)
    print(f"    mass T+I+F over {label}: min={srt[0]:.3f} median={srt[n // 2]:.3f} "
          f"max={srt[-1]:.3f}")
    print(f"      exceeds 1: {above}/{n} ({above / n:.1%})   "
          f"below 1: {below}/{n} ({below / n:.1%})   "
          f"exactly 1: {n - above - below}/{n}")


def check_random_uniform_r(draws: int, rng: random.Random) -> float:
    """r drawn uniformly -- the memo's Appendix A check, repeated."""
    worst = 0.0
    for _ in range(draws):
        e, n, c = random_simplex(rng)
        r = rng.random()
        T, I, F = channels(r, e, n, c)
        worst = max(worst, deviation(T, I, F, e, c))
    return worst


def check_random_computed_r(draws: int, rng: random.Random) -> float:
    """r computed by the NEW sufficiency definition over synthetic retrieval.

    This is the condition the prediction is actually about: r carrying genuine,
    varying retrieval information rather than a constant or a uniform draw.
    """
    vocab = ["drug", "condition", "trial", "risk", "patients", "reduction",
             "market", "climate", "bridge", "novel"]
    worst = 0.0
    for _ in range(draws):
        e, n, c = random_simplex(rng)
        claim = " ".join(rng.sample(vocab, rng.randint(2, 5)))
        passages = [" ".join(rng.sample(vocab, rng.randint(2, 6)))
                    for _ in range(rng.randint(1, 5))]
        relevances = [rng.random() for _ in passages]
        sources = [f"doc{rng.randint(0, 4)}" for _ in passages]
        r = retrieval_sufficiency(claim, passages, relevances, sources=sources).r
        T, I, F = channels(r, e, n, c)
        worst = max(worst, deviation(T, I, F, e, c))
    return worst


def check_real_run_fever(csv_path: Path) -> float:
    """Real run: the 200 FEVER claims, replaying the recorded NLI outputs."""
    from neutrorag.fever import load_fever_hf

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    examples = load_fever_hf(split="validation", max_examples=len(rows))

    worst = 0.0
    masses = []
    for ex, row in zip(examples, rows):
        if row["e"] == "":
            continue  # empty-evidence short circuit; no NLI call was made
        e, n, c = float(row["e"]), float(row["n"]), float(row["c"])
        scorer = NLIRetrievalScorer(nli=_ReplayNLI([(e, n, c)] * len(ex.evidence)))
        tif = scorer.score(ex.claim, ex.evidence)
        worst = max(worst, deviation(tif.T, tif.I, tif.F, e, c))
        masses.append(tif.mass)
    _report_masses(masses, "FEVER 200")
    return worst


def check_real_run_pipeline() -> float:
    """Real run where all four sufficiency components are live.

    FEVER cannot vary r (memo F9), so this uses the pipeline demo's graph and
    corpus, where coverage, support_count, connectivity and diversity all move.
    """
    graph = KnowledgeGraph()
    graph.add_facts([
        Fact("Drug X", "studied_in", "Trial A",
             "In a randomized controlled trial, Drug X reduced the risk of Condition Y "
             "by 30 percent in high-risk patients."),
        Fact("Drug X", "studied_in", "Meta-analysis M",
             "A large meta-analysis found no significant reduction in Condition Y risk "
             "among high-risk patients taking Drug X."),
        Fact("Drug X", "prescribed_to", "high-risk patients",
             "Drug X is commonly prescribed to high-risk patients for a range of conditions."),
    ])
    scorer = NLIRetrievalScorer(nli=MockNLIBackend(), graph=graph)

    claims = [
        "Drug X reduces the risk of Condition Y in high-risk patients.",
        "Condition Y is more common with age.",
        "Drug X is prescribed to high-risk patients.",
        "An unrelated claim about the stock market.",
    ]
    corpora = [
        [Evidence("In a randomized controlled trial, Drug X reduced the risk of Condition Y "
                  "by 30 percent in high-risk patients.", 0.95, source="trialA"),
         Evidence("A large meta-analysis found no significant reduction in Condition Y risk "
                  "among high-risk patients taking Drug X.", 0.90, source="metaM"),
         Evidence("Drug X is commonly prescribed to high-risk patients.", 0.55, source="label")],
        [Evidence("Condition Y prevalence increases significantly with age.", 0.8, source="epi")],
        [Evidence("Drug X is commonly prescribed to high-risk patients.", 0.7, source="label"),
         Evidence("Drug X was approved over a decade ago.", 0.4, source="regulatory")],
        [Evidence("The stock market fell sharply on Tuesday.", 0.3, source="news")],
    ]

    worst = 0.0
    masses = []
    print("\n    per-claim r (all four components live):")
    for claim, evidence in zip(claims, corpora):
        breakdown = scorer.sufficiency(claim, evidence)
        tif = scorer.score(claim, evidence)
        # recover pooled e/n/c the same way the scorer does, for the identity check
        weights = [clamp(ev.relevance) for ev in evidence]
        preds = [scorer.nli.predict(ev.text, claim) for ev in evidence]
        wsum = sum(weights) or 1.0
        e = sum(w * p[0] for w, p in zip(weights, preds)) / wsum
        c = sum(w * p[2] for w, p in zip(weights, preds)) / wsum
        dev = deviation(tif.T, tif.I, tif.F, e, c)
        worst = max(worst, dev)
        masses.append(tif.mass)
        comps = " ".join(f"{k}={v:.2f}" for k, v in breakdown.components.items())
        print(f"      r={breakdown.r:.3f}  [{comps}]  {tif}  mass={tif.mass:.3f}  dev={dev:.2e}")
    _report_masses(masses, "pipeline corpus")
    return worst


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print("Gate B prediction test: does T + I + F = 1 + lambda*disagree still hold?")
    print(f"  lambda = {LAM}, seed = {args.seed}, draws = {args.draws}")
    print("  PREDICTION (Gate B proposal): the identity BREAKS once r is informative.")

    results = {}

    print("\n[1] random draws, r ~ Uniform(0,1)")
    results["random uniform r"] = check_random_uniform_r(args.draws, rng)
    print(f"    max deviation = {results['random uniform r']:.3e}")

    print("\n[2] random draws, r computed by neutrorag.sufficiency")
    results["random computed r"] = check_random_computed_r(args.draws // 20, rng)
    print(f"    max deviation = {results['random computed r']:.3e}")

    print("\n[3] real run: pipeline corpus + knowledge graph (all four components live)")
    results["real run (pipeline)"] = check_real_run_pipeline()
    print(f"    max deviation = {results['real run (pipeline)']:.3e}")

    csv_path = REPO / args.csv
    if csv_path.exists():
        print("\n[4] real run: 200 FEVER claims, replaying the recorded NLI outputs")
        results["real run (FEVER 200)"] = check_real_run_fever(csv_path)
        print(f"    max deviation = {results['real run (FEVER 200)']:.3e}")
    else:
        print(f"\n[4] skipped: {args.csv} not found")

    worst = max(results.values())
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    for name, dev in results.items():
        print(f"  {name:28s} max |deviation| = {dev:.3e}")
    print(f"\n  worst overall: {worst:.3e}")
    if worst > 1e-9:
        print("\n  The identity BROKE. The Gate B prediction is CONFIRMED.")
    else:
        print("\n  The identity STILL HOLDS, to machine precision.")
        print("  The Gate B prediction is FALSIFIED: redefining r does not break it.")
        print("  Per the session instruction, STOP -- do not proceed to the r-ablation,")
        print("  the sensitivity sweep, or the repeated Gate A.")
    print()


if __name__ == "__main__":
    main()
