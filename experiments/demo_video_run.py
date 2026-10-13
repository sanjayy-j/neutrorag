"""A scripted demonstration of the five-way policy, sized for a 3-5 minute video.

Shows the thing the project actually claims: that different evidence states
produce different ACTIONS, rather than one confidence score and one refusal.

    python experiments/demo_video_run.py              # real NLI (default)
    python experiments/demo_video_run.py --nli mock   # instant, offline fallback

Four cases:
  1  clear support            -> ANSWER
  2  two opposed sources      -> SURFACE_CONTRADICTION   (the memo F8 fix)
  3  thin evidence, retrieval -> EXPAND, then ABSTAIN when it finds nothing new
     exhausted

Case 3 prints the retrieval sufficiency r it computed, which is where the memo F1
fix (r was a hardcoded constant for all 200 FEVER claims) is visible. An earlier
fourth case swept r across three retrieval states; it was cut because it used the
CONFLICTS saturation targets while cases 1-3 use the library defaults, so the same
one-passage evidence set printed two different r values -- correct, but not
explainable in the time a demo allows. The r distribution is reported properly in
RESULTS.md.

WHAT THIS IS NOT. No accuracy is reported here and none should be quoted from
it. These are hand-built cases chosen to exercise each branch of the policy.
The project's measured evaluation is in RESULTS.md, and its headline finding is
negative -- see DEMO_SCRIPT.md for the honest framing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neutrorag.policy import NeutroPolicy, PolicyConfig  # noqa: E402
from neutrorag.scoring import (  # noqa: E402
    Evidence, HFNLIBackend, MockNLIBackend, NLIRetrievalScorer,
)

CFG = PolicyConfig()


def rule(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show(label: str, tif, action, rationale, extra: str = "") -> None:
    print(f"\n  evidence state : {label}")
    print(f"  T={tif.T:.3f}  I={tif.I:.3f}  F={tif.F:.3f}")
    if extra:
        print(f"  {extra}")
    print(f"  --> ACTION: {action.value.upper()}")
    print(f"      because: {rationale}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nli", choices=["real", "mock"], default="real")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    print("=" * 70)
    print("NeutroRAG -- the five-way decision policy")
    print("=" * 70)
    print(f"NLI backend: {'DeBERTa-v3-base-mnli-fever-anli (real)' if args.nli == 'real' else 'MockNLIBackend (lexical heuristic)'}")
    print(f"Thresholds : tau_T={CFG.tau_T}  tau_I={CFG.tau_I}  tau_F={CFG.tau_F}  "
          f"tau_T_conflict={CFG.tau_T_conflict}")
    print("Hand-built cases. No accuracy is reported here -- see RESULTS.md.")

    nli = HFNLIBackend() if args.nli == "real" else MockNLIBackend()
    scorer = NLIRetrievalScorer(nli=nli)
    policy = NeutroPolicy()

    # ---- 1. ANSWER --------------------------------------------------------
    rule("CASE 1 -- corroborated evidence  ->  ANSWER")
    claim1 = "The heart pumps blood through the body."
    ev1 = [
        Evidence("The heart is a muscular organ that pumps blood through the "
                 "circulatory system.", 0.95, source="physiology-text"),
        Evidence("Cardiac contraction drives blood through the body's arteries "
                 "and veins.", 0.90, source="cardiology-review"),
    ]
    tif1 = scorer.score(claim1, ev1)
    a1, r1 = policy.decide_once(tif1)
    print(f'\n  claim: "{claim1}"')
    show("two sources, both entailing", tif1, a1, r1)

    # ---- 2. SURFACE_CONTRADICTION (memo F8) -------------------------------
    rule("CASE 2 -- sources that disagree  ->  SURFACE CONTRADICTION")
    print("\n  This is the case the first review found BROKEN (memo F8): conflict")
    print("  was pooled into a mean, so it washed out and the system abstained.")
    claim2 = "Drug X reduces the risk of Condition Y in high-risk patients."
    ev2 = [
        Evidence("In a randomized controlled trial, Drug X reduced the risk of "
                 "Condition Y by 30 percent in high-risk patients.", 0.95,
                 source="trial-A"),
        Evidence("A large meta-analysis found no significant reduction in "
                 "Condition Y risk among high-risk patients taking Drug X.", 0.90,
                 source="meta-analysis-M"),
    ]
    tif2 = scorer.score(claim2, ev2)
    a2, r2 = policy.decide_once(tif2)
    print(f'\n  claim: "{claim2}"')
    show("one source supports, one refutes", tif2, a2, r2,
         extra=f"F={tif2.F:.3f} vs tau_F={CFG.tau_F} -> the conflict now fires")
    print("\n  The system does NOT average these into 'probably true'. It reports")
    print("  that the sources disagree, which is the behaviour the project argues")
    print("  a single confidence score cannot express.")

    # ---- 3. EXPAND then ABSTAIN ------------------------------------------
    rule("CASE 3 -- thin evidence  ->  EXPAND, then ABSTAIN")
    claim3 = "The bridge was repainted blue in 2005."
    thin = [Evidence("The bridge spans a wide river in the northern region.",
                     0.30, source="gazetteer")]

    def exhausted(_claim, evidence, _step):
        return evidence  # a retriever that finds nothing new

    decision = policy.run(claim3, thin, scorer.score, exhausted)
    print(f'\n  claim: "{claim3}"')
    print(f"  retrieval budget: {CFG.max_expansions} expansions\n")
    r_thin = scorer.sufficiency(claim3, thin).r
    print(f"    retrieval sufficiency r = {r_thin:.3f}; the one passage is on a")
    print("    different subject, so NLI returns near-pure neutral.")
    for i, step in enumerate(decision.history):
        print(f"    step {i}: T={step.tif.T:.3f} I={step.tif.I:.3f} "
              f"F={step.tif.F:.3f}  -> {step.action.value.upper()}")
    print(f"\n  --> FINAL ACTION: {decision.action.value.upper()} "
          f"after {decision.expansions_used} expansions")
    print("      High indeterminacy asks for more evidence FIRST. Only when")
    print("      retrieval is exhausted does it abstain. That staging is the")
    print("      difference between 'I don't know' and 'I have looked and")
    print("      still don't know'.")

    rule("END")
    print(f"\n  wall-clock: {time.time() - t0:.1f}s")
    print("  Measured evaluation, including the negative headline result and")
    print("  what is retracted: RESULTS.md")


if __name__ == "__main__":
    main()
