"""Does pairwise mode behave correctly on cases where we already know the answer?

This is a GATE, and it costs about a minute. It runs before any experiment
because the previous venue attempt spent 107 minutes of compute before the
framing error surfaced -- a check this cheap would have caught it.

Three hand-built cases, two passages each:
  contradicting -> F high
  agreeing      -> T (corroboration) high, F low
  unrelated     -> I high

    python experiments/check_pairwise_sanity.py            # mock NLI
    python experiments/check_pairwise_sanity.py --nli real # real DeBERTa NLI
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neutrorag.scoring import (  # noqa: E402
    Evidence, HFNLIBackend, MockNLIBackend, NLIRetrievalScorer,
)

# (name, query, passage_a, passage_b, expected dominant channel)
CASES = [
    ("contradicting", "Did Drug X reduce mortality in the trial?",
     "The trial found that Drug X reduced mortality by 30 percent in treated patients.",
     "The trial found that Drug X did not reduce mortality in treated patients.",
     "F"),
    ("agreeing", "What is the capital of France?",
     "Paris is the capital of France.",
     "The capital city of France is Paris.",
     "T"),
    ("unrelated", "What happened in the market on Tuesday?",
     "The stock market fell sharply on Tuesday amid inflation concerns.",
     "Penguins are flightless birds that live in the southern hemisphere.",
     "I"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nli", choices=["mock", "real"], default="mock")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    nli = MockNLIBackend() if args.nli == "mock" else HFNLIBackend()
    scorer = NLIRetrievalScorer(nli=nli)

    print(f"Pairwise-mode sanity gate -- {args.nli} NLI")
    print("T is CORROBORATION in this mode, not support for a claim.\n")
    print(f"  {'case':>14} {'r':>6} {'corrob':>7} {'contra':>7} {'neutral':>8} "
          f"{'T':>6} {'I':>6} {'F':>6} {'dominant':>9} {'expect':>7} {'':>5}")
    print("  " + "-" * 88)

    failures = []
    for name, query, a, b, expect in CASES:
        ev = [Evidence(a, 0.9, source="src_a"), Evidence(b, 0.9, source="src_b")]
        bd = scorer.pairwise_breakdown(query, ev)
        got = bd.tif.dominant
        ok = got == expect
        if not ok:
            failures.append((name, expect, got))
        print(f"  {name:>14} {bd.r:>6.3f} {bd.corroboration:>7.3f} "
              f"{bd.contradiction:>7.3f} {bd.neutrality:>8.3f} "
              f"{bd.tif.T:>6.3f} {bd.tif.I:>6.3f} {bd.tif.F:>6.3f} "
              f"{got:>9} {expect:>7} {'OK' if ok else 'FAIL':>5}")

    print()
    if failures:
        print("  GATE FAILED:")
        for name, expect, got in failures:
            print(f"    {name}: expected dominant {expect}, got {got}")
        print("\n  The frame is wrong. Stop -- nothing downstream matters.")
        raise SystemExit(1)
    print("  GATE PASSED -- all three hand-built cases land on the expected channel.")


if __name__ == "__main__":
    main()
