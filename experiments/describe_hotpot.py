"""Describe the HotpotQA distractor validation split, before any system runs.

Every structural number quoted in neutrorag/hotpot.py comes from here, so the
docstring's claims are reproducible rather than asserted. This script reads the
dataset and nothing else: no retriever, no reader, no NLI, no accuracy.

    python experiments/describe_hotpot.py

It also derives the Decision-2 saturation targets from the corpus depth
distribution ALONE. Those targets must be fixed before any accuracy is
measured; choosing them afterwards is post-hoc tuning.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neutrorag.hotpot import (  # noqa: E402
    MIN_DISTRACTORS, examples_from_rows, eligible_for_unanswerable,
)


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:5.2f}%"


def quantile(sorted_vals, q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def main() -> None:
    from datasets import load_dataset

    print("=" * 74)
    print("HotpotQA distractor / validation -- structural description")
    print("=" * 74)

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rows = list(ds)
    n = len(rows)
    print(f"instances: {n}")

    examples = examples_from_rows(rows)
    print(f"parsed by neutrorag.hotpot: {len(examples)}"
          f"  (dropped {n - len(examples)} with unusable gold)")

    # ---- question types -------------------------------------------------
    types = Counter(e.qtype for e in examples)
    print("\nquestion type")
    for t, c in types.most_common():
        print(f"  {t:12s} {c:5d}  {pct(c, len(examples))}")

    # ---- corpus depth ----------------------------------------------------
    depths = sorted(len(e.paragraphs) for e in examples)
    golds = Counter(len(e.gold_titles) for e in examples)
    gsents = Counter(len(e.gold_sentences) for e in examples)
    print("\nparagraphs per instance")
    for k, c in sorted(Counter(depths).items()):
        print(f"  {k:3d} paragraphs {c:5d}  {pct(c, len(examples))}")
    print("\ndistinct gold TITLES per instance")
    for k, c in sorted(golds.items()):
        print(f"  {k:3d} {c:5d}  {pct(c, len(examples))}")
    print("\ngold SENTENCES per instance")
    for k, c in sorted(gsents.items()):
        print(f"  {k:3d} {c:5d}  {pct(c, len(examples))}")

    # ---- eligibility for the unanswerable condition ----------------------
    ineligible = [e for e in examples if not eligible_for_unanswerable(e)]
    print(f"\nunanswerable condition (needs >= {MIN_DISTRACTORS} distractors left)")
    print(f"  eligible   {len(examples) - len(ineligible):5d}  "
          f"{pct(len(examples) - len(ineligible), len(examples))}")
    print(f"  INELIGIBLE {len(ineligible):5d}  {pct(len(ineligible), len(examples))}"
          f"   <- excluded from sampling")
    no_distractor = sum(1 for e in examples
                        if len(e.paragraphs) == len(e.gold_titles))
    print(f"    of which have no distractors at all: {no_distractor}")

    # ---- the span-reader ceiling ----------------------------------------
    print("\nextractive-reader ceiling (the answer must appear in the context)")
    hdr = f"  {'type':12s} {'n':>6s} {'yes/no':>7s} {'absent':>7s} {'ceiling':>8s}"
    print(hdr)
    tot = Counter()
    yn = Counter()
    ab = Counter()
    for e in examples:
        tot[e.qtype] += 1
        if e.is_yes_no:
            yn[e.qtype] += 1
        elif not e.answer_in_context():
            ab[e.qtype] += 1
    for t in sorted(tot):
        reach = tot[t] - yn[t] - ab[t]
        print(f"  {t:12s} {tot[t]:6d} {yn[t]:7d} {ab[t]:7d} {reach / tot[t]:8.3f}")
    T = sum(tot.values())
    R = T - sum(yn.values()) - sum(ab.values())
    print(f"  {'OVERALL':12s} {T:6d} {sum(yn.values()):7d} {sum(ab.values()):7d} "
          f"{R / T:8.3f}")
    print("\n  Every yes/no answer is a comparison question, so the ceiling on")
    print("  comparison is far below the overall figure. Report EM against this")
    print("  ceiling, never against 100%.")

    # ---- Decision-2 saturation targets, from depth only ------------------
    print("\n" + "=" * 74)
    print("DECISION-2 SATURATION TARGETS, derived from depth alone")
    print("=" * 74)
    p75 = quantile(depths, 0.75)
    srcs = sorted(len({p.title for p in e.paragraphs}) for e in examples)
    p75_src = quantile(srcs, 0.75)
    print(f"  paragraph-count p75      : {p75}")
    print(f"  distinct-source p75      : {p75_src}")
    print("\n  These are corpus-level. The scorer, however, only ever sees the")
    print("  retriever's top-k passages, so the depth it observes is k, not the")
    print("  corpus depth. SUPPORT_TARGET / DIVERSITY_TARGET must therefore be")
    print("  set from k -- and because every retrieved HotpotQA paragraph has a")
    print("  distinct title, diversity saturates identically on every instance")
    print("  and is INERT here, exactly as it was on FEVER.")
    print("\n  No accuracy has been measured at this point. Whatever targets are")
    print("  committed must be committed before the first accuracy run.")


if __name__ == "__main__":
    main()
