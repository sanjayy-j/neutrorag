"""Derive the saturation targets for support_count and diversity.

DECISION 2 ORDER OF OPERATIONS. This script reads ONLY retrieval-depth
statistics -- passages per instance, distinct domains per instance -- and the
retrieval-side `r` they imply. It never computes, loads, or prints an accuracy,
an attribution, or a gold label. Choosing a target after seeing accuracy would
be post-hoc tuning; the commit order is the evidence that it was not.

    python experiments/derive_targets.py

Requires data/conflicts.jsonl (gitignored, see neutrorag/conflicts.py) and the
cached FEVER split.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neutrorag.conflicts import load_conflicts  # noqa: E402
from neutrorag.fever import load_fever_hf  # noqa: E402
from neutrorag.sufficiency import coverage, retrieval_sufficiency  # noqa: E402

PERCENTILE = 75  # see "The rule", below


def pctile(values, p):
    v = sorted(values)
    i = min(len(v) - 1, max(0, int(round(p / 100 * (len(v) - 1)))))
    return v[i]


def depth_profile(name, counts, domains=None):
    print(f"\n{name} (n={len(counts)})")
    print(f"  passages/instance : min {min(counts)}  p25 {pctile(counts,25)}  "
          f"median {pctile(counts,50)}  p75 {pctile(counts,75)}  "
          f"p90 {pctile(counts,90)}  max {max(counts)}")
    print(f"  histogram         : {dict(sorted(Counter(counts).items()))}")
    if domains is None:
        print("  domains/instance  : UNDEFINED (no provenance in this corpus)")
    else:
        print(f"  domains/instance  : min {min(domains)}  p25 {pctile(domains,25)}  "
              f"median {pctile(domains,50)}  p75 {pctile(domains,75)}  "
              f"p90 {pctile(domains,90)}  max {max(domains)}")


def sweep(name, counts, label):
    """Spread of min(1, k/target) across targets -- depth only."""
    print(f"\n  {name}: can a fixed target make min(1, {label}/target) discriminate?")
    print(f"    {'target':>7} {'distinct':>9} {'std':>7} {'%saturated':>11} {'%at floor':>10}")
    for t in (1, 2, 3, 5, 8, 9, 10, 12, 15, 20):
        v = [min(1.0, k / t) for k in counts]
        sat = sum(1 for x in v if x >= 1.0) / len(v)
        floor = sum(1 for x in v if x <= min(v) + 1e-9) / len(v)
        print(f"    {t:>7} {len(set(round(x,6) for x in v)):>9} "
              f"{statistics.pstdev(v):>7.3f} {sat:>10.1%} {floor:>9.1%}")


def main() -> None:
    print(__doc__.split("\n\n")[1])  # the order-of-operations warning

    # ---------------- depth only -----------------------------------------
    fever = load_fever_hf(split="validation", max_examples=200)
    kf = [len([e for e in x.evidence if e.text.strip()]) for x in fever]
    sources_f = sum(1 for x in fever for e in x.evidence if e.source)

    conflicts = load_conflicts(str(REPO / "data" / "conflicts.jsonl"),
                               relevance_fn=lambda q, ps: [1.0] * len(ps))
    kc = [len(x.evidence) for x in conflicts]
    dc = [len({e.source for e in x.evidence if e.source}) for x in conflicts]

    print("=" * 74)
    print("STEP 1 -- retrieval-depth distributions")
    print("=" * 74)
    depth_profile("FEVER  (copenlu/fever_gold_evidence, validation[:200])", kf, None)
    print(f"  evidence items carrying a source: {sources_f}")
    depth_profile("CONFLICTS (rag_conflicts, all 458)", kc, dc)

    print("\n" + "=" * 74)
    print("STEP 2 -- can one fixed target serve both corpora?")
    print("=" * 74)
    sweep("FEVER", kf, "k")
    sweep("CONFLICTS", kc, "k")
    sweep("CONFLICTS", dc, "domains")

    print("""
  FINDING. No fixed target can make support_count discriminate on BOTH corpora,
  and on FEVER no target can make it discriminate AT ALL:

    * FEVER's passage count is a point mass -- p25 = median = p75 = p90 = 1,
      with 92.5% of instances at exactly k=1. A saturating function of k cannot
      manufacture variance that the input does not contain. The best spread over
      every target tried is std 0.132, and it comes entirely from the 7.5% tail.
    * CONFLICTS is concentrated at 8-10 (p25=8, p75=10). At the current target
      of 3 it is 100% saturated -- std exactly 0.000.
    * The two requirements are mutually exclusive: any target small enough to
      lift FEVER (<=5) leaves CONFLICTS completely saturated, and any target
      large enough to spread CONFLICTS (>=10) pins FEVER at k/target ~ 0.1.

  So the honest answer to "what is the right fixed target" is: there isn't one.
  What matters is not the lost discrimination -- the input has none to give on
  FEVER -- but the BIAS. At target=3, FEVER's support_count is the constant
  0.333, which does not merely fail to discriminate, it drags r down by a fixed
  amount on every instance, which is what put 143/200 FEVER instances below the
  F12 cliff at r <= 0.5. A constant of 1.0 is equally non-discriminating and
  unbiased.

  THE RULE ADOPTED. target = p75 of that corpus's own depth distribution.

    - p75, not max: max is outlier-driven (CONFLICTS max=20 occurs twice in 458;
      FEVER max=15 once in 200).
    - p75, not median: the median saturates half the corpus by construction.
    - p75 leaves the lower three quartiles on the discriminating ramp and
      saturates only the top quarter, which is the intended reading of the
      component -- "as many passages as a well-retrieved instance in this corpus".
    - It is chosen by a stated rule, not by maximising spread. (For reference the
      spread-maximising target on CONFLICTS is 12 vs the p75 value of 10; we take
      10. Maximising a statistic, even a depth-only one, is still fitting.)
    - Where the rule yields a degenerate component (FEVER: p75 = 1, so every
      instance saturates), the component is non-discriminating BY THE CORPUS, and
      is reported as such rather than silently contributing a constant penalty.
""")

    st_f, st_c = pctile(kf, PERCENTILE), pctile(kc, PERCENTILE)
    dv_c = pctile(dc, PERCENTILE)
    print("=" * 74)
    print("STEP 3 -- targets adopted")
    print("=" * 74)
    print(f"  FEVER     support_target = p{PERCENTILE}(passages) = {st_f}"
          f"   diversity: N/A (no provenance)")
    print(f"  CONFLICTS support_target = p{PERCENTILE}(passages) = {st_c}"
          f"   diversity_target = p{PERCENTILE}(domains) = {dv_c}")

    # ---------------- predicted effect on r (still not accuracy) ----------
    print("\n" + "=" * 74)
    print("STEP 4 -- PREDICTED effect on r, stated before any accuracy run")
    print("=" * 74)

    def r_dist(examples, claim_of, support_target, diversity_target):
        out = []
        for x in examples:
            ev = [e for e in x.evidence if e.text.strip()]
            if not ev:
                continue
            srcs = [e.source for e in ev]
            out.append(retrieval_sufficiency(
                claim_of(x), [e.text for e in ev], [e.relevance for e in ev],
                sources=srcs if any(srcs) else None, graph=None,
                support_target=support_target, diversity_target=diversity_target).r)
        return out

    for name, exs, claim_of, old, new in (
        ("FEVER", fever, lambda x: x.claim, (3, 3), (st_f, 3)),
        ("CONFLICTS", conflicts, lambda x: x.question, (3, 3), (st_c, dv_c)),
    ):
        o = r_dist(exs, claim_of, *old)
        n = r_dist(exs, claim_of, *new)
        print(f"\n  {name}")
        print(f"    r BEFORE (targets {old}): min {min(o):.3f} median "
              f"{statistics.median(o):.3f} max {max(o):.3f}  "
              f"| <=0.5: {sum(1 for v in o if v <= 0.5)}/{len(o)}")
        print(f"    r AFTER  (targets {new}): min {min(n):.3f} median "
              f"{statistics.median(n):.3f} max {max(n):.3f}  "
              f"| <=0.5: {sum(1 for v in n if v <= 0.5)}/{len(n)}")
        print(f"    spread (pstdev): {statistics.pstdev(o):.4f} -> {statistics.pstdev(n):.4f}")

    print("""
  PREDICTIONS ON THE RECORD (to be checked, not adjusted, after the runs):
    1. FEVER's r rises and stops being dragged below the F12 cliff; the count of
       instances at r <= 0.5 should fall to near zero.
    2. CONFLICTS' r falls slightly and GAINS spread, because support_count and
       diversity stop being pinned at 1.0.
    3. Neither corpus gains much support_count DISCRIMINATION, because neither
       depth distribution carries much. The fix removes a bias; it does not
       create a signal. If a later result depends on support_count varying, that
       dependence is unsupported on these two corpora.
""")


if __name__ == "__main__":
    main()
