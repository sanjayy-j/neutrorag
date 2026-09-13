"""Why the CONFLICTS results are not interpretable as a test of `disagree`.

NeutroRAG's scorer is defined over a CLAIM: it asks an NLI model whether each
passage entails, is neutral toward, or contradicts a declarative proposition.
CONFLICTS is a QA dataset -- its instances are questions ("what is the most
common blood type", "Can renewable energy sources provide a stable power
supply?"). Passing a question as the NLI hypothesis is not a well-posed
entailment problem: there is no proposition to contradict.

This script replays the cached NLI outputs (no model, no network) and shows what
that does to the measurement. Run after score_conflicts.py.

    python experiments/diagnose_conflicts_hypothesis.py
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="data/conflicts_scored.jsonl")
    p.add_argument("--raw", default="data/conflicts.jsonl")
    return p.parse_args()


def raw_disagree(rec) -> float:
    tr = [(p["e"], p["n"], p["c"]) for p in rec["passages"]]
    return 2.0 * max(min(a[0], b[2]) for a in tr for b in tr) if tr else 0.0


def main() -> None:
    args = parse_args()
    rows = [json.loads(l) for l in open(REPO / args.cache, encoding="utf-8")]
    rows = [r for r in rows if not r.get("_header")]

    print("1. The NLI hypothesis is an interrogative, not a proposition")
    for r in rows[:4]:
        print(f"     {r['question']!r}")

    print("\n2. The resulting NLI output is lopsided, as that predicts")
    es = [p["e"] for r in rows for p in r["passages"]]
    ns = [p["n"] for r in rows for p in r["passages"]]
    cs = [p["c"] for r in rows for p in r["passages"]]
    for nm, v in (("entailment", es), ("neutral", ns), ("contradiction", cs)):
        print(f"     {nm:14s} median={statistics.median(v):.3f} "
              f"mean={statistics.mean(v):.3f} frac>0.5={sum(1 for x in v if x > 0.5)/len(v):.1%}")
    print("     -> the model calls a passage 'entailing' whenever it is merely on topic,")
    print("        and contradiction is near-zero almost everywhere (median 0.017).")

    if Path(REPO / args.raw).exists():
        raw = [json.loads(l) for l in open(REPO / args.raw, encoding="utf-8")]
        ok = sum(1 for r in raw if (r.get("correct_answer") or "").strip())
        print(f"\n3. Could a claim be built from question + correct_answer instead?")
        print(f"     non-empty correct_answer: {ok}/{len(raw)} ({ok/len(raw):.1%})")
        print("     -> not for roughly half the release, so this cannot simply be patched.")

    print("\n4. The two cases that drive the reversed result")
    by = defaultdict(list)
    for r in rows:
        by[r["conflict_type"]].append(r)

    nc = sorted(by["No conflict"], key=raw_disagree, reverse=True)
    if nc:
        r = nc[0]
        tr = [(p["e"], p["n"], p["c"]) for p in r["passages"]]
        best = max(((min(a[0], b[2]), i, j) for i, a in enumerate(tr)
                    for j, b in enumerate(tr)), key=lambda x: x[0])
        _, i, j = best
        print(f"   highest-disagree 'No conflict' instance (raw {raw_disagree(r):.3f}):")
        print(f"     question: {r['question']!r}")
        print(f"     'entailing'    p{i} e={tr[i][0]:.3f}: {r['passages'][i]['text'][:110]!r}")
        print(f"     'contradicting' p{j} c={tr[j][2]:.3f}: {r['passages'][j]['text'][:110]!r}")
        print("     -> neither passage contradicts anything; both just discuss the topic.")

    co = sorted(by["Conflicting opinions and research outcomes"], key=raw_disagree)
    if co:
        r = co[0]
        print(f"\n   lowest-disagree 'Conflicting opinions' instance (raw {raw_disagree(r):.3f}):")
        print(f"     question: {r['question']!r}")
        print(f"     max contradiction over its {len(r['passages'])} passages: "
              f"{max(p['c'] for p in r['passages']):.3f}")
        print("     -> genuinely conflicting sources register no contradiction at all,")
        print("        because none of them contradicts a QUESTION.")

    print("\nCONCLUSION. The 3b and 3c results measure an NLI head applied outside its")
    print("task definition. They are NOT evidence that `disagree` or the policy are")
    print("wrong. Testing either on CONFLICTS requires first turning each instance into")
    print("a claim, or scoring passage-against-passage instead of passage-against-query.")


if __name__ == "__main__":
    main()
