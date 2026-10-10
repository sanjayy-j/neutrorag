"""Cache passage-vs-passage NLI for CONFLICTS (Decision 1, pairwise mode).

The previous cache (score_conflicts.py) holds passage-vs-QUESTION outputs, which
is the ill-posed call pairwise mode exists to replace, so this cannot reuse it.

COST, measured before running rather than estimated: 4.48s per passage-vs-passage
call on CPU (passage-vs-question was 3.6s; both texts are long here). Cost is
k(k-1) per instance, so an uncapped k=9 instance is 72 calls = 323s and the full
453-instance release would be ~40 hours.

SAMPLING, fixed in advance:
  max_passages = 5  -> 20 ordered pairs -> ~90s per instance
  40 Conflicting opinions and research outcomes   (of 115)
  40 No conflict                                  (of 161)
  10 Complementary information                    (of 115)  context only
  10 Conflict due to outdated information         (of 62)   context only
   0 Conflict due to misinformation               (of 5)    excluded, too few
  = 100 instances, ~2.5 hours.

The two mapped types get the large shares because they carry the expected
actions and drive the stop conditions; the other two are for the disagree-by-type
table. n=40 per class gives a Wald half-width near 0.155 on a proportion of 0.5,
which is wide, and is reported rather than hidden.

    python experiments/score_conflicts_pairwise.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neutrorag.conflicts import load_conflicts  # noqa: E402
from neutrorag.core import clamp  # noqa: E402

QUOTA = {
    "Conflicting opinions and research outcomes": 40,
    "No conflict": 40,
    "Complementary information": 10,
    "Conflict due to outdated information": 10,
}
MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MAX_PASSAGES = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/conflicts.jsonl")
    p.add_argument("--out", default="data/conflicts_pairwise.jsonl")
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--max-passages", type=int, default=MAX_PASSAGES)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    examples = load_conflicts(str(REPO / args.data))
    by_type = defaultdict(list)
    for ex in examples:
        by_type[ex.conflict_type].append(ex)

    rng = random.Random(args.seed)
    selected = []
    for ctype, k in QUOTA.items():
        pool = list(by_type.get(ctype, []))
        rng.shuffle(pool)
        selected.extend(pool[:k])
    rng.shuffle(selected)

    n_pairs = sum(min(len(e.evidence), args.max_passages) *
                  (min(len(e.evidence), args.max_passages) - 1) for e in selected)
    print(f"instances {len(selected)}  ordered pairs {n_pairs}  "
          f"projected {n_pairs * 4.48 / 3600:.2f} h", flush=True)

    from transformers import pipeline
    pipe = pipeline("text-classification", model=MODEL, top_k=None, device="cpu",
                    truncation=True, max_length=512)

    out_path = REPO / args.out
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_header": True, "model": MODEL,
                             "mode": "pairwise", "max_passages": args.max_passages,
                             "sampling": {"seed": args.seed, "quota": QUOTA},
                             "instances": len(selected), "pairs": n_pairs}) + "\n")
        for done, ex in enumerate(selected, 1):
            kept = sorted([e for e in ex.evidence if e.text.strip()],
                          key=lambda e: -clamp(e.relevance))[:args.max_passages]
            idx = [(i, j) for i in range(len(kept)) for j in range(len(kept)) if i != j]
            inputs = [{"text": kept[i].text, "text_pair": kept[j].text} for i, j in idx]
            outs = pipe(inputs, batch_size=args.batch_size) if inputs else []
            pairs = []
            for (i, j), out in zip(idx, outs):
                s = {d["label"].lower(): d["score"] for d in out}
                e, n, c = (s.get("entailment", 0.0), s.get("neutral", 0.0),
                           s.get("contradiction", 0.0))
                tot = e + n + c or 1.0
                pairs.append({"i": i, "j": j, "e": e / tot, "n": n / tot, "c": c / tot})
            fh.write(json.dumps({
                "question": ex.question, "conflict_type": ex.conflict_type,
                "source": ex.source,
                "passages": [{"text": e.text, "relevance": e.relevance,
                              "source": e.source} for e in kept],
                "pairs": pairs}) + "\n")
            fh.flush()
            if done % 5 == 0 or done == len(selected):
                el = time.time() - t0
                print(f"  {done}/{len(selected)}  ({el/60:.1f} min elapsed, "
                      f"~{el/done*(len(selected)-done)/60:.1f} min left)", flush=True)

    print(f"\nwrote {out_path} ({len(selected)} instances, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
