"""Score CONFLICTS instances with the real NLI model and cache the raw outputs.

Separated from the analyses because the NLI pass is the expensive part: on CPU
this model costs ~3.6 s per (passage, question) pair on CONFLICTS-length text,
and the full release is 4212 pairs -- about 4.4 hours. Batching does not help
(measured 3.9 s/call at batch 8, 3.6 s/call at batch 16: the box is
compute-bound, not overhead-bound).

So this caches PER-PASSAGE (e, n, c) to JSONL, and every downstream experiment
replays from that cache. Per-passage matters: fever_run_200_gateA.csv stored only
the
POOLED triple, which silently made pairwise and pooled `disagree` identical on
replay. That cannot happen here.

SAMPLING. By default this scores a seeded stratified subsample rather than all
458 instances, purely for compute. The sample is weighted toward the two types
that carry an expected action, since those drive the policy test:

    Conflicting opinions and research outcomes : 45 of 115
    No conflict                                : 45 of 161
    Complementary information                  : 20 of 115
    Conflict due to outdated information       : 20 of 62
    Conflict due to misinformation             :  0 of 5   (excluded, too few)

Pass --all to score the whole release instead. The sampling procedure and seed
are recorded in the cache header so any number derived from it is traceable
(memo F17 asked for exactly this on the FEVER run).

    python experiments/score_conflicts.py                  # subsample, ~70 min
    python experiments/score_conflicts.py --all            # full release, ~4.4 h
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

from neutrorag.conflicts import (  # noqa: E402
    CONFLICT_TYPES, TOO_FEW_TO_EVALUATE, load_conflicts,
)

DEFAULT_QUOTA = {
    "Conflicting opinions and research outcomes": 45,
    "No conflict": 45,
    "Complementary information": 20,
    "Conflict due to outdated information": 20,
}
MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/conflicts.jsonl")
    p.add_argument("--out", default="data/conflicts_scored.jsonl")
    p.add_argument("--all", action="store_true", help="score every instance")
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def stratified_sample(examples, quota, seed):
    by_type = defaultdict(list)
    for ex in examples:
        by_type[ex.conflict_type].append(ex)
    rng = random.Random(seed)
    picked = []
    for ctype, k in quota.items():
        pool = by_type.get(ctype, [])
        rng.shuffle(pool)
        picked.extend(pool[:k])
    rng.shuffle(picked)
    return picked


def main() -> None:
    args = parse_args()
    t_start = time.time()

    print(f"loading {args.data} (embedding relevance; downloads MiniLM on first use)")
    examples = load_conflicts(str(REPO / args.data))
    print(f"  {len(examples)} instances")

    if args.all:
        selected = [e for e in examples if e.conflict_type not in TOO_FEW_TO_EVALUATE]
        sampling = {"mode": "all_except_too_few", "excluded": sorted(TOO_FEW_TO_EVALUATE)}
    else:
        selected = stratified_sample(examples, DEFAULT_QUOTA, args.seed)
        sampling = {"mode": "stratified", "seed": args.seed, "quota": DEFAULT_QUOTA}
    n_pairs = sum(len(e.evidence) for e in selected)
    print(f"  scoring {len(selected)} instances / {n_pairs} passages")

    from transformers import pipeline
    pipe = pipeline("text-classification", model=MODEL, top_k=None, device="cpu",
                    truncation=True, max_length=512)

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_header": True, "model": MODEL, "sampling": sampling,
                             "instances": len(selected), "passages": n_pairs}) + "\n")
        for ex in selected:
            inputs = [{"text": ev.text, "text_pair": ex.question} for ev in ex.evidence]
            outs = pipe(inputs, batch_size=args.batch_size) if inputs else []
            passages = []
            for ev, out in zip(ex.evidence, outs):
                s = {d["label"].lower(): d["score"] for d in out}
                e, n, c = (s.get("entailment", 0.0), s.get("neutral", 0.0),
                           s.get("contradiction", 0.0))
                tot = e + n + c or 1.0
                passages.append({"text": ev.text, "relevance": ev.relevance,
                                 "source": ev.source,
                                 "e": e / tot, "n": n / tot, "c": c / tot})
            fh.write(json.dumps({"question": ex.question,
                                 "conflict_type": ex.conflict_type,
                                 "source": ex.source,
                                 "passages": passages}) + "\n")
            fh.flush()
            done += 1
            if done % 5 == 0 or done == len(selected):
                el = time.time() - t_start
                rate = el / done
                print(f"  {done}/{len(selected)} instances  "
                      f"({el/60:.1f} min elapsed, ~{rate*(len(selected)-done)/60:.1f} min left)",
                      flush=True)

    print(f"\nwrote {out_path} ({done} instances, {time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
