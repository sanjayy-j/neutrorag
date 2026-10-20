"""Stage 1a of Addendum 6: how hard must retrieval be to bind?

Report 7.7.4 concludes the decomposition fails because r predicts retrieval
adequacy while answer correctness is decoupled from it -- retrieval is not the
binding constraint. That is falsifiable: make retrieval the binding constraint
and r should begin to predict answer correctness.

The HotpotQA distractor setting hands each question its own 10 paragraphs, 2 of
them gold, so Recall@10 is 1.0 by construction and retrieval is easy. This
builds ONE pooled dense index over the paragraphs of N questions and retrieves
from that instead, which makes retrieval progressively harder as N grows.

Sweeps N over {250, 500, 1000, 2000} and reports Recall@3 and Recall@10.
Selection rule, fixed in advance: take the SMALLEST N where

    Recall@3 <= 0.6 * Recall@10

so retrieval genuinely limits answers while expansion still has headroom.

RETRIEVAL ONLY. No reader, no NLI, no gate, no accuracy, no label beyond the
published gold supporting facts.

    python experiments/sweep_pool_size.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from neutrorag.hotpot import load_hotpot_hf, stratified_sample  # noqa: E402
from neutrorag.retrieval import SentenceTransformerBackend  # noqa: E402

NS = (250, 500, 1000, 2000)
MAX_N = max(NS)
SWEEP_SEED = 20261020          # the superset draw; the experiment seed differs
RATIO = 0.6


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print("STAGE 1a -- pooled-index size sweep. RETRIEVAL ONLY, no accuracy.")
    print("=" * 78)
    print(f"commit {commit()}   selection rule: smallest N with "
          f"Recall@3 <= {RATIO} * Recall@10")

    allex = load_hotpot_hf()
    # One nested superset, so the sweep is monotone in N: the first 250 are a
    # subset of the first 500, and so on.
    pool_qs = stratified_sample(allex, MAX_N, seed=SWEEP_SEED)
    print(f"\nsuperset draw: {len(pool_qs)} questions, seed {SWEEP_SEED}")

    # Deduplicate paragraphs by title: different questions share articles, and
    # a pooled index must hold each article once.
    title_to_text: dict = {}
    for ex in pool_qs:
        for p in ex.paragraphs:
            title_to_text.setdefault(p.title, p.text)
    titles = sorted(title_to_text)
    print(f"distinct paragraphs across the superset: {len(titles)}")

    emb = SentenceTransformerBackend()
    print("\nembedding paragraphs once, then slicing per N ...")
    P = emb.embed([title_to_text[t] for t in titles])
    P = np.asarray(P, dtype=np.float32)
    P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    print(f"  paragraph matrix {P.shape}  ({time.time() - t0:.0f}s)")

    Q = emb.embed([ex.question for ex in pool_qs])
    Q = np.asarray(Q, dtype=np.float32)
    Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    print(f"  question matrix  {Q.shape}  ({time.time() - t0:.0f}s)")

    idx_of = {t: i for i, t in enumerate(titles)}

    print("\n" + "-" * 78)
    print(f"{'N':>6} {'pool size':>10} {'Recall@3':>10} {'Recall@10':>10} "
          f"{'ratio':>8} {'rule met':>9}")
    print("-" * 78)
    rows = []
    for n in NS:
        qs = pool_qs[:n]
        # the index is exactly the paragraphs those n questions bring
        sub_titles = sorted({p.title for ex in qs for p in ex.paragraphs})
        cols = np.array([idx_of[t] for t in sub_titles])
        Psub = P[cols]
        r3 = r10 = 0.0
        for j, ex in enumerate(qs):
            sims = Psub @ Q[j]
            top = np.argsort(-sims)
            gold = ex.gold_titles
            ranked = [sub_titles[i] for i in top[:10]]
            r3 += sum(1 for t in ranked[:3] if t in gold) / len(gold)
            r10 += sum(1 for t in ranked[:10] if t in gold) / len(gold)
        r3 /= len(qs)
        r10 /= len(qs)
        met = r3 <= RATIO * r10
        rows.append((n, len(sub_titles), r3, r10, met))
        print(f"{n:>6} {len(sub_titles):>10} {r3:>10.4f} {r10:>10.4f} "
              f"{r3 / r10 if r10 else float('nan'):>8.4f} "
              f"{'YES' if met else 'no':>9}")

    print("-" * 78)
    chosen = next((r for r in rows if r[4]), None)
    print()
    if chosen is None:
        print("*** NO N SATISFIES THE RULE. Sweep reported above; stopping for")
        print("*** a decision rather than picking an N that does not meet it.")
    else:
        n, size, r3, r10, _ = chosen
        print(f"FIXED: N = {n}   pooled index = {size} paragraphs")
        print(f"       Recall@3 = {r3:.4f}   Recall@10 = {r10:.4f}   "
              f"ratio {r3 / r10:.4f}")
        print(f"\n  Retrieval now binds: only {r3:.1%} of gold reaches the top 3,")
        print(f"  against {r10:.1%} at depth 10, so expansion has real headroom")
        print("  and answers are limited by what retrieval finds.")
        print("\n  Record N, both recalls and the superset seed in Addendum 6.")
    print(f"\nwall-clock {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
