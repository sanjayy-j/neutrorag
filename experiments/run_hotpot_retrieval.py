"""Track B retrieval metrics on HotpotQA distractor.

Properties of the retriever alone -- no reader, no NLI, no policy -- so these
are shared by every condition in Phases 3 and 5 and are reported once.

    python experiments/run_hotpot_retrieval.py
    python experiments/run_hotpot_retrieval.py --n 400 --seed 20261019

Metrics, all against the published gold supporting facts:

  Recall@k        fraction of an instance's gold PARAGRAPHS inside the top k,
                  averaged over instances. Gold is always exactly 2, so this is
                  0, 0.5 or 1 per instance.
  MRR             mean reciprocal rank of the FIRST gold paragraph.
  nDCG@10         binary gain on gold paragraphs, ideal ordering = both gold
                  first.
  context precision@k   fraction of the retrieved top k that is gold.
  context recall@k      paragraph-level: same quantity as Recall@k, reported
                  under its RAG-evaluation name so the two vocabularies line up.
  sentence recall@k     fraction of gold SENTENCES contained in the retrieved
                  paragraphs. Reported separately because supporting facts are
                  published per sentence while retrieval here is per paragraph,
                  and conflating the two overstates what retrieval delivered.

Recall@10 is 1.0 for every instance by construction: all of an instance's
paragraphs are indexed, so retrieving ten of at most ten retrieves the gold. It
is an index-completeness check, not a retrieval result.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neutrorag.hotpot import (  # noqa: E402
    load_hotpot_hf, stratified_sample, type_counts,
)
from neutrorag.retrieval import DenseRetriever, Document  # noqa: E402

KS = (1, 2, 3, 5, 10)
#: Fixed before any run and recorded in PREREGISTRATION.md.
SAMPLE_SEED = 20261019
SAMPLE_N = 400


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=SAMPLE_N)
    p.add_argument("--seed", type=int, default=SAMPLE_SEED)
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def wald(p: float, n: int):
    se = math.sqrt(p * (1 - p) / n) if n else 0.0
    return se, max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def main() -> None:
    args = parse_args()
    print("=" * 76)
    print("Track B retrieval metrics -- HotpotQA distractor, validation")
    print("=" * 76)

    allex = load_hotpot_hf()
    sample = stratified_sample(allex, args.n, seed=args.seed)
    print(f"corpus     : {len(allex)} parsed instances")
    print(f"sample     : {len(sample)}  seed={args.seed}  {type_counts(sample)}")
    print(f"retriever  : DenseRetriever(all-MiniLM-L6-v2), paragraph-level")
    print(f"commit     : {git_commit()}\n")

    rows: List[dict] = []
    for ex in sample:
        r = DenseRetriever()
        r.index([Document(p.title, p.text) for p in ex.paragraphs])
        got = r.retrieve(ex.question, k=len(ex.paragraphs))
        # DenseRetriever.retrieve returns Evidence(text, relevance) and does NOT
        # propagate the Document id, so identity has to be recovered by text.
        # (That dropped field is also why `diversity` is inert on this corpus:
        # the scorer receives sources=None. Noted in RESULTS.md.)
        title_of = {p.text: p.title for p in ex.paragraphs}
        ranked = [title_of[e.text] for e in got]
        gold = ex.gold_titles
        ranks = [i + 1 for i, t in enumerate(ranked) if t in gold]

        row: Dict[str, object] = {
            "qid": ex.qid, "qtype": ex.qtype,
            "n_paragraphs": len(ex.paragraphs), "n_gold": len(gold),
            "first_gold_rank": min(ranks) if ranks else 0,
            "mrr": 1.0 / min(ranks) if ranks else 0.0,
        }
        for k in KS:
            topk = ranked[:k]
            hit = sum(1 for t in topk if t in gold)
            row[f"recall@{k}"] = hit / len(gold)
            row[f"precision@{k}"] = hit / max(1, len(topk))
            # sentence-level: a gold sentence counts only if its paragraph is in
            # the top k
            got_sents = sum(1 for (t, _s) in ex.gold_sentences if t in set(topk))
            row[f"sent_recall@{k}"] = (got_sents / len(ex.gold_sentences)
                                       if ex.gold_sentences else 0.0)
        gains = [1.0 if t in gold else 0.0 for t in ranked[:10]]
        ideal = sorted(gains, reverse=True)
        row["ndcg@10"] = dcg(gains) / dcg(ideal) if dcg(ideal) else 0.0
        rows.append(row)

    n = len(rows)

    def mean(col):
        return statistics.mean(r[col] for r in rows)

    print("-" * 76)
    print(f"{'k':>3} {'Recall@k':>10} {'95% CI':>18} {'CtxPrec@k':>10} {'SentRec@k':>10}")
    print("-" * 76)
    for k in KS:
        rc = mean(f"recall@{k}")
        _se, lo, hi = wald(rc, n)
        print(f"{k:>3} {rc:>10.4f} {f'[{lo:.3f}, {hi:.3f}]':>18} "
              f"{mean(f'precision@{k}'):>10.4f} {mean(f'sent_recall@{k}'):>10.4f}")
    print("-" * 76)
    print(f"  MRR (first gold paragraph) : {mean('mrr'):.4f}")
    print(f"  nDCG@10                    : {mean('ndcg@10'):.4f}")

    perfect10 = sum(1 for r in rows if r["recall@10"] == 1.0)
    print(f"\n  index completeness check: Recall@10 == 1.0 on {perfect10}/{n} "
          f"instances {'(as expected)' if perfect10 == n else '(UNEXPECTED)'}")
    both2 = sum(1 for r in rows if r["recall@2"] == 1.0)
    none2 = sum(1 for r in rows if r["recall@2"] == 0.0)
    print(f"  at k=2: both gold retrieved on {both2}/{n} ({100*both2/n:.1f}%), "
          f"neither on {none2}/{n} ({100*none2/n:.1f}%)")

    # ---- by question type ------------------------------------------------
    print("\n" + "-" * 76)
    print("by question type")
    print("-" * 76)
    print(f"{'type':>12} {'n':>5} {'R@2':>8} {'R@3':>8} {'R@5':>8} {'MRR':>8} {'nDCG@10':>9}")
    for qt in sorted({r["qtype"] for r in rows}):
        sub = [r for r in rows if r["qtype"] == qt]
        m = lambda c: statistics.mean(r[c] for r in sub)  # noqa: E731
        print(f"{qt:>12} {len(sub):>5} {m('recall@2'):>8.4f} {m('recall@3'):>8.4f} "
              f"{m('recall@5'):>8.4f} {m('mrr'):>8.4f} {m('ndcg@10'):>9.4f}")

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "hotpot_retrieval.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
