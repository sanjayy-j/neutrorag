"""Derive the r_v2 relevance floor from the Phase 3 retrievals. CORPUS ONLY.

Addendum 5 registers the floor as "the MEDIAN relevance across all passages
retrieved in the Phase 3 run", fixed before the experiment so it is a property
of the corpus rather than a parameter tuned to an outcome.

This script reads **no** accuracy, no EM, no label and no gate decision. It
reconstructs the Phase 3 sample deterministically from its committed seed,
re-retrieves, and reports the relevance distribution. It is the same category of
computation as experiments/describe_hotpot.py and derive_targets.py: a
corpus statistic used to set a parameter in advance.

It needs the embedding model, because no committed artefact stores per-passage
relevance -- only `max_relevance` per question, which is a median of maxima and
biased upward (0.8286 against the true per-passage median reported below).

    python experiments/derive_relevance_floor.py

Why the floor must move at all: the existing floor of 0.3 never bound on any of
the 1200 base-depth passages. DenseRetriever remaps cosine from [-1, 1] to
[0, 1], so `relevance >= 0.3` means `cos >= -0.4`, which essentially every text
pair satisfies. A floor that never binds makes the count-based component a
constant.
"""
from __future__ import annotations

import csv
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neutrorag.hotpot import load_hotpot_hf, stratified_sample  # noqa: E402
from neutrorag.retrieval import DenseRetriever, Document  # noqa: E402

PHASE3_SEED, PHASE3_N, PHASE3_K = 20261019, 400, 3
GATE_CSV = REPO / "results" / "hotpot_gate.csv"


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def quantiles(v):
    s = sorted(v)
    def q(p):
        return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]
    return q(0.0), q(0.25), q(0.5), q(0.75), q(1.0)


def main() -> None:
    print("=" * 78)
    print("r_v2 RELEVANCE FLOOR -- derived from corpus statistics only")
    print("=" * 78)
    print(f"commit {commit()}   no accuracy, no label, no gate decision is read")

    sample = stratified_sample(load_hotpot_hf(), PHASE3_N, seed=PHASE3_SEED)
    print(f"\nreconstructed Phase 3 sample: n={len(sample)} seed={PHASE3_SEED}")

    # verify we rebuilt the same sample the Phase 3 run used
    if GATE_CSV.exists():
        used = {r["qid"] for r in csv.DictReader(open(GATE_CSV, encoding="utf-8"))}
        got = {e.qid for e in sample}
        print(f"matches the committed Phase 3 qids: "
              f"{'YES' if got == used else 'NO -- ABORT'}"
              f"  ({len(got & used)}/{len(used)} shared)")
        if got != used:
            raise SystemExit("ABORT: could not reproduce the Phase 3 sample")

    base, allp = [], []
    for ex in sample:
        r = DenseRetriever()
        r.index([Document(p.title, p.text) for p in ex.paragraphs])
        base += [g.relevance for g in r.retrieve(ex.question, k=PHASE3_K)]
        allp += [g.relevance for g in r.retrieve(ex.question, k=len(ex.paragraphs))]

    for label, v, note in (
        (f"base depth k={PHASE3_K}", base,
         "the set the 1200-passage claim refers to, and the depth at which "
         "support_frac is first evaluated"),
        ("all paragraphs", allp, "for transparency; a wider set"),
    ):
        lo, q1, med, q3, hi = quantiles(v)
        print(f"\n{label}: {len(v)} passages")
        print(f"  min {lo:.4f}  Q1 {q1:.4f}  MEDIAN {med:.4f}  Q3 {q3:.4f}  "
              f"max {hi:.4f}")
        print(f"  mean {statistics.mean(v):.4f}  sd {statistics.pstdev(v):.4f}")
        print(f"  ({note})")
        print(f"  fraction clearing the OLD floor 0.3 : "
              f"{sum(1 for x in v if x >= 0.3) / len(v):.4f}")
        print(f"  fraction clearing a floor at this median : "
              f"{sum(1 for x in v if x >= med) / len(v):.4f}")

    lo, q1, med, q3, hi = quantiles(base)
    print("\n" + "=" * 78)
    print("FLOOR TO FIX IN ADDENDUM 5")
    print("=" * 78)
    print(f"  RELEVANCE_FLOOR_V2 = {med:.6f}")
    print(f"\n  Chosen as the median of the base-depth (k={PHASE3_K}) relevances,")
    print("  because Addendum 5 requires support_frac to START near 0.5 with")
    print("  variance in both directions, and base depth is where it is first")
    print("  evaluated. By construction support_frac is then ~0.5 at k=3 on the")
    print("  Phase 3 corpus, and can move up or down as retrieval changes.")
    print("\n  Sanity: support_frac at k=3 takes values in {0, 1/3, 2/3, 1}, so")
    print("  a median-set floor puts the modal question at 1/3 or 2/3 rather")
    print("  than pinning every question at 1.0 as the 0.3 floor did.")
    print("\n  Once recorded in Addendum 5 this value is FIXED and must not be")
    print("  adjusted afterwards for any reason.")


if __name__ == "__main__":
    main()
