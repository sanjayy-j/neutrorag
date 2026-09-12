"""Real FEVER premise-check experiment (real NLI model, real dataset).

Unlike ``demo.py`` -- which uses the deterministic mock NLI backend purely as an
offline scaffold -- this script downloads a real HuggingFace NLI model and real
FEVER claims (via ``neutrorag.fever.load_fever_hf``, see that module's docstring
for the exact dataset and evidence-format gotcha) and reports genuine
failure-type attribution numbers plus the fuzzy-collapse ablation.

    python experiments/run_fever_eval.py --limit 200
    python experiments/run_fever_eval.py --limit 50 --model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli --device cpu

Requires the optional NLI extras: pip install -e ".[nli]" (plus `datasets`).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neutrorag import HFNLIBackend, NLIRetrievalScorer  # noqa: E402
from neutrorag.eval import (  # noqa: E402
    failure_type_attribution, format_confusion, fuzzy_collapse_ablation,
)
from neutrorag.fever import load_fever_hf  # noqa: E402
from neutrorag.core import clamp  # noqa: E402

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_DATASET = "copenlu/fever_gold_evidence"
DEFAULT_SPLIT = "validation"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=200, help="number of FEVER claims to score")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace NLI model id")
    p.add_argument("--device", type=str, default="cpu", help="'cpu', 'cuda', or 'cuda:0' etc.")
    p.add_argument("--csv", type=str, default=None,
                   help="dump per-example r,e,n,c,T,I,F + gold/predicted to this CSV")
    return p.parse_args()


class _RecordingNLI:
    """Transparent proxy recording every (e, n, c) the wrapped backend returns.

    Instrumentation only: values pass through unmodified, so the TIF triples
    produced downstream are bit-identical to an uninstrumented run. Needed
    because NLIRetrievalScorer.score aggregates e/n/c internally and returns
    only the TIF -- and the scoring formula is not to be modified.
    """

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def predict(self, premise: str, hypothesis: str):
        out = self.inner.predict(premise, hypothesis)
        self.calls.append(out)
        return out


def score_with_instrumentation(examples, scorer, recorder):
    """Score every example, capturing the intermediate r/e/n/c alongside the TIF.

    The TIF is whatever NLIRetrievalScorer.score returns (authoritative); r and
    the pooled e/n/c are recomputed from the recorded per-passage NLI outputs
    using the same relevance-weighted mean the scorer uses, and cross-checked
    against the scorer's own T/I/F before anything is written out.
    """
    tifs, gold, rows = [], [], []
    for ex in examples:
        recorder.calls.clear()
        tif = scorer.score(ex.claim, ex.evidence)
        per_item = list(recorder.calls)

        kept = [e for e in ex.evidence if e.text.strip()]
        weights = [clamp(e.relevance) for e in kept]
        if per_item and weights:
            wsum = sum(weights) or 1.0
            e = sum(w * x[0] for w, x in zip(weights, per_item)) / wsum
            n = sum(w * x[1] for w, x in zip(weights, per_item)) / wsum
            c = sum(w * x[2] for w, x in zip(weights, per_item)) / wsum
            r = clamp(max(weights))
        else:  # empty-evidence short-circuit in scorer.score -> no NLI call made
            e = n = c = r = ""

        tifs.append(tif)
        gold.append(ex.gold_channel)
        rows.append({
            "claim": ex.claim,
            "r": r, "e": e, "n": n, "c": c,
            "T": tif.T, "I": tif.I, "F": tif.F,
            "gold_label": ex.gold_label,
            "gold_channel": ex.gold_channel,
            "predicted_channel": tif.dominant,
            "evidence_count": len(kept),
        })
    return tifs, gold, rows


def write_csv(rows, path: str) -> None:
    cols = ["claim", "r", "e", "n", "c", "T", "I", "F",
            "gold_label", "gold_channel", "predicted_channel", "evidence_count"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    args = parse_args()

    print("NeutroRAG FEVER premise check  --  REAL NLI model + REAL FEVER data")
    print("(unlike demo.py, these numbers are NOT from the mock NLI backend)")
    print(f"  model   : {args.model}")
    print(f"  device  : {args.device}")
    print(f"  dataset : {DEFAULT_DATASET} [{DEFAULT_SPLIT}]  (via neutrorag.fever.load_fever_hf)")
    print(f"  N       : {args.limit}")

    t0 = time.time()
    examples = load_fever_hf(split=DEFAULT_SPLIT, max_examples=args.limit)
    print(f"\nLoaded {len(examples)} real FEVER examples in {time.time() - t0:.1f}s")

    recorder = _RecordingNLI(HFNLIBackend(model=args.model, device=args.device))
    scorer = NLIRetrievalScorer(nli=recorder)

    t0 = time.time()
    tifs, gold, rows = score_with_instrumentation(examples, scorer, recorder)
    print(f"Scored {len(tifs)} claims with real NLI in {time.time() - t0:.1f}s")

    # Cross-check: the recomputed aggregates must reproduce the scorer's own
    # T/I/F, otherwise the instrumentation is not measuring the real pipeline.
    mismatches = 0
    for row, tif in zip(rows, tifs):
        if row["r"] == "":
            continue
        r, e, n, c = row["r"], row["e"], row["n"], row["c"]
        exp_T = clamp(r * e)
        exp_F = clamp(r * c + scorer.lam * clamp(2.0 * min(e, c)))
        exp_I = clamp(1.0 - r + r * n)
        if max(abs(exp_T - tif.T), abs(exp_I - tif.I), abs(exp_F - tif.F)) > 1e-9:
            mismatches += 1
    print(f"Instrumentation cross-check: {mismatches} mismatch(es) vs scorer output"
          f" (expect 0)")

    if args.csv:
        write_csv(rows, args.csv)
        print(f"Wrote per-example CSV: {args.csv}  ({len(rows)} rows)")

    hr("Failure-type attribution (real NLI)")
    attr = failure_type_attribution(tifs, gold)
    print(f"Accuracy: {attr['accuracy']:.3f}  (argmax(T,I,F) vs gold SUPPORTS/NEI/REFUTES)\n")
    print(format_confusion(attr["confusion"]))

    hr("Fuzzy-collapse ablation (full T,I,F vs I+F merged)")
    ab = fuzzy_collapse_ablation(tifs, gold)
    print(f"  full neutrosophic (T,I,F) attribution : {ab.full_attribution:.3f}")
    print(f"  collapsed (I+F merged) attribution    : {ab.collapsed_attribution:.3f}")
    print(f"  delta (value of the I/F distinction)  : {ab.delta:+.3f}")


if __name__ == "__main__":
    main()
