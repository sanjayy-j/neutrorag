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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neutrorag import HFNLIBackend, NLIRetrievalScorer  # noqa: E402
from neutrorag.eval import (  # noqa: E402
    failure_type_attribution, format_confusion, fuzzy_collapse_ablation,
)
from neutrorag.fever import load_fever_hf, score_examples  # noqa: E402

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_DATASET = "copenlu/fever_gold_evidence"
DEFAULT_SPLIT = "validation"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=200, help="number of FEVER claims to score")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace NLI model id")
    p.add_argument("--device", type=str, default="cpu", help="'cpu', 'cuda', or 'cuda:0' etc.")
    return p.parse_args()


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

    scorer = NLIRetrievalScorer(nli=HFNLIBackend(model=args.model, device=args.device))

    t0 = time.time()
    tifs, gold = score_examples(examples, scorer)
    print(f"Scored {len(tifs)} claims with real NLI in {time.time() - t0:.1f}s")

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
