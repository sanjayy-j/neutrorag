"""Memo F10: is the FEVER evaluation set contaminated by the NLI model's training data?

MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli is fine-tuned on MNLI + FEVER-NLI
+ ANLI, and we evaluate it on a FEVER validation split. This checks, at both
claim-text and claim-ID level, whether the evaluated claims appear in FEVER-NLI's
train or dev splits.

FEVER-NLI is read from `pietrolesci/nli_fever` (a community re-host of Nie et
al. 2019); in that dataset the CLAIM is in the `premise` field and `cid` is the
FEVER claim id, which lines up with `original_id` in copenlu/fever_gold_evidence.

    python experiments/check_contamination.py
    python experiments/check_contamination.py --limit 200

Requires network on first run (downloads both datasets, then cached).
"""
from __future__ import annotations

import argparse
import re

from datasets import load_dataset

EVAL_DATASET = "copenlu/fever_gold_evidence"
EVAL_SPLIT = "validation"
NLI_FEVER = "pietrolesci/nli_fever"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=200,
                   help="how many rows of the evaluation split to check")
    return p.parse_args()


def normalise(s: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace, so trivial
    formatting differences don't hide a genuine overlap."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", s.lower())).strip()


def main() -> None:
    args = parse_args()

    evaluated = load_dataset(EVAL_DATASET, split=f"{EVAL_SPLIT}[:{args.limit}]")
    eval_text = {normalise(r["claim"]) for r in evaluated}
    eval_ids = {int(r["original_id"]) for r in evaluated}
    print(f"evaluated: {len(evaluated)} rows from {EVAL_DATASET}[{EVAL_SPLIT}]")
    print(f"           {len(eval_text)} distinct normalised claim texts, "
          f"{len(eval_ids)} distinct claim ids")

    for split in ("train", "dev"):
        ds = load_dataset(NLI_FEVER, split=split)
        texts, cids = set(), set()
        for row in ds:
            texts.add(normalise(row["premise"]))  # `premise` holds the CLAIM here
            cids.add(int(row["cid"]))
        t_over = eval_text & texts
        c_over = eval_ids & cids
        print(f"\n{NLI_FEVER}[{split}]: {len(ds)} rows, {len(texts)} distinct claims")
        print(f"  claim-TEXT overlap: {len(t_over)}/{len(eval_text)} "
              f"({len(t_over)/len(eval_text):.1%})")
        print(f"  claim-ID   overlap: {len(c_over)}/{len(eval_ids)} "
              f"({len(c_over)/len(eval_ids):.1%})")
        for example in list(t_over)[:3]:
            print(f"    e.g. {example[:88]}")

    print("\nInterpretation: overlap with TRAIN would be disqualifying. Overlap with "
          "DEV means the evaluation set is the split the model's own card reports on "
          "('the dev set for Fever-NLI', 0.777) -- disclose it, and do not present "
          "numbers measured there as held-out.")


if __name__ == "__main__":
    main()
