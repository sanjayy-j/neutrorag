"""FEVER -> (T, I, F) proof-of-concept.

FEVER labels each claim SUPPORTS / REFUTES / NOT ENOUGH INFO, which map exactly
onto the neutrosophic channels T / F / I. So FEVER is the cleanest possible test
of the NeutroRAG premise: if the (T, I, F) scorer's dominant channel matches the
gold label, the core hypothesis holds.

A small synthetic FEVER-style sample is embedded so this runs offline. Replace
``load_sample()`` with :func:`load_fever_hf` (real dataset, HuggingFace Hub) or
:func:`load_fever_jsonl` (real dataset, your own JSONL) to score the real thing.

**The FEVER evidence gotcha.** Raw FEVER (Thorne et al., 2018) records evidence
only as ``(wikipedia_page, sentence_id)`` pointers -- the sentence *text* isn't
in the dataset at all, so scoring it requires joining against a ~5M-sentence
Wikipedia dump the annotators used, which is out of scope for a premise check.
:func:`load_fever_hf` sidesteps this by loading
`copenlu/fever_gold_evidence <https://huggingface.co/datasets/copenlu/fever_gold_evidence>`_
from the HuggingFace Hub, which already bundles each claim with the resolved
evidence *text* inline: gold evidence sentences for SUPPORTS/REFUTES (from the
original FEVER annotations), and a system-retrieved, topically-related-but-
non-confirming sentence for NOT ENOUGH INFO (from the adversarial-claims
pipeline in Atanasova et al., EMNLP 2020). Each row's ``evidence`` field is a
list of ``[wikipedia_page, sentence_id, sentence_text]`` triples.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .scoring import Evidence, NLIRetrievalScorer

LABEL_TO_CHANNEL = {"SUPPORTS": "T", "NOT ENOUGH INFO": "I", "REFUTES": "F"}


@dataclass
class FeverExample:
    claim: str
    evidence: List[Evidence]
    gold_label: str  # SUPPORTS | REFUTES | NOT ENOUGH INFO

    @property
    def gold_channel(self) -> str:
        return LABEL_TO_CHANNEL[self.gold_label]


def load_sample() -> List[FeverExample]:
    """A tiny, balanced, embedded sample (offline). Illustrative, not for the paper."""
    def ev(*texts: str) -> List[Evidence]:
        return [Evidence(t, relevance=0.9) for t in texts]

    return [
        # --- SUPPORTS (evidence entails the claim) ---
        FeverExample("The Eiffel Tower is located in Paris.",
                     ev("The Eiffel Tower is a wrought-iron tower located in Paris, France."),
                     "SUPPORTS"),
        FeverExample("Water boils at 100 degrees Celsius at sea level.",
                     ev("At sea level water boils at 100 degrees Celsius."),
                     "SUPPORTS"),
        FeverExample("Python is a programming language.",
                     ev("Python is a high-level general-purpose programming language."),
                     "SUPPORTS"),
        FeverExample("The heart pumps blood.",
                     ev("The heart is a muscular organ that pumps blood through the body."),
                     "SUPPORTS"),
        # --- REFUTES (evidence contradicts the claim) ---
        FeverExample("The Great Wall of China is not visible from space.",
                     ev("The Great Wall of China is visible from space with the naked eye."),
                     "REFUTES"),
        FeverExample("Mount Everest is the shortest mountain on Earth.",
                     ev("Mount Everest is the tallest mountain on Earth above sea level."),
                     "REFUTES"),
        FeverExample("Sharks are mammals.",
                     ev("Sharks are not mammals; sharks are a group of fish."),
                     "REFUTES"),
        FeverExample("The sun revolves around the Earth.",
                     ev("The Earth revolves around the sun; the sun does not orbit the Earth."),
                     "REFUTES"),
        # --- NOT ENOUGH INFO (evidence is off-topic / absent) ---
        FeverExample("The mayor of Springfield won an award in 1998.",
                     ev("Springfield is a common place name used in many countries."),
                     "NOT ENOUGH INFO"),
        FeverExample("Company X increased its revenue last quarter.",
                     ev("Company X was founded several decades ago in the technology sector."),
                     "NOT ENOUGH INFO"),
        FeverExample("The novel sold ten million copies.",
                     ev("The novel was written over a period of three years."),
                     "NOT ENOUGH INFO"),
        FeverExample("The bridge was painted blue in 2005.",
                     ev("The bridge spans a wide river in the northern region."),
                     "NOT ENOUGH INFO"),
    ]


def load_fever_jsonl(path: str, max_examples: int | None = None) -> List[FeverExample]:
    """Load real FEVER-format JSONL. Expects fields: claim, label, and a list of
    evidence sentences under 'evidence_sentences' (adapt to your retriever)."""
    out: List[FeverExample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            label = row["label"].upper()
            if label not in LABEL_TO_CHANNEL:
                continue
            sents = row.get("evidence_sentences", []) or []
            out.append(FeverExample(
                claim=row["claim"],
                evidence=[Evidence(s, relevance=0.9) for s in sents],
                gold_label=label,
            ))
            if max_examples and len(out) >= max_examples:
                break
    return out


def _examples_from_rows(rows: Iterable[dict], max_examples: Optional[int] = None) -> List[FeverExample]:
    """Convert rows shaped like ``copenlu/fever_gold_evidence`` (claim, label,
    evidence=[[page, sentence_id, text], ...]) into :class:`FeverExample`.

    Split out from :func:`load_fever_hf` so the parsing logic can be unit-tested
    with an inline fixture, with no network call or dataset library involved.
    Unknown labels are skipped; malformed or empty evidence entries are dropped
    rather than raising, since NEI rows and dataset noise can legitimately lack
    usable evidence text.
    """
    out: List[FeverExample] = []
    for row in rows:
        label = row["label"].upper()
        if label not in LABEL_TO_CHANNEL:
            continue
        sents: List[str] = []
        for item in row.get("evidence") or []:
            if isinstance(item, (list, tuple)) and len(item) >= 3 and item[2]:
                sents.append(str(item[2]))
        out.append(FeverExample(
            claim=row["claim"],
            evidence=[Evidence(s, relevance=0.9) for s in sents],
            gold_label=label,
        ))
        if max_examples and len(out) >= max_examples:
            break
    return out


def load_fever_hf(
    split: str = "validation",
    max_examples: Optional[int] = None,
    dataset_name: str = "copenlu/fever_gold_evidence",
) -> List[FeverExample]:
    """Load real FEVER claims with resolved evidence text from the HuggingFace Hub.

    See the module docstring for why ``copenlu/fever_gold_evidence`` (rather
    than raw FEVER) is the pragmatic choice here. Requires the optional
    ``datasets`` dependency (kept out of the core library's import path).
    """
    from datasets import load_dataset  # noqa: local import -- optional, heavy dep

    split_expr = f"{split}[:{max_examples}]" if max_examples else split
    rows = load_dataset(dataset_name, split=split_expr)
    return _examples_from_rows(rows, max_examples=max_examples)


def score_examples(examples: Sequence[FeverExample], scorer: NLIRetrievalScorer):
    """Return (list_of_TIF, list_of_gold_channels)."""
    tifs = [scorer.score(ex.claim, ex.evidence) for ex in examples]
    gold = [ex.gold_channel for ex in examples]
    return tifs, gold
