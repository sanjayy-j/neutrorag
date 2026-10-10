"""CONFLICTS -> NeutroRAG evidence (memo Step 9).

Cattan et al., "DRAGged into CONFLICTS: Detecting and Addressing Conflicting
Sources in Search-Augmented LLMs", arXiv 2506.08500. Released at
https://github.com/google-research-datasets/rag_conflicts as a single
``conflicts.jsonl`` (458 instances, 46.7 MB).

Why this venue and not FEVER (memo F9): FEVER supplies one annotator-selected
gold passage per claim, so `r` cannot vary and `disagree` is structurally near
zero. CONFLICTS supplies real Google Search results -- 9.2 per question on
average, from 8.3 distinct domains -- and annotates each instance with a
conflict type. Two of those types have a directly corresponding NeutroRAG
action, which for the first time gives external ground truth for the POLICY
rather than only for the scorer's label attribution.

Schema, verified against the published release (not assumed)::

    {"source": str,
     "question": str,
     "search_results": [{"title", "url", "snippet", "date",
                         "response_str",   # full page text
                         "short_text"}],   # 512-token window
     "conflict_type": str,
     "correct_answer": str}

**There is no retrieval score in the release.** The union of every key appearing
under ``search_results`` across all 458 instances is exactly
``{date, response_str, short_text, snippet, title, url}`` -- no score, no rank,
no relevance. Since ``Evidence.relevance`` feeds the sufficiency components, a
value has to come from somewhere, and inventing a constant is precisely the F1
defect that started this whole review. See :data:`RELEVANCE_SOURCE` and
:func:`embedding_relevance` for what is used instead, and why.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

from .scoring import Evidence

# The five annotated conflict types, with their released counts.
CONFLICT_TYPES = {
    "No conflict": 161,
    "Complementary information": 115,
    "Conflicting opinions and research outcomes": 115,
    "Conflict due to outdated information": 62,
    "Conflict due to misinformation": 5,
}

# Only two of the five types define a behaviour that maps onto a NeutroRAG
# action, and this mapping is the external ground truth for the policy test.
# The other three are deliberately left unmapped: "Complementary information"
# and "outdated information" have expected behaviours (synthesise; prefer the
# recent source) that the five-way action space does not encode, and inventing
# a correspondence for them would be grading the policy against a target
# invented to fit it.
TYPE_TO_EXPECTED_ACTION = {
    "Conflicting opinions and research outcomes": "surface_contradiction",
    "No conflict": "answer",
}

# Excluded from every per-type claim: 5 instances cannot support one.
TOO_FEW_TO_EVALUATE = {"Conflict due to misinformation"}

# Saturation targets for CONFLICTS (Decision 2): p75 of this corpus's own depth
# distributions -- 10 passages, 9 distinct domains. At the previous target of 3
# both components were 100% saturated (std exactly 0.000), i.e. constant and
# useless. Derived in experiments/derive_targets.py from depth alone.
SUPPORT_TARGET = 10
DIVERSITY_TARGET = 9

RELEVANCE_SOURCE = """derived, not published.

The release carries no retrieval score, so relevance is computed here as the
cosine similarity between the question and the passage text, using the same
sentence-embedding model DenseRetriever uses (all-MiniLM-L6-v2), remapped from
[-1, 1] to [0, 1] exactly as neutrorag.retrieval.DenseRetriever.retrieve does.

Chosen over the obvious alternative, rank position, because the release does not
state that ``search_results`` is in ranked order; deriving a relevance from an
ordering that may be arbitrary would manufacture a signal rather than measure
one. Cosine similarity is at least a real measurement of the thing relevance is
supposed to mean, computed identically to the project's own retriever.

It is still a PROXY, and it is NLI-free by construction (the embedding model
never sees entailment labels), which keeps the Gate B route (i) constraint
intact. Any r-dependent result on this dataset inherits this choice."""


@dataclass
class ConflictExample:
    question: str
    evidence: List[Evidence]
    conflict_type: str
    correct_answer: str = ""
    source: str = ""

    @property
    def expected_action(self) -> Optional[str]:
        """The action the dataset's annotation implies, or None if unmapped."""
        return TYPE_TO_EXPECTED_ACTION.get(self.conflict_type)


def passage_text(result: dict) -> str:
    """Pick the text to score for one search result.

    ``short_text`` (the 512-token window) is preferred: it is present for 84% of
    results and is the right granularity for an NLI model that truncates anyway.
    ``snippet`` is only present for 64%, and ``response_str`` is the whole page
    (median 4671 characters), most of which is irrelevant to the question.
    Falls back snippet -> title so a result is dropped only when it has no text
    at all.
    """
    for field in ("short_text", "snippet", "title"):
        value = (result.get(field) or "").strip()
        if value:
            return value
    return ""


def result_domain(result: dict) -> str:
    """Registrable domain, used as the provenance key for the diversity component."""
    url = (result.get("url") or "").strip()
    if not url:
        return ""
    return urlparse(url).netloc.lower().removeprefix("www.")


RelevanceFn = Callable[[str, Sequence[str]], Sequence[float]]


def embedding_relevance(question: str, passages: Sequence[str]) -> List[float]:
    """Cosine similarity between the question and each passage, in [0, 1].

    See :data:`RELEVANCE_SOURCE`. Imports sentence-transformers lazily, so the
    module and its tests stay importable without it.
    """
    import numpy as np

    from .retrieval import SentenceTransformerBackend

    if not passages:
        return []
    backend = SentenceTransformerBackend()
    vecs = backend.embed([question, *passages])
    q, docs = vecs[0], vecs[1:]
    norms = np.linalg.norm(docs, axis=1) * np.linalg.norm(q) + 1e-9
    cos = (docs @ q) / norms
    return [float(x) for x in np.clip((cos + 1.0) / 2.0, 0.0, 1.0)]


def _examples_from_rows(
    rows: Iterable[dict],
    relevance_fn: RelevanceFn,
    max_examples: Optional[int] = None,
) -> List[ConflictExample]:
    """Convert released CONFLICTS rows into :class:`ConflictExample`.

    Split out from :func:`load_conflicts` so parsing is unit-testable with an
    inline fixture and an injected relevance function -- no file, no network, no
    embedding model. Mirrors fever._examples_from_rows.

    Results with no usable text are dropped; rows whose conflict_type is not one
    of the five released types are skipped rather than silently relabelled.
    """
    out: List[ConflictExample] = []
    for row in rows:
        ctype = (row.get("conflict_type") or "").strip()
        if ctype not in CONFLICT_TYPES:
            continue
        results = [s for s in (row.get("search_results") or []) if passage_text(s)]
        texts = [passage_text(s) for s in results]
        relevances = list(relevance_fn(row.get("question", ""), texts)) if texts else []
        evidence = [
            Evidence(text=t, relevance=float(rel), source=result_domain(s) or None)
            for t, rel, s in zip(texts, relevances, results)
        ]
        out.append(ConflictExample(
            question=row.get("question", ""),
            evidence=evidence,
            conflict_type=ctype,
            correct_answer=(row.get("correct_answer") or "").strip(),
            source=(row.get("source") or "").strip(),
        ))
        if max_examples and len(out) >= max_examples:
            break
    return out


def load_conflicts(
    path: str = "data/conflicts.jsonl",
    max_examples: Optional[int] = None,
    relevance_fn: Optional[RelevanceFn] = None,
) -> List[ConflictExample]:
    """Load the released conflicts.jsonl.

    Download it first (46.7 MB, gitignored under data/)::

        curl -sL -o data/conflicts.jsonl \\
          https://raw.githubusercontent.com/google-research-datasets/rag_conflicts/main/conflicts.jsonl
    """
    with open(path, "r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return _examples_from_rows(rows, relevance_fn or embedding_relevance, max_examples)


def type_counts(examples: Sequence[ConflictExample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ex in examples:
        counts[ex.conflict_type] = counts.get(ex.conflict_type, 0) + 1
    return counts
