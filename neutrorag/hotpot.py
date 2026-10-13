"""HotpotQA (distractor) loader for the Track B evidence base.

Why this dataset. Track B requires retrieval metrics, answer metrics and a
question-type breakdown. HotpotQA distractor supplies all three directly:

  * **gold supporting facts**, so Recall@k / MRR / nDCG and context
    precision/recall are computable against a published ground truth rather
    than an LLM judge;
  * **short extractive answers**, so Exact Match and token-F1 need no judge;
  * **bridge vs comparison** types, which is the required breakdown.

Shape of the validation split, measured rather than assumed
(``experiments/describe_hotpot.py``):

  * **Gold paragraphs are always exactly 2** -- 7405/7405 instances. So
    paragraph-level recall always has a denominator of 2.
  * **99.19% of instances carry 10 paragraphs** (2 gold + 8 distractors), but
    60 instances (0.81%) carry fewer, down to 2. Twenty-one of those have *no
    distractors at all*, so removing the gold leaves an empty corpus -- see
    :data:`MIN_DISTRACTORS` and :func:`eligible_for_unanswerable`.
  * ``Recall@10`` is therefore **1.0 for every instance**: all paragraphs are
    indexed, so retrieving ten of at most ten retrieves the gold. It is
    reported as an index-completeness check, not as a retrieval result.
  * Supporting facts are published at **sentence** level (title, sent_id),
    while retrieval here operates on **paragraphs**. Both granularities are
    exposed (:attr:`HotpotExample.gold_titles` and
    :attr:`HotpotExample.gold_sentences`) so a metric states which one it used.
    Gold sentences number 2 on 67.4% of instances and up to 8 on the rest.

As in :mod:`neutrorag.fever` and :mod:`neutrorag.conflicts`, parsing is
separated from I/O: :func:`examples_from_rows` is pure and unit-tested against
an inline fixture, while :func:`load_hotpot_hf` is the only function that
touches the network.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

DATASET = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLIT = "validation"

#: Modal paragraph count: 2 gold + 8 distractors. NOT guaranteed -- 0.81% of
#: validation instances carry fewer, so never assume this when indexing.
MODAL_PARAGRAPHS = 10
#: Distractors that must survive for the unanswerable condition to be
#: constructible. Below this the retriever has nothing to rank and the
#: instance measures the empty-corpus path rather than hallucination.
MIN_DISTRACTORS = 3

QUESTION_TYPES = ("bridge", "comparison")

#: Answers that no span reader can produce, because they appear in no passage.
YES_NO = ("yes", "no")

#: Measured on the 7405-row validation split (see experiments/describe_hotpot.py).
#: Recorded here because it is the ceiling on every Exact Match figure in this
#: track, and a reader of those figures needs it in the same place.
SPAN_CEILING = {
    "overall": 0.938,       # 6947/7405 answers appear verbatim in the context
    "yes_no_count": 458,    # every one of them a comparison question
    "absent_count": 0,      # no other answer is missing, once sentences are
                            # whitespace-normalised as Paragraph.text does
    "comparison": 0.692,    # 1029/1487 -- all 458 shortfalls are yes/no
    "bridge": 1.000,        # 5918/5918
}


def _norm_title(title: str) -> str:
    """Titles are the join key between ``context`` and ``supporting_facts``.

    They match exactly in the published data; this only guards against
    whitespace drift so a silent join failure cannot zero the gold set.
    """
    return " ".join(title.split())


@dataclass(frozen=True)
class Paragraph:
    """One retrievable unit: a titled paragraph of sentences."""

    title: str
    sentences: Tuple[str, ...]

    @property
    def text(self) -> str:
        """Retrieval/reader text. The title is prepended because it is often the
        answer-bearing entity and is not otherwise present in the sentences."""
        body = " ".join(s.strip() for s in self.sentences if s.strip())
        return f"{self.title}. {body}".strip()


@dataclass
class HotpotExample:
    qid: str
    question: str
    answer: str
    qtype: str                      # bridge | comparison
    level: str
    paragraphs: List[Paragraph]
    gold_titles: Set[str] = field(default_factory=set)
    gold_sentences: Set[Tuple[str, int]] = field(default_factory=set)
    #: True once the gold paragraphs have been removed (Task 3).
    gold_removed: bool = False

    # -- properties --------------------------------------------------------
    @property
    def is_yes_no(self) -> bool:
        """Unreachable by an extractive reader -- see :data:`SPAN_CEILING`."""
        return self.answer.strip().lower() in YES_NO

    @property
    def answerable(self) -> bool:
        """False once the gold paragraphs are gone: no passage supports the answer."""
        return not self.gold_removed

    @property
    def texts(self) -> List[str]:
        return [p.text for p in self.paragraphs]

    @property
    def titles(self) -> List[str]:
        return [p.title for p in self.paragraphs]

    def answer_in_context(self) -> bool:
        """Whether the gold answer appears verbatim in the paragraphs.

        The honest ceiling for a span reader on this instance.
        """
        if self.is_yes_no:
            return False
        hay = " ".join(self.texts).lower()
        return self.answer.strip().lower() in hay

    # -- the Task 3 construction -------------------------------------------
    def without_gold(self) -> "HotpotExample":
        """Return a copy with the gold supporting paragraphs removed.

        This is the unanswerable condition: the question is well posed and the
        corpus is topically plausible (the distractors were retrieved for this
        question), but the supporting evidence is absent. A system that answers
        anyway is hallucinating, and one that abstains is correct to.

        The gold set is kept on the copy so metrics can still say which
        paragraphs *were* the gold ones; ``gold_removed`` records that they are
        no longer present.
        """
        kept = [p for p in self.paragraphs if p.title not in self.gold_titles]
        return HotpotExample(
            qid=self.qid, question=self.question, answer=self.answer,
            qtype=self.qtype, level=self.level, paragraphs=kept,
            gold_titles=set(self.gold_titles),
            gold_sentences=set(self.gold_sentences),
            gold_removed=True,
        )


# ---------------------------------------------------------------------------
# Parsing -- pure, no I/O
# ---------------------------------------------------------------------------
def examples_from_rows(rows: Iterable[dict],
                       max_examples: Optional[int] = None) -> List[HotpotExample]:
    """Parse raw HotpotQA rows into :class:`HotpotExample`.

    Expects the HuggingFace ``hotpotqa/hotpot_qa`` column layout, where
    ``context`` is ``{"title": [...], "sentences": [[...], ...]}`` and
    ``supporting_facts`` is ``{"title": [...], "sent_id": [...]}``.

    Rows whose supporting-fact titles do not appear in their own context are
    **skipped rather than silently kept**: a row with an empty gold set would
    score 0 recall for reasons that have nothing to do with the retriever.
    """
    out: List[HotpotExample] = []
    for row in rows:
        if max_examples is not None and len(out) >= max_examples:
            break
        ctx = row.get("context") or {}
        titles = [_norm_title(t) for t in ctx.get("title", [])]
        sents = ctx.get("sentences", [])
        if not titles or len(titles) != len(sents):
            continue

        paragraphs = [Paragraph(title=t, sentences=tuple(s))
                      for t, s in zip(titles, sents)]

        sf = row.get("supporting_facts") or {}
        sf_titles = [_norm_title(t) for t in sf.get("title", [])]
        sf_ids = list(sf.get("sent_id", []))
        available = set(titles)
        gold_titles = {t for t in sf_titles if t in available}
        if not gold_titles:
            continue
        gold_sentences = {(t, int(i)) for t, i in zip(sf_titles, sf_ids)
                          if t in available}

        out.append(HotpotExample(
            qid=str(row.get("id", "")),
            question=(row.get("question") or "").strip(),
            answer=(row.get("answer") or "").strip(),
            qtype=(row.get("type") or "").strip(),
            level=(row.get("level") or "").strip(),
            paragraphs=paragraphs,
            gold_titles=gold_titles,
            gold_sentences=gold_sentences,
        ))
    return out


def stratified_sample(examples: Sequence[HotpotExample], n: int, seed: int,
                      require_unanswerable_eligible: bool = True
                      ) -> List[HotpotExample]:
    """Sample ``n`` examples preserving the bridge/comparison proportions.

    Stratified rather than uniform because comparison is only 20.1% of the
    split, and the type breakdown (Task 4) needs enough comparison instances to
    carry a confidence interval. Preserving the natural proportion keeps the
    pooled figures comparable to the full split.
    """
    rng = random.Random(seed)
    if require_unanswerable_eligible:
        examples = [e for e in examples if eligible_for_unanswerable(e)]
    by_type: Dict[str, List[HotpotExample]] = {}
    for ex in examples:
        by_type.setdefault(ex.qtype, []).append(ex)

    total = len(examples)
    picked: List[HotpotExample] = []
    for qtype in sorted(by_type):
        pool = sorted(by_type[qtype], key=lambda e: e.qid)
        share = round(n * len(pool) / total)
        rng.shuffle(pool)
        picked.extend(pool[:share])

    # rounding can leave the sample one short or long
    picked.sort(key=lambda e: e.qid)
    rng.shuffle(picked)
    return picked[:n]


def split_answerable(examples: Sequence[HotpotExample], seed: int,
                     fraction: float = 0.5
                     ) -> Tuple[List[HotpotExample], List[HotpotExample]]:
    """Partition into (answerable, unanswerable) for Task 3.

    A random ``fraction`` of the examples have their gold paragraphs removed.
    The split is seeded and stratified by question type, so the answerable and
    unanswerable halves have the same bridge/comparison mix and a difference
    between them cannot be an artefact of type imbalance.
    """
    rng = random.Random(seed)
    ineligible = [e for e in examples if not eligible_for_unanswerable(e)]
    if ineligible:
        raise ValueError(
            f"{len(ineligible)} example(s) cannot form the unanswerable "
            f"condition: removing the gold would leave fewer than "
            f"{MIN_DISTRACTORS} passages. Filter with "
            f"eligible_for_unanswerable() before splitting -- first offender "
            f"qid={ineligible[0].qid!r}."
        )
    by_type: Dict[str, List[HotpotExample]] = {}
    for ex in examples:
        by_type.setdefault(ex.qtype, []).append(ex)

    answerable: List[HotpotExample] = []
    unanswerable: List[HotpotExample] = []
    for qtype in sorted(by_type):
        pool = sorted(by_type[qtype], key=lambda e: e.qid)
        rng.shuffle(pool)
        cut = int(round(len(pool) * fraction))
        unanswerable.extend(p.without_gold() for p in pool[:cut])
        answerable.extend(pool[cut:])
    answerable.sort(key=lambda e: e.qid)
    unanswerable.sort(key=lambda e: e.qid)
    return answerable, unanswerable


def eligible_for_unanswerable(ex: HotpotExample) -> bool:
    """Whether the unanswerable condition can be built from this instance.

    Removing the 2 gold paragraphs must leave at least :data:`MIN_DISTRACTORS`
    passages. Twenty-one validation instances have no distractors at all, and
    another 22 are left with 1-3; on those, "the system answered with no
    supporting evidence" is confounded with "the system was handed an empty or
    near-empty corpus", which is a different measurement.

    Sampling is restricted to eligible instances so the answerable and
    unanswerable arms are drawn from one population.
    """
    return len(ex.paragraphs) - len(ex.gold_titles & set(ex.titles)) >= MIN_DISTRACTORS


def type_counts(examples: Sequence[HotpotExample]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for ex in examples:
        out[ex.qtype] = out.get(ex.qtype, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Answer scoring -- the official HotpotQA normalisation
# ---------------------------------------------------------------------------
_PUNC = re.compile(r"[^\w\s]")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace.

    This is the SQuAD/HotpotQA official normalisation. It is reproduced here
    rather than imported so EM and F1 are defined inside the repository and a
    reviewer can read exactly what "exact" means.
    """
    s = s.lower()
    s = _PUNC.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 on the normalised strings, SQuAD-style."""
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common: Dict[str, int] = {}
    for t in p:
        if t in g:
            common[t] = min(p.count(t), g.count(t))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_hotpot_hf(split: str = SPLIT, max_examples: Optional[int] = None,
                   config: str = CONFIG) -> List[HotpotExample]:
    """Load HotpotQA from the HuggingFace hub. The only networked function here."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, config, split=split)
    return examples_from_rows(ds, max_examples=max_examples)
