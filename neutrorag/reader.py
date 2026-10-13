"""Extractive span reader -- the single generator shared by all three Track B systems.

Track B requires the vector-RAG baseline to use "the same corpus and generator"
as the system under test, so one reader instance serves the closed-book, vector-RAG
and NeutroRAG conditions. An extractive reader is used rather than a generative
LLM because:

  * it is deterministic (argmax over spans, no sampling, no temperature), so a
    run is reproducible from a commit rather than from a seed plus a server;
  * Exact Match and token-F1 are computable directly against the gold short
    answer, with no LLM judge -- which avoids the same-model-generates-and-judges
    circularity;
  * it is CPU-feasible at the sample sizes this project can afford;
  * it reports WHICH passage the span came from, so citation accuracy is
    computable rather than estimated.

Its cost is a hard ceiling: a span reader can only return text present in the
context, so HotpotQA's yes/no answers (6.2% of the validation split, and 30.8%
of its comparison questions) are unreachable by construction. That ceiling is a
property of the reader choice and must be reported alongside any EM figure --
see ``SPAN_CEILING_NOTE``.

The default model is SQuAD-2.0-tuned, which matters for the unanswerable
condition: it has a trained null ("no answer") output, so the no-answer score is
a learned signal rather than a threshold invented here.

    from neutrorag.reader import ExtractiveReader
    reader = ExtractiveReader()
    ans = reader.read("Who directed Sinister?", passages)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple

DEFAULT_READER_MODEL = "deepset/roberta-base-squad2"

#: Maximum answer length in tokens. SQuAD convention.
MAX_ANSWER_TOKENS = 30
#: Sliding-window stride for contexts longer than the model's limit.
DOC_STRIDE = 128
#: Start/end positions considered per window. The exhaustive span search is
#: O(n^2) in Python and dominates runtime at n=512; SQuAD evaluation
#: conventionally scores the top-20 starts against the top-20 ends, which is
#: the same answer in practice at 1/150th the cost.
N_BEST = 20

SPAN_CEILING_NOTE = (
    "An extractive reader can only return a span present in the context. On "
    "HotpotQA validation that caps Exact Match at 93.8%: 458/7405 answers "
    "(6.2%) are 'yes' or 'no', which appear in no passage, and 4 more have a "
    "gold answer absent from the context. All 458 yes/no items are comparison "
    "questions, so the cap on comparison questions alone is 69.2%. Report EM "
    "against this ceiling, not against 100%."
)


@dataclass
class ReadResult:
    """One reader output.

    ``passage_index`` is the index, within the passage list handed to
    :meth:`ExtractiveReader.read`, of the passage the span was taken from --
    the citation. It is ``None`` when the reader declines to answer.
    """

    answer: str
    score: float             #: probability of the chosen span (0..1)
    null_score: float        #: probability assigned to "no answer" (0..1)
    passage_index: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    @property
    def abstained(self) -> bool:
        return self.answer == ""

    def as_row(self) -> dict:
        return {"answer": self.answer, "score": round(self.score, 6),
                "null_score": round(self.null_score, 6),
                "passage_index": self.passage_index}


@dataclass
class ExtractiveReader:
    """SQuAD-style span reader over a list of passages.

    Passages are concatenated with a separator and read as one context, with a
    sliding window when the result exceeds the model's token limit. The offset
    mapping is kept so the chosen span can be attributed back to its passage.
    """

    model: str = DEFAULT_READER_MODEL
    device: str = "cpu"
    max_answer_tokens: int = MAX_ANSWER_TOKENS
    doc_stride: int = DOC_STRIDE
    separator: str = " "

    n_best: int = N_BEST

    _tok: object = field(default=None, repr=False, compare=False)
    _mdl: object = field(default=None, repr=False, compare=False)

    #: loaded once per (model, device) and shared by every instance, so the
    #: three systems in a run hold one copy of the weights between them
    _SHARED: ClassVar[Dict[tuple, tuple]] = {}

    def _load(self):
        if self._mdl is not None:
            return self._tok, self._mdl
        key = (self.model, self.device)
        if key not in ExtractiveReader._SHARED:
            import torch
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.model)
            mdl = AutoModelForQuestionAnswering.from_pretrained(self.model)
            mdl.to(self.device)
            mdl.eval()
            torch.set_grad_enabled(False)
            ExtractiveReader._SHARED[key] = (tok, mdl)
        self._tok, self._mdl = ExtractiveReader._SHARED[key]
        return self._tok, self._mdl

    # ------------------------------------------------------------------
    def _build_context(self, passages: Sequence[str]) -> Tuple[str, List[Tuple[int, int]]]:
        """Concatenate passages, returning the text and each passage's char span."""
        parts, spans, pos = [], [], 0
        for p in passages:
            parts.append(p)
            spans.append((pos, pos + len(p)))
            pos += len(p) + len(self.separator)
        return self.separator.join(parts), spans

    def _passage_of(self, char_start: int, spans: Sequence[Tuple[int, int]]) -> Optional[int]:
        for i, (lo, hi) in enumerate(spans):
            if lo <= char_start < hi:
                return i
        return None

    def read(self, question: str, passages: Sequence[str]) -> ReadResult:
        """Read ``question`` against ``passages``.

        An empty passage list is the closed-book condition: the model sees the
        question alone and can only fall back to its null output.
        """
        import torch

        tok, mdl = self._load()
        context, spans = self._build_context([p for p in passages if p.strip()])

        if not context:
            # Closed book. There is no span to extract, so the reader abstains
            # with full null mass. This is a property of an extractive reader,
            # not a prediction about the question.
            return ReadResult(answer="", score=0.0, null_score=1.0)

        enc = tok(question, context,
                  truncation="only_second",
                  max_length=min(tok.model_max_length, 512),
                  stride=self.doc_stride,
                  return_overflowing_tokens=True,
                  return_offsets_mapping=True,
                  padding=True,
                  return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        enc.pop("overflow_to_sample_mapping", None)
        seq_ids = [enc.sequence_ids(i) for i in range(len(offsets))]

        out = mdl(**{k: v.to(self.device) for k, v in enc.items()})
        starts, ends = out.start_logits, out.end_logits

        # Null score is the CLS-to-CLS span, as SQuAD 2.0 defines it. Take the
        # most confident window's null rather than summing across windows.
        null_logit = float((starts[:, 0] + ends[:, 0]).max())

        best = (-1e30, None, None, None)  # logit, char_start, char_end, window
        for w in range(len(offsets)):
            allowed = {j for j, sid in enumerate(seq_ids[w]) if sid == 1
                       and offsets[w][j][1] > offsets[w][j][0]}
            if not allowed:
                continue
            s_log, e_log = starts[w], ends[w]
            # Mask everything that is not a context token, then take the n-best
            # starts and ends and score only those pairs.
            mask = torch.full_like(s_log, float("-inf"))
            idx = torch.tensor(sorted(allowed), dtype=torch.long)
            mask[idx] = 0.0
            s_cand = torch.topk(s_log + mask, min(self.n_best, len(allowed))).indices.tolist()
            e_cand = torch.topk(e_log + mask, min(self.n_best, len(allowed))).indices.tolist()
            for si in s_cand:
                for ei in e_cand:
                    if ei < si or (ei - si + 1) > self.max_answer_tokens:
                        continue
                    sc = float(s_log[si] + e_log[ei])
                    if sc > best[0]:
                        best = (sc, int(offsets[w][si][0]), int(offsets[w][ei][1]), w)

        if best[1] is None:
            return ReadResult(answer="", score=0.0, null_score=1.0)

        # Two-way softmax between the best span and the null option, so score
        # and null_score are a probability pair and usable as a confidence.
        import math
        m = max(best[0], null_logit)
        p_span = math.exp(best[0] - m)
        p_null = math.exp(null_logit - m)
        z = p_span + p_null
        p_span, p_null = p_span / z, p_null / z

        return ReadResult(answer=context[best[1]:best[2]].strip(),
                          score=p_span, null_score=p_null,
                          passage_index=self._passage_of(best[1], spans),
                          start_char=best[1], end_char=best[2])

    def read_or_abstain(self, question: str, passages: Sequence[str]) -> ReadResult:
        """As :meth:`read`, but returns an empty answer when null outscores the span.

        This is the reader's OWN abstention, learned from SQuAD 2.0. It is kept
        separate from :meth:`read` so the baselines can be run in the
        always-answer mode the comparison requires, while the same signal
        remains available for a stronger baseline.
        """
        r = self.read(question, passages)
        if r.null_score > r.score:
            return ReadResult(answer="", score=r.score, null_score=r.null_score)
        return r
