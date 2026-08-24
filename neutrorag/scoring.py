"""Compute a neutrosophic (T, I, F) score for a claim given retrieved evidence.

Construction (see README for the derivation):

    r      = retrieval sufficiency in [0, 1]  (did we retrieve anything on-topic)
    e,n,c  = relevance-weighted mean NLI entailment / neutral / contradiction
             over the retrieved passages (each in [0, 1], e+n+c approx 1)
    disagree = 2 * min(e, c)                 (cross-source conflict signal)

    T = r * e                       # support requires on-topic evidence that entails
    F = r * c + lambda * disagree   # contradiction from conflict + source disagreement
    I = 1 - r + r * n               # structural (no evidence) + semantic (neutral) doubt

Because ``disagree`` enters F independently, T + I + F can exceed 1 whenever some
sources support and others contradict -- exactly the case a summed-to-one
(fuzzy/probabilistic) score cannot express.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Protocol, Sequence, Tuple

from .core import TIF, clamp

_WORD = re.compile(r"[a-z0-9]+")
_NEGATION = {
    "not", "no", "never", "none", "cannot", "cant", "isnt", "wasnt", "arent",
    "false", "incorrect", "denies", "denied", "refutes", "refuted", "contradicts",
    "contradicted", "disproven", "untrue", "wrong", "fails",
}


def _tokens(text: str) -> List[str]:
    return _WORD.findall(text.lower())


# ---------------------------------------------------------------------------
# NLI backends
# ---------------------------------------------------------------------------
class NLIBackend(Protocol):
    """Return (entailment, neutral, contradiction) probabilities summing to 1."""

    def predict(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        ...


class MockNLIBackend:
    """Deterministic, dependency-free NLI stand-in for offline development.

    Heuristic: lexical overlap drives entailment; negation cues flip overlap to
    contradiction; low overlap yields neutral. This is a SCAFFOLD ONLY -- swap in
    :class:`HFNLIBackend` for paper-grade numbers.
    """

    def predict(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        p, h = set(_tokens(premise)), set(_tokens(hypothesis))
        if not h:
            return (0.0, 1.0, 0.0)
        overlap = len(p & h) / len(h)
        negated = bool((_NEGATION & p) ^ (_NEGATION & h))  # negation on exactly one side
        if overlap < 0.25:
            return (0.05, 0.90, 0.05)  # off-topic -> neutral
        strength = clamp(0.2 + overlap)
        if negated:
            return (clamp(0.1), clamp(1 - strength - 0.1), clamp(strength))
        return (clamp(strength), clamp(1 - strength - 0.1), clamp(0.1))


class HFNLIBackend:
    """Real NLI via HuggingFace transformers (optional, downloaded on first use).

    Example model: ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli``.
    The ``transformers`` import and the model/pipeline load are both deferred
    until the first :meth:`predict` call (not ``__init__``) so constructing a
    backend -- e.g. via a CLI default argument -- never pays the load cost or
    requires the dependency unless it is actually used. Loaded pipelines are
    cached per (model, device) at the class level, so multiple backends (or
    repeated instantiation) that share a model don't reload the weights.
    """

    _pipeline_cache: dict = {}

    def __init__(
        self,
        model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        device: str = "cpu",
    ):
        self._model = model
        self._device = device
        self._pipe = None  # lazy -- populated by _load() on first predict()

    def _load(self):
        if self._pipe is None:
            key = (self._model, self._device)
            if key not in HFNLIBackend._pipeline_cache:
                from transformers import pipeline  # noqa: local import by design

                HFNLIBackend._pipeline_cache[key] = pipeline(
                    "text-classification",
                    model=self._model,
                    top_k=None,
                    device=self._device,
                )
            self._pipe = HFNLIBackend._pipeline_cache[key]
        return self._pipe

    def predict(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        pipe = self._load()
        out = pipe({"text": premise, "text_pair": hypothesis})
        scores = {d["label"].lower(): d["score"] for d in out}
        e = scores.get("entailment", 0.0)
        n = scores.get("neutral", 0.0)
        c = scores.get("contradiction", 0.0)
        s = e + n + c or 1.0
        return (e / s, n / s, c / s)


# ---------------------------------------------------------------------------
# Evidence + scorer
# ---------------------------------------------------------------------------
@dataclass
class Evidence:
    """A retrieved passage with its retrieval relevance in [0, 1]."""

    text: str
    relevance: float = 1.0


@dataclass
class NLIRetrievalScorer:
    """Combine retrieval statistics and NLI into a neutrosophic (T, I, F) score."""

    nli: NLIBackend = field(default_factory=MockNLIBackend)
    lam: float = 0.5  # weight of cross-source disagreement in F

    def score(self, claim: str, evidence: Sequence[Evidence]) -> TIF:
        evidence = [e for e in evidence if e.text.strip()]
        if not evidence:
            # No evidence retrieved at all -> pure structural indeterminacy.
            return TIF(T=0.0, I=1.0, F=0.0)

        weights, ent, neu, con = [], [], [], []
        for ev in evidence:
            e, n, c = self.nli.predict(ev.text, claim)
            weights.append(clamp(ev.relevance))
            ent.append(e)
            neu.append(n)
            con.append(c)

        wsum = sum(weights) or 1.0
        e = sum(w * x for w, x in zip(weights, ent)) / wsum
        n = sum(w * x for w, x in zip(weights, neu)) / wsum
        c = sum(w * x for w, x in zip(weights, con)) / wsum

        r = clamp(max(weights))            # retrieval sufficiency
        disagree = clamp(2.0 * min(e, c))  # high only if BOTH support & conflict present

        T = clamp(r * e)
        F = clamp(r * c + self.lam * disagree)
        I = clamp(1.0 - r + r * n)
        return TIF(T=T, I=I, F=F)
