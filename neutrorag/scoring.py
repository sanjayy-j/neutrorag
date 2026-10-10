"""Compute a neutrosophic (T, I, F) score for a claim given retrieved evidence.

Construction (see README for the derivation):

    r      = retrieval sufficiency in [0, 1]  (did we retrieve anything on-topic)
    e,n,c  = relevance-weighted mean NLI entailment / neutral / contradiction
             over the retrieved passages (each in [0, 1], e+n+c approx 1)
    disagree = 2 * max over pairs (i, j) of min(e_i, c_j)   (conflict signal)

    T = r * e                       # support requires on-topic evidence that entails
    F = r * c + lambda * disagree   # contradiction from conflict + source disagreement
    I = max(1 - r, n)               # the WORSE of thin retrieval and neutral evidence

**Why ``disagree`` is pairwise (memo F6).** It was ``2 * min(e, c)`` on the
*pooled* means, so adding neutral passages diluted it monotonically even though
the conflict between the two opposed passages had not changed at all: the memo's
own filler test decayed F = 1.000 -> 0.752 -> 0.631 -> 0.545 -> 0.499 as neutral
passages were added one at a time. Conflict is a property of the *set*, and a
mean cannot recover it from its marginals. It is now the strongest opposition
between any entailing passage i and any contradicting passage j.

Two consequences of keeping the existing factor of 2 rather than rescaling, both
deliberate and both worth revisiting separately from this change:

* With one passage the only pair is (i, i) and the expression reduces EXACTLY to
  the old ``2 * min(e, c)``. That is why single-passage corpora such as FEVER are
  unaffected -- which is the point: the reformulation should change behaviour only
  where there are actually pairs to compare.
* For a genuinely opposed pair, ``min(e_i, c_j)`` can reach 1.0 unconstrained by
  the simplex, so the doubled value saturates at the clamp once the best pair
  reaches 0.5. Conflict strength above that point is no longer resolved, and a
  single maximally ambiguous passage (e = c = 0.5) still scores as high as two
  flatly opposed sources. Rescaling to fix either would be a change of scale, not
  the reformulation the memo asked for, so it is left for a separate decision.

Indeterminacy has two independent causes: retrieval was thin (``1 - r``), or what
was retrieved is on-topic but uninformative (``n``). ``I`` takes the worse of the
two, so a claim is indeterminate if *either* holds.

Why not ``I = (1 - r) + r * n`` (the form used until Gate B)? Because it blends the
two multiplicatively, and that blending is exactly what made the mass identity
structural: ``r`` multiplied the whole NLI simplex across the three channels and
then cancelled against ``I``'s ``(1 - r)`` term, giving

    T + I + F = r*(e + n + c) + (1 - r) + lambda*disagree = 1 + lambda*disagree

for ANY r, however the retrieval signal was derived (memo F5). Under ``max`` the
cancellation does not occur, so the mass carries information beyond ``disagree``.

Note that ``I = 1 - r * (1 - n)`` is NOT an alternative: it expands to
``1 - r + r * n``, i.e. the old form.

``max`` was chosen over the additive ``(1 - r) + n``, which also breaks the
identity and behaves near-identically on realistic inputs, because ``max`` stays
in [0, 1] without clamping. The additive form reaches 1.8 before clamping when
retrieval is thin AND evidence is neutral, so the clamp would bind precisely
where I should be most informative.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, Tuple

from .core import TIF, clamp
from .sufficiency import (
    DEFAULT_DIVERSITY_TARGET, DEFAULT_SUPPORT_TARGET, DEFAULT_WEIGHTS, GraphLike,
    SufficiencyBreakdown, SufficiencyWeights, retrieval_sufficiency,
)

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
            e, n, c = clamp(0.1), clamp(1 - strength - 0.1), clamp(strength)
        else:
            e, n, c = clamp(strength), clamp(1 - strength - 0.1), clamp(0.1)
        # Renormalise so the triple is a distribution. Clamping truncates the
        # neutral term at 0 for high overlap, which otherwise returns e.g.
        # (0.10, 0.00, 1.00) summing to 1.1. HFNLIBackend.predict already
        # normalises the same way.
        s = e + n + c or 1.0
        return (e / s, n / s, c / s)


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
    """A retrieved passage with its retrieval relevance in [0, 1].

    ``source`` is the document/origin the passage came from. It is optional and
    purely retrieval-side metadata, used by the diversity component of
    :mod:`neutrorag.sufficiency`; when it is None that component is dropped
    rather than scored zero.
    """

    text: str
    relevance: float = 1.0
    source: Optional[str] = None


@dataclass
class NLIRetrievalScorer:
    """Combine retrieval statistics and NLI into a neutrosophic (T, I, F) score."""

    nli: NLIBackend = field(default_factory=MockNLIBackend)
    lam: float = 0.5  # weight of cross-source disagreement in F

    # -- retrieval sufficiency (Gate B, route (i)) --------------------------
    # r is computed by neutrorag.sufficiency from retrieval-side quantities
    # only. Supplying a graph enables the connectivity component; supplying
    # Evidence.source enables diversity. Both degrade gracefully when absent.
    graph: Optional[GraphLike] = None
    sufficiency_weights: SufficiencyWeights = DEFAULT_WEIGHTS
    # Saturation targets for the two count-based components. Per-corpus by
    # design (Decision 2): target = p75 of that corpus's depth distribution.
    # Supply fever.SUPPORT_TARGET or conflicts.SUPPORT_TARGET/DIVERSITY_TARGET;
    # the defaults are the unbiased fallback for unknown depth.
    support_target: int = DEFAULT_SUPPORT_TARGET
    diversity_target: int = DEFAULT_DIVERSITY_TARGET
    # Set to a float to bypass the sufficiency computation and pin r to that
    # value. Exists for the r-ablation (r = 0.9 reproduces the pre-Gate-B
    # behaviour exactly); not for production use.
    r_override: Optional[float] = None

    def sufficiency(self, claim: str, evidence: Sequence[Evidence]) -> SufficiencyBreakdown:
        """Compute r and its per-component breakdown for this claim/evidence set.

        Pure and NLI-free: this makes no model call, so instrumentation can
        report r alongside e/n/c without perturbing anything.
        """
        usable = [e for e in evidence if e.text.strip()]
        if self.r_override is not None:
            return SufficiencyBreakdown(r=clamp(self.r_override),
                                        components={}, weights_used={})
        sources = [e.source for e in usable]
        return retrieval_sufficiency(
            claim,
            passages=[e.text for e in usable],
            relevances=[clamp(e.relevance) for e in usable],
            sources=sources if any(s for s in sources) else None,
            graph=self.graph,
            weights=self.sufficiency_weights,
            support_target=self.support_target,
            diversity_target=self.diversity_target,
        )

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

        r = clamp(self.sufficiency(claim, evidence).r)  # retrieval sufficiency
        # Pairwise conflict (memo F6): the strongest opposition between ANY
        # passage that entails and ANY passage that contradicts. Computed on
        # the per-passage outputs, not the pooled means, because conflict is a
        # set-level property -- see the module docstring.
        disagree = clamp(2.0 * max(min(e_i, c_j) for e_i in ent for c_j in con))

        T = clamp(r * e)
        F = clamp(r * c + self.lam * disagree)
        I = clamp(max(1.0 - r, n))
        return TIF(T=T, I=I, F=F)
