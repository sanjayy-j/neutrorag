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
    DEFAULT_DIVERSITY_TARGET,
    DEFAULT_SUPPORT_TARGET,
    DEFAULT_WEIGHTS,
    GraphLike,
    RELEVANCE_FLOOR_V2,
    SufficiencyBreakdown,
    SufficiencyWeights,
    retrieval_sufficiency,
    retrieval_sufficiency_v2,
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


# Default cap on passages compared in pairwise mode. Cost is k(k-1) NLI calls,
# and at 4.48s/call measured on CPU an uncapped k=9 instance costs 323s. Five is
# also a realistic RAG top-k, and it bounds the upward drift of the
# max-over-pairs statistics with k.
DEFAULT_MAX_PASSAGES = 5


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


@dataclass(frozen=True)
class PairwiseBreakdown:
    """Pairwise-mode channels plus their inputs, for instrumentation.

    ``corroboration`` is named for what it is. It is NOT support for a claim.
    """

    tif: TIF
    r: float
    corroboration: float
    contradiction: float
    neutrality: float
    n_passages: int
    n_pairs: int

    def as_row(self) -> dict:
        return {"r": self.r, "corroboration": self.corroboration,
                "contradiction": self.contradiction, "neutrality": self.neutrality,
                "T_corroboration": self.tif.T, "I": self.tif.I, "F": self.tif.F,
                "n_passages": self.n_passages, "n_pairs": self.n_pairs}


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
    # Select the Addendum 5 repair, r_v2 = (coverage + support_frac) / 2.
    # DEFAULT FALSE: every figure already in the report must stay reproducible
    # from this same code, so the legacy path is untouched and still the
    # default. See neutrorag.sufficiency.retrieval_sufficiency_v2.
    use_r_v2: bool = False
    relevance_floor_v2: float = RELEVANCE_FLOOR_V2

    def sufficiency(self, claim: str, evidence: Sequence[Evidence]) -> SufficiencyBreakdown:
        """Compute r and its per-component breakdown for this claim/evidence set.

        Pure and NLI-free: this makes no model call, so instrumentation can
        report r alongside e/n/c without perturbing anything.
        """
        usable = [e for e in evidence if e.text.strip()]
        if self.r_override is not None:
            return SufficiencyBreakdown(r=clamp(self.r_override),
                                        components={}, weights_used={})
        if self.use_r_v2:
            # r_v2 takes no graph, no sources and no saturation target.
            return retrieval_sufficiency_v2(
                claim,
                passages=[e.text for e in usable],
                relevances=[clamp(e.relevance) for e in usable],
                floor=self.relevance_floor_v2,
            )
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

    # -- pairwise mode (Decision 1) ----------------------------------------
    def pairwise_breakdown(self, query: str, evidence: Sequence[Evidence],
                           max_passages: int = DEFAULT_MAX_PASSAGES) -> "PairwiseBreakdown":
        """Score the evidence set by comparing passages AGAINST EACH OTHER.

        Claim mode asks "does passage i entail the claim?" and infers conflict
        from the combination of per-passage answers. That is conflict measured
        by proxy, through marginals -- the thing memo F6 warns about -- and it
        is unusable where no claim exists, which is ~48% of CONFLICTS. Pairwise
        mode drops the detour: the thesis is about INTER-SOURCE conflict, which
        is a relation between passages, so it is measured as one.

        ``query`` is used ONLY for retrieval sufficiency (coverage, graph
        connectivity). It is never passed to the NLI model as a hypothesis --
        that is precisely the ill-posed call this mode exists to avoid.

        CHANNEL SEMANTICS UNDER THIS MODE:
          T = CORROBORATION      mutual agreement between sources.
                                 NOT "support for the claim" -- a different
                                 quantity, which this mode does not measure and
                                 must not be reported as.
          F = inter-source contradiction
          I = retrieval insufficiency

        AGGREGATION, and what was rejected:
          corroboration = max over unordered pairs of min(e_ij, e_ji)
            Both directions must entail, so it registers genuine mutual
            agreement. REJECTED max(e_ij) one-directional: entailment is
            directional, and a specific passage entails a general one by being
            more specific, which is not agreement between sources. REJECTED the
            mean over pairs: it dilutes exactly as F6 describes. REJECTED
            "fraction of pairs above a threshold": a new free parameter with no
            principled value, and the sweep would have to invent one.
          contradiction = max over ordered pairs of c_ij
            Mirrors the Step 6 form: the strongest opposition present anywhere in
            the set. Unordered would also work since contradiction is close to
            symmetric, but the ordered pass is already paid for by
            corroboration's need for both directions.
          neutrality = mean over ordered pairs of n_ij
            CONFOUND, stated: pairwise neutrality means the sources neither agree
            nor conflict, which also describes genuinely COMPLEMENTARY sources
            covering different aspects. On CONFLICTS that is an annotated class
            of its own, so I cannot separate "retrieval was thin" from "sources
            are complementary" here. Reported, not papered over.
          No lambda*disagree term. In claim mode that term exists to recover a
            set-level property from marginals; here the set-level property is
            measured directly, so adding it would double-count the same signal.

        KNOWN WEAKNESS: both T and F are max-over-pairs, i.e. extreme-value
        statistics over k(k-1) pairs, so they drift upward with k -- more
        passages, more chances at a spurious extreme. ``max_passages`` caps k,
        which bounds that drift as well as the quadratic cost.
        """
        usable = [e for e in evidence if e.text.strip()]
        # r is computed on the FULL retrieved set, not the capped one. The cap is
        # a compute budget on the O(k^2) pairwise comparison; it is not a claim
        # that retrieval returned fewer passages. Computing r on the capped set
        # would make every CONFLICTS instance show exactly max_passages, which
        # re-saturates support_count at a constant and reintroduces precisely the
        # bias Decision 2 removed.
        r = clamp(self.sufficiency(query, usable).r)
        kept = sorted(usable, key=lambda e: -clamp(e.relevance))[:max_passages]

        if len(kept) < 2:
            # Nothing to compare against: corroboration and contradiction are
            # undefined with fewer than two sources, so the set is purely
            # indeterminate on the retrieval axis.
            return PairwiseBreakdown(tif=TIF(T=0.0, I=clamp(max(1.0 - r, 1.0)), F=0.0),
                                     r=r, corroboration=0.0, contradiction=0.0,
                                     neutrality=1.0, n_passages=len(kept), n_pairs=0)

        ent, neu, con = {}, [], []
        for i, a in enumerate(kept):
            for j, b in enumerate(kept):
                if i == j:
                    continue
                e_ij, n_ij, c_ij = self.nli.predict(a.text, b.text)
                ent[(i, j)] = e_ij
                neu.append(n_ij)
                con.append(c_ij)

        corroboration = max(min(ent[(i, j)], ent[(j, i)])
                            for i in range(len(kept)) for j in range(i + 1, len(kept)))
        contradiction = max(con)
        neutrality = sum(neu) / len(neu)

        tif = TIF(T=clamp(r * corroboration),
                  I=clamp(max(1.0 - r, neutrality)),
                  F=clamp(r * contradiction))
        return PairwiseBreakdown(tif=tif, r=r, corroboration=corroboration,
                                 contradiction=contradiction, neutrality=neutrality,
                                 n_passages=len(kept), n_pairs=len(con))

    def score_pairwise(self, query: str, evidence: Sequence[Evidence],
                       max_passages: int = DEFAULT_MAX_PASSAGES) -> TIF:
        """Pairwise-mode TIF. See :meth:`pairwise_breakdown` for the semantics --
        in particular that T is CORROBORATION here, not support for a claim."""
        return self.pairwise_breakdown(query, evidence, max_passages).tif

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
