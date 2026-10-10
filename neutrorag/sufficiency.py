"""Retrieval sufficiency: the ``r`` signal, computed from retrieval alone.

Gate B, route (i). Before this module ``r`` was ``max(evidence relevance)``,
and every FEVER row was constructed with ``relevance=0.9``, so ``r`` was the
constant 0.9 across the whole corpus and carried no information (memo F1).

**The hard constraint, and the entire point of route (i): no quantity here may
be a function of the NLI outputs e, n or c -- not directly, not transitively.**
That is what makes ``I`` depend on something the NLI head cannot see. This
module therefore does not import from :mod:`neutrorag.scoring` at all, and no
public function accepts entailment/neutral/contradiction in any form. It takes
plain strings and floats. ``tests/test_sufficiency.py`` asserts both of these
structurally rather than trusting convention.

Four components, each a pure function into [0, 1], combined as a weighted mean:

    coverage       how much of the claim the retrieved text actually talks about
    support_count  how many passages cleared a relevance floor (saturating)
    connectivity   whether the claim's entities are related in the graph
    diversity      how many distinct sources the passages came from (saturating)

Components whose inputs are unavailable (no graph, no provenance) are dropped
and the remaining weights renormalised, so the scorer keeps working offline
with the mock backend and no retriever -- see :func:`retrieval_sufficiency`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

# ---------------------------------------------------------------------------
# Saturation targets (Decision 2)
# ---------------------------------------------------------------------------
# THE RULE: target = p75 of that corpus's own retrieval-depth distribution.
# Derived in experiments/derive_targets.py from passage and domain counts only,
# and committed before any accuracy was computed from them.
#
# Why a per-corpus target rather than one global constant: no fixed target can
# work. FEVER's passage count is a point mass (p25=median=p75=p90=1, 92.5% of
# instances at k=1), so no saturating function of k can discriminate there at
# all; CONFLICTS is concentrated at 8-10, so any target <= 5 saturates it
# completely (std exactly 0.000 at the old target of 3). The two requirements
# are mutually exclusive.
#
# The real defect in target=3 was never the lost discrimination -- the input
# carries none on FEVER -- it was the BIAS: support_count sat at the constant
# 0.333 and dragged r down on every single instance, which is what pushed
# 143/200 FEVER instances below the F12 cliff at r <= 0.5.
#
# These module defaults are therefore the UNBIASED fallback for a corpus whose
# depth distribution is unknown: a target of 1 saturates every instance, so the
# component contributes a constant 1.0 -- equally non-discriminating, but it
# subtracts nothing. Corpus-specific targets live with their loaders
# (fever.SUPPORT_TARGET, conflicts.SUPPORT_TARGET / DIVERSITY_TARGET).
DEFAULT_SUPPORT_TARGET = 1
DEFAULT_DIVERSITY_TARGET = 1

_WORD = re.compile(r"[a-z0-9]+")

# Function words carry no topical information, so counting them in `coverage`
# would make every claim look well covered by any passage of English prose.
_STOPWORDS = frozenset("""
a an the this that these those of in on at to for from by with about into over
after before between during is are was were be been being am do does did doing
has have had having will would shall should can could may might must
and or but if then than as it its it's they them their there here not no
""".split())


def _content_terms(text: str) -> List[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS]


class GraphLike(Protocol):
    """The slice of :class:`neutrorag.graph.KnowledgeGraph` this module uses.

    Declared structurally so sufficiency.py never imports the graph module (and
    so tests can substitute a stub without networkx).
    """

    def entities_in(self, text: str) -> List[str]: ...

    def hops_between(self, a: str, b: str, max_hops: int = 3) -> Optional[int]: ...


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def coverage(claim: str, passages: Sequence[str]) -> float:
    """Fraction of the claim's content terms that appear in the retrieved text.

    Rationale: the most basic sense in which retrieval is insufficient is that
    the passages simply do not talk about what the claim talks about. A claim
    mentioning an entity, a date and a quantity, whose retrieved passages
    mention only the entity, is under-covered no matter how confident the NLI
    head is about the sentence it did get.

    Deliberately lexical: an embedding-based version would be stronger, but it
    would add a model dependency to a signal that must stay cheap and, above
    all, independent of the NLI head.
    """
    terms = set(_content_terms(claim))
    if not terms:
        return 0.0
    seen = set()
    for passage in passages:
        seen.update(_content_terms(passage))
    return len(terms & seen) / len(terms)


def support_count(relevances: Sequence[float], floor: float = 0.3,
                  target: int = DEFAULT_SUPPORT_TARGET) -> float:
    """How many passages cleared the relevance floor, saturating at ``target``.

    Rationale: one on-topic passage is thin evidence; three independent ones is
    ordinarily enough to call retrieval sufficient, and the tenth adds very
    little. The signal must therefore saturate rather than grow without bound.

    Saturation choice -- ``min(1, k / target)``, linear to the target then flat.
    Chosen over the smooth alternative ``k / (k + 1)`` because (a) it actually
    reaches 1.0, so a well-retrieved claim can attain r = 1 rather than
    asymptotically approaching it, and (b) ``target`` is a single interpretable
    parameter ("how many passages count as enough") that the sensitivity sweep
    can vary, rather than a shape constant with no operational meaning. Both are
    arbitrary in the sense that neither is derived; this one is at least legible.
    """
    if target <= 0:
        raise ValueError("target must be >= 1")
    k = sum(1 for rel in relevances if rel >= floor)
    return min(1.0, k / target)


def connectivity(claim: str, graph: Optional[GraphLike] = None,
                 max_hops: int = 3) -> float:
    """Whether the claim's entities are related in the knowledge graph, and how closely.

    Rationale: retrieval can return on-topic text about each entity separately
    while the corpus says nothing about the *relation* the claim asserts. If the
    graph connects the claim's entities in one hop, the corpus contains a direct
    statement relating them; at three hops the relation is inferential at best;
    if they are unconnected, the corpus does not relate them at all.

    Normalisation choice -- ``1 / hops`` over the closest connected pair: 1 hop
    scores 1.0, 2 hops 0.5, 3 hops 0.33, unconnected 0.0. Chosen over linear
    decay ``1 - (hops-1)/max_hops`` because evidential value falls off sharply
    rather than evenly with path length: a two-hop path is already a great deal
    weaker than a direct assertion, which reciprocal decay reflects and linear
    decay understates. Fewer than two entities found -> 0.0, since there is no
    relation to check.
    """
    if graph is None:
        return 0.0
    entities = list(dict.fromkeys(graph.entities_in(claim)))
    if len(entities) < 2:
        return 0.0
    best: Optional[int] = None
    for i, a in enumerate(entities):
        for b in entities[i + 1:]:
            hops = graph.hops_between(a, b, max_hops=max_hops)
            if hops is not None and hops > 0 and (best is None or hops < best):
                best = hops
    return 0.0 if best is None else 1.0 / best


def diversity(sources: Sequence[str], target: int = DEFAULT_DIVERSITY_TARGET) -> float:
    """How many distinct sources the passages came from, saturating at ``target``.

    Rationale: five passages copied from one document are one source's word, not
    five. Corroboration that all traces to a single origin should not read as
    strong retrieval, and -- the case this project exists for -- a corpus that
    can disagree with itself must contain more than one voice.

    Normalisation choice -- ``min(1, distinct / target)``, matching
    :func:`support_count` so the two counting components behave alike. The ratio
    ``distinct / len(passages)`` was rejected: a single passage from a single
    source would score 1.0, reporting maximal diversity for the least diverse
    possible retrieval.
    """
    if target <= 0:
        raise ValueError("target must be >= 1")
    distinct = len({s for s in sources if s})
    return min(1.0, distinct / target)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SufficiencyWeights:
    """Weights for the four components.

    Equal weights are the documented default and a DESIGN PARAMETER, not a
    tuned result: no data was used to choose them. They are exposed here so the
    sensitivity sweep (memo F12) can vary them, and so any future tuned values
    have to be introduced deliberately and reported as tuned.
    """

    coverage: float = 0.25
    support_count: float = 0.25
    connectivity: float = 0.25
    diversity: float = 0.25

    def as_dict(self) -> dict:
        return {"coverage": self.coverage, "support_count": self.support_count,
                "connectivity": self.connectivity, "diversity": self.diversity}


DEFAULT_WEIGHTS = SufficiencyWeights()


@dataclass(frozen=True)
class SufficiencyBreakdown:
    """Per-component values plus the aggregate, for instrumentation and audit."""

    r: float
    components: dict
    weights_used: dict

    def as_row(self) -> dict:
        row = {"r": self.r}
        row.update({f"r_{k}": v for k, v in self.components.items()})
        return row


def retrieval_sufficiency(
    claim: str,
    passages: Sequence[str],
    relevances: Sequence[float],
    sources: Optional[Sequence[str]] = None,
    graph: Optional[GraphLike] = None,
    weights: SufficiencyWeights = DEFAULT_WEIGHTS,
    floor: float = 0.3,
    support_target: int = DEFAULT_SUPPORT_TARGET,
    diversity_target: int = DEFAULT_DIVERSITY_TARGET,
    max_hops: int = 3,
) -> SufficiencyBreakdown:
    """Combine the four components into r in [0, 1].

    Graceful degradation: ``connectivity`` needs a graph and ``diversity`` needs
    passage provenance. When either is absent the component is *dropped* and the
    remaining weights are renormalised, rather than scored 0 -- scoring an
    unavailable signal as 0 would silently report "retrieval was insufficient"
    when the truth is "this signal was not measured". This is what lets demo.py,
    the mock backend and the offline tests keep working with no retriever.

    Note that no argument is, or can be derived from, an NLI output.
    """
    if not passages:
        return SufficiencyBreakdown(r=0.0, components={}, weights_used={})

    components = {
        "coverage": coverage(claim, passages),
        "support_count": support_count(relevances, floor=floor, target=support_target),
    }
    active = {"coverage": weights.coverage, "support_count": weights.support_count}

    if graph is not None:
        components["connectivity"] = connectivity(claim, graph, max_hops=max_hops)
        active["connectivity"] = weights.connectivity
    if sources is not None:
        components["diversity"] = diversity(sources, target=diversity_target)
        active["diversity"] = weights.diversity

    total = sum(active.values())
    if total <= 0:
        return SufficiencyBreakdown(r=0.0, components=components, weights_used=active)

    r = sum(components[k] * w for k, w in active.items()) / total
    return SufficiencyBreakdown(r=max(0.0, min(1.0, r)),
                                components=components,
                                weights_used={k: w / total for k, w in active.items()})
