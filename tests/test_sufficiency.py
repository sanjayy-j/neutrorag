"""Tests for neutrorag/sufficiency.py -- the Gate B route (i) definition of r.

Includes the structural enforcement of the hard constraint: no part of r may be
a function of the NLI outputs e, n or c. That is asserted by inspecting the
module's imports and its public signatures, not by convention.
"""
import ast
import inspect
import pathlib

import pytest

from neutrorag import sufficiency
from neutrorag.sufficiency import (
    DEFAULT_DIVERSITY_TARGET,
    DEFAULT_SUPPORT_TARGET,
    DEFAULT_WEIGHTS,
    SufficiencyWeights,
    connectivity,
    coverage,
    diversity,
    retrieval_sufficiency,
    support_count,
)

SOURCE = pathlib.Path(sufficiency.__file__).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# THE HARD CONSTRAINT: r must not depend on NLI output
# ---------------------------------------------------------------------------
def test_sufficiency_does_not_import_the_scoring_module():
    """Structural, not by convention: r cannot see the NLI head at all.

    If sufficiency.py imported neutrorag.scoring it would have access to the NLI
    backends, and the route (i) argument -- that I depends on something the NLI
    head cannot see -- would rest on discipline rather than construction.
    """
    tree = ast.parse(SOURCE)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            if node.level:  # relative import, e.g. `from .scoring import ...`
                imported.add("." * node.level + (node.module or ""))

    offenders = {m for m in imported if m and "scoring" in m}
    assert not offenders, f"sufficiency.py must not import scoring: {offenders}"


def test_no_public_function_accepts_nli_output():
    """No e/n/c may reach r through a parameter, under any spelling."""
    banned = {"e", "n", "c", "ent", "neu", "con", "entailment", "neutral",
              "contradiction", "nli", "probs", "tif", "scorer", "backend"}
    public = [(name, obj) for name, obj in vars(sufficiency).items()
              if not name.startswith("_") and inspect.isfunction(obj)
              and obj.__module__ == sufficiency.__name__]
    assert public, "expected public functions to inspect"

    for name, fn in public:
        params = set(inspect.signature(fn).parameters)
        clash = params & banned
        assert not clash, f"{name}() takes NLI-shaped parameter(s): {clash}"


def test_module_never_mentions_the_nli_backends():
    for forbidden in ("MockNLIBackend", "HFNLIBackend", "NLIBackend", "predict("):
        assert forbidden not in SOURCE, f"sufficiency.py references {forbidden}"


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------
def test_coverage_is_one_when_every_content_term_appears():
    assert coverage("Drug X reduces stroke risk.",
                    ["Drug X reduces stroke risk in patients."]) == 1.0


def test_coverage_is_zero_for_unrelated_text():
    assert coverage("Drug X reduces stroke risk.",
                    ["The stock market fell on Tuesday."]) == 0.0


def test_coverage_is_partial_when_passages_miss_part_of_the_claim():
    # "1998" and "award" are absent from the passage.
    value = coverage("The mayor won an award in 1998.",
                     ["The mayor was elected to a second term."])
    assert 0.0 < value < 1.0


def test_coverage_ignores_stopwords():
    # Without stopword removal, the shared function words alone would score > 0.
    assert coverage("The cat is on the mat.", ["It is the one of these."]) == 0.0


def test_coverage_accumulates_across_passages():
    claim = "Drug X reduces stroke risk."
    split = coverage(claim, ["Drug X was studied.", "It reduces stroke risk."])
    assert split == 1.0


def test_coverage_of_empty_claim_is_zero():
    assert coverage("", ["anything"]) == 0.0


# ---------------------------------------------------------------------------
# support_count
# ---------------------------------------------------------------------------
def test_support_count_saturates_at_target():
    assert support_count([0.9, 0.9, 0.9], target=3) == 1.0
    assert support_count([0.9] * 10, target=3) == 1.0  # more does not grow r


def test_support_count_is_linear_below_target():
    assert support_count([0.9], target=3) == pytest.approx(1 / 3)
    assert support_count([0.9, 0.9], target=3) == pytest.approx(2 / 3)


def test_support_count_ignores_passages_below_the_floor():
    assert support_count([0.9, 0.1, 0.2], floor=0.3, target=3) == pytest.approx(1 / 3)


def test_support_count_of_nothing_is_zero():
    assert support_count([], target=3) == 0.0


def test_support_count_rejects_a_nonsense_target():
    with pytest.raises(ValueError):
        support_count([0.9], target=0)


# ---------------------------------------------------------------------------
# connectivity
# ---------------------------------------------------------------------------
class _StubGraph:
    """Minimal GraphLike stub -- no networkx, no KnowledgeGraph import."""

    def __init__(self, entities, hops):
        self._entities = entities
        self._hops = hops  # {(a, b): distance}

    def entities_in(self, text):
        return [e for e in self._entities if e in text.lower()]

    def hops_between(self, a, b, max_hops=3):
        d = self._hops.get((a, b), self._hops.get((b, a)))
        return d if d is not None and d <= max_hops else None


def test_connectivity_without_a_graph_is_zero():
    assert connectivity("Drug X and Condition Y", graph=None) == 0.0


def test_connectivity_needs_at_least_two_entities():
    g = _StubGraph(["drug x"], {})
    assert connectivity("drug x alone", g) == 0.0


def test_connectivity_is_one_for_directly_connected_entities():
    g = _StubGraph(["drug x", "condition y"], {("drug x", "condition y"): 1})
    assert connectivity("drug x and condition y", g) == 1.0


def test_connectivity_decays_reciprocally_with_hop_distance():
    g2 = _StubGraph(["drug x", "condition y"], {("drug x", "condition y"): 2})
    g3 = _StubGraph(["drug x", "condition y"], {("drug x", "condition y"): 3})
    assert connectivity("drug x and condition y", g2) == pytest.approx(0.5)
    assert connectivity("drug x and condition y", g3) == pytest.approx(1 / 3)


def test_connectivity_is_zero_when_entities_are_unconnected():
    g = _StubGraph(["drug x", "condition y"], {})
    assert connectivity("drug x and condition y", g) == 0.0


def test_connectivity_uses_the_closest_connected_pair():
    g = _StubGraph(["a", "b", "c"],
                   {("a", "b"): 3, ("b", "c"): 1})
    assert connectivity("a b c", g) == 1.0


# ---------------------------------------------------------------------------
# diversity
# ---------------------------------------------------------------------------
def test_diversity_counts_distinct_sources_and_saturates():
    assert diversity(["doc1", "doc2", "doc3"], target=3) == 1.0
    assert diversity(["doc1"] * 5, target=3) == pytest.approx(1 / 3)


def test_diversity_does_not_reward_a_single_passage_from_a_single_source():
    # The rejected ratio-based definition would have scored this 1.0.
    assert diversity(["doc1"], target=3) == pytest.approx(1 / 3)


def test_diversity_ignores_missing_provenance():
    assert diversity(["doc1", "", None, "doc1"], target=3) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# aggregate + graceful degradation
# ---------------------------------------------------------------------------
def test_aggregate_uses_only_available_components_and_renormalises():
    """No graph and no provenance -> coverage and support_count only, at equal
    weight, rather than scoring the missing signals as zero."""
    out = retrieval_sufficiency(
        "Drug X reduces stroke risk.",
        passages=["Drug X reduces stroke risk in patients."],
        relevances=[0.9],
    )
    assert set(out.components) == {"coverage", "support_count"}
    assert out.weights_used == pytest.approx({"coverage": 0.5, "support_count": 0.5})
    # Against the NAMED default, not a magic number: Decision 2 moved the module
    # default from 3 to 1 (the unbiased fallback for unknown corpus depth), so a
    # single passage now saturates support_count instead of scoring 1/3.
    expected_support = min(1.0, 1 / DEFAULT_SUPPORT_TARGET)
    assert out.r == pytest.approx((1.0 + expected_support) / 2)


def test_aggregate_includes_all_four_when_inputs_are_available():
    g = _StubGraph(["drug x", "condition y"], {("drug x", "condition y"): 1})
    out = retrieval_sufficiency(
        "drug x and condition y",
        passages=["drug x affects condition y", "another passage about drug x"],
        relevances=[0.9, 0.8],
        sources=["doc1", "doc2"],
        graph=g,
    )
    assert set(out.components) == {"coverage", "support_count", "connectivity", "diversity"}
    assert all(w == pytest.approx(0.25) for w in out.weights_used.values())


def test_aggregate_is_bounded_in_the_unit_interval():
    g = _StubGraph(["drug x", "condition y"], {("drug x", "condition y"): 1})
    out = retrieval_sufficiency(
        "drug x and condition y",
        passages=["drug x and condition y"] * 10,
        relevances=[1.0] * 10,
        sources=[f"doc{i}" for i in range(10)],
        graph=g,
    )
    assert out.r == 1.0


def test_aggregate_with_no_passages_is_zero():
    assert retrieval_sufficiency("a claim", passages=[], relevances=[]).r == 0.0


def test_weights_are_configurable_for_the_sensitivity_sweep():
    weights = SufficiencyWeights(coverage=1.0, support_count=0.0,
                                 connectivity=0.0, diversity=0.0)
    out = retrieval_sufficiency(
        "Drug X reduces stroke risk.",
        passages=["Drug X reduces stroke risk."],
        relevances=[0.9],
        weights=weights,
    )
    assert out.r == pytest.approx(1.0)  # coverage alone


def test_default_weights_are_equal():
    assert DEFAULT_WEIGHTS.as_dict() == {"coverage": 0.25, "support_count": 0.25,
                                         "connectivity": 0.25, "diversity": 0.25}


def test_module_default_targets_are_the_unbiased_fallback():
    """Decision 2: an unknown-depth corpus must not be silently penalised.

    A target of 1 saturates every instance, so the component contributes a
    constant 1.0 -- non-discriminating but unbiased. The old target of 3 pinned
    FEVER's support_count at 0.333 and dragged r down on every instance, which
    is what put 143/200 instances below the F12 cliff.
    """
    assert DEFAULT_SUPPORT_TARGET == 1
    assert DEFAULT_DIVERSITY_TARGET == 1
    assert support_count([0.9]) == 1.0
    assert diversity(["one-source"]) == 1.0
