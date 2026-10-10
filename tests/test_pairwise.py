"""Tests for pairwise scoring mode (Decision 1).

Includes the Task 3 sanity cases, run here against the mock backend so they stay
in the offline suite; experiments/check_pairwise_sanity.py runs the same three
cases against the real model as a pre-experiment gate.
"""
import pytest

from neutrorag.scoring import (
    DEFAULT_MAX_PASSAGES, Evidence, MockNLIBackend, NLIRetrievalScorer,
)


def _scorer():
    return NLIRetrievalScorer(nli=MockNLIBackend())


# ---------------------------------------------------------------------------
# The sanity gate (Task 3)
# ---------------------------------------------------------------------------
SANITY = [
    ("contradicting", "Did Drug X reduce mortality in the trial?",
     "The trial found that Drug X reduced mortality by 30 percent in treated patients.",
     "The trial found that Drug X did not reduce mortality in treated patients.",
     "F"),
    ("agreeing", "What is the capital of France?",
     "Paris is the capital of France.",
     "The capital city of France is Paris.",
     "T"),
    ("unrelated", "What happened in the market on Tuesday?",
     "The stock market fell sharply on Tuesday amid inflation concerns.",
     "Penguins are flightless birds that live in the southern hemisphere.",
     "I"),
]


@pytest.mark.parametrize("name,query,a,b,expected", SANITY)
def test_sanity_cases_land_on_the_expected_channel(name, query, a, b, expected):
    bd = _scorer().pairwise_breakdown(
        query, [Evidence(a, 0.9, source="a"), Evidence(b, 0.9, source="b")])
    assert bd.tif.dominant == expected, f"{name}: got {bd.tif}"


def test_contradiction_drives_F_not_corroboration():
    _, query, a, b, _ = SANITY[0]
    bd = _scorer().pairwise_breakdown(
        query, [Evidence(a, 0.9, source="a"), Evidence(b, 0.9, source="b")])
    assert bd.contradiction > bd.corroboration


def test_agreement_drives_corroboration_not_contradiction():
    _, query, a, b, _ = SANITY[1]
    bd = _scorer().pairwise_breakdown(
        query, [Evidence(a, 0.9, source="a"), Evidence(b, 0.9, source="b")])
    assert bd.corroboration > bd.contradiction


# ---------------------------------------------------------------------------
# The query must never reach the NLI model
# ---------------------------------------------------------------------------
class _RecordingNLI:
    def __init__(self):
        self.calls = []

    def predict(self, premise, hypothesis):
        self.calls.append((premise, hypothesis))
        return (0.3, 0.4, 0.3)


def test_query_is_never_passed_to_the_nli_model():
    """The whole point of this mode: a question is not a valid NLI hypothesis.

    Claim mode passes the claim as the hypothesis, which on a QA corpus means
    asking whether a passage entails a question. Pairwise mode must compare
    passages only, using the query solely for retrieval sufficiency.
    """
    rec = _RecordingNLI()
    scorer = NLIRetrievalScorer(nli=rec)
    query = "what is the most common blood type"
    passages = [Evidence(f"passage number {i} about blood types", 0.9, source=f"s{i}")
                for i in range(3)]

    scorer.pairwise_breakdown(query, passages)

    assert rec.calls, "expected NLI calls"
    texts = {p for call in rec.calls for p in call}
    assert query not in texts
    for premise, hypothesis in rec.calls:
        assert premise != hypothesis


def test_all_ordered_pairs_are_compared():
    rec = _RecordingNLI()
    NLIRetrievalScorer(nli=rec).pairwise_breakdown(
        "q", [Evidence(f"p{i}", 0.9, source=f"s{i}") for i in range(4)])
    assert len(rec.calls) == 4 * 3  # k(k-1) ordered pairs, self-pairs excluded


# ---------------------------------------------------------------------------
# Corroboration requires MUTUAL entailment
# ---------------------------------------------------------------------------
class _DirectionalNLI:
    """Entails strongly one way only -- e.g. a specific passage entailing a
    general one. That is specificity, not agreement between sources."""

    def predict(self, premise, hypothesis):
        if premise == "specific" and hypothesis == "general":
            return (0.95, 0.04, 0.01)
        return (0.05, 0.94, 0.01)


def test_one_directional_entailment_is_not_corroboration():
    scorer = NLIRetrievalScorer(nli=_DirectionalNLI())
    bd = scorer.pairwise_breakdown(
        "q", [Evidence("specific", 0.9, source="a"), Evidence("general", 0.9, source="b")])
    # min(e_ij, e_ji) = min(0.95, 0.05) = 0.05, not 0.95
    assert bd.corroboration == pytest.approx(0.05)
    assert bd.tif.dominant == "I"


# ---------------------------------------------------------------------------
# Degenerate inputs and the passage cap
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [0, 1])
def test_fewer_than_two_passages_is_pure_indeterminacy(n):
    ev = [Evidence("only one passage", 0.9, source="a")][:n]
    bd = _scorer().pairwise_breakdown("a query", ev)
    assert bd.n_pairs == 0
    assert bd.tif.I == 1.0
    assert bd.tif.T == 0.0 and bd.tif.F == 0.0


def test_passage_cap_bounds_the_quadratic_cost():
    rec = _RecordingNLI()
    ev = [Evidence(f"passage {i}", 0.5 + i / 100, source=f"s{i}") for i in range(12)]
    bd = NLIRetrievalScorer(nli=rec).pairwise_breakdown("q", ev, max_passages=5)
    assert bd.n_passages == 5
    assert bd.n_pairs == 5 * 4
    assert len(rec.calls) == 20


def test_cap_keeps_the_most_relevant_passages():
    rec = _RecordingNLI()
    ev = [Evidence(f"passage {i}", i / 10, source=f"s{i}") for i in range(6)]
    NLIRetrievalScorer(nli=rec).pairwise_breakdown("q", ev, max_passages=2)
    used = {p for call in rec.calls for p in call}
    assert used == {"passage 5", "passage 4"}


def test_default_cap_is_documented_constant():
    assert DEFAULT_MAX_PASSAGES == 5


# ---------------------------------------------------------------------------
# Claim mode is untouched
# ---------------------------------------------------------------------------
def test_claim_mode_still_works_and_is_independent():
    scorer = _scorer()
    claim = "Sharks are mammals."
    ev = [Evidence("Sharks are not mammals; they are fish.", 0.9)]
    claim_tif = scorer.score(claim, ev)
    assert claim_tif.dominant == "F"
    # the two modes are different measurements and need not agree
    pair_bd = scorer.pairwise_breakdown(claim, ev)
    assert pair_bd.n_pairs == 0  # single passage: nothing to compare
