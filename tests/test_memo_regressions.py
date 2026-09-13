"""Regression guards for the defects found in the supervisory review memo.

Every finding below sat green in CI for months because the existing tests
either used hand-written score_fn stubs (never the real scorer) or asserted
only on the dominant channel (never on exact values). These tests are written
to fail if the defect is present, so the defect is visible rather than hidden.

Covers memo findings F3, F7 (Gate C) and F11.
"""
import os

import pytest

from neutrorag.core import TIF
from neutrorag.fever import _examples_from_rows
from neutrorag.policy import Action, NeutroPolicy, PolicyConfig
from neutrorag.scoring import Evidence, MockNLIBackend, NLIRetrievalScorer


def _triple(tif: TIF):
    return (tif.T, tif.I, tif.F)


# ---------------------------------------------------------------------------
# F3 -- the empty-evidence shortcut
# ---------------------------------------------------------------------------
# neutrorag/scoring.py:145-148 filters blank passages and then short-circuits
# to TIF(0, 1, 0) without calling the NLI model at all. On a FEVER row whose
# gold label is NOT ENOUGH INFO (channel I) that is a guaranteed-correct
# answer obtained without looking at any evidence.
#
# N_empty was 0 in the 200-row run committed as fever_run_200.csv, so this
# path did not contaminate that number -- but it is live code.

_UNUSABLE_FEVER_ROW = {
    "claim": "Some claim whose evidence was all dropped as malformed.",
    "label": "NOT ENOUGH INFO",
    "evidence": [["Page", "0", ""], ["OnlyPage"], None, ["Page", "1", "   "]],
}


def test_fever_row_with_unusable_evidence_yields_a_free_correct_I():
    """Characterisation test: documents the F3 leak as it currently behaves.

    This test PASSES today. It exists so the behaviour is pinned and any
    future change to it is deliberate rather than accidental.
    """
    (example,) = _examples_from_rows([_UNUSABLE_FEVER_ROW])
    assert example.gold_channel == "I"

    scorer = NLIRetrievalScorer(nli=MockNLIBackend())
    tif = scorer.score(example.claim, example.evidence)

    # No usable passage survived, yet the row scores a confident, correct I.
    assert [e for e in example.evidence if e.text.strip()] == []
    assert _triple(tif) == (0.0, 1.0, 0.0)
    assert tif.dominant == example.gold_channel  # "correct" without reading evidence


@pytest.mark.xfail(
    strict=True,
    reason="memo F3: scoring.py:145-148 collapses 'nothing was retrieved' and "
           "'passages were retrieved but none usable' onto the same TIF(0,1,0). "
           "The scorer cannot flag the difference, so an NEI row scores a "
           "guaranteed-correct I either way. Fixing this is downstream of the "
           "Gate B decision on r and must not be done ad hoc.",
)
def test_unusable_evidence_is_distinguishable_from_nothing_retrieved():
    scorer = NLIRetrievalScorer(nli=MockNLIBackend())
    claim = "Some claim whose evidence was all dropped as malformed."

    nothing_retrieved = scorer.score(claim, [])
    retrieved_but_unusable = scorer.score(
        claim, [Evidence("", 0.9), Evidence("   ", 0.9)]
    )

    assert _triple(nothing_retrieved) != _triple(retrieved_but_unusable)


# ---------------------------------------------------------------------------
# F7 / Gate C -- the expansion loop must not be a no-op
# ---------------------------------------------------------------------------
# Every test in tests/test_policy.py supplies a hand-written score_fn that
# returns hardcoded triples, so none of them can detect an inert expansion
# loop. These two drive NeutroPolicy.run with the REAL
# NLIRetrievalScorer.score.

_NEW_PASSAGES = [
    Evidence("The heart is a muscular organ that pumps blood through the body.", 0.9),
    Evidence("Cardiac muscle contracts to pump blood into the arteries.", 0.9),
]


def test_expansion_with_a_real_scorer_changes_the_triple():
    """Gate C: fails if the expansion loop is a no-op.

    Drives the real scorer, with a stub retriever that returns a genuinely NEW
    passage on each call, and asserts the triple actually moves between steps.
    """
    scorer = NLIRetrievalScorer(nli=MockNLIBackend())
    policy = NeutroPolicy(PolicyConfig(max_expansions=2))
    claim = "The heart pumps blood."
    initial = [Evidence("The region has a temperate climate.", 0.3)]

    calls = {"n": 0}

    def expand_fn(claim_, evidence, step):
        calls["n"] += 1
        return list(evidence) + [_NEW_PASSAGES[step]]

    decision = policy.run(claim, initial, scorer.score, expand_fn)

    assert calls["n"] >= 1, "expand_fn was never called"
    assert decision.history[0].action is Action.EXPAND
    assert len(decision.history) >= 2, "policy never re-scored after expanding"

    before, after = decision.history[0].tif, decision.history[1].tif
    assert _triple(before) != _triple(after), (
        "expansion loop is inert: the triple is unchanged after the retriever "
        "returned a new passage"
    )
    assert decision.expansions_used >= 1


def test_expansion_is_inert_when_the_retriever_returns_nothing_new():
    """The converse, documenting memo F7/F8.

    experiments/run_pipeline_demo.py exhausts its toy corpus on the first
    query, so expand_fn returns the same passages every time and the triple
    never moves -- which is what produced the identical steps 0/1/2 then
    ABSTAIN in the report's Section 5(d) screenshot. That is a property of the
    retriever, not of the policy.
    """
    scorer = NLIRetrievalScorer(nli=MockNLIBackend())
    policy = NeutroPolicy(PolicyConfig(max_expansions=2))
    initial = [Evidence("The region has a temperate climate.", 0.3)]

    def exhausted_expand_fn(claim_, evidence, step):
        return evidence  # retriever has nothing left to give

    decision = policy.run("The heart pumps blood.", initial,
                          scorer.score, exhausted_expand_fn)

    triples = {_triple(s.tif) for s in decision.history}
    assert len(triples) == 1, "expected an inert loop in this scenario"
    assert decision.action is Action.ABSTAIN
    assert decision.expansions_used == 2


# ---------------------------------------------------------------------------
# F11 -- every NLI backend must return a probability distribution
# ---------------------------------------------------------------------------
# MockNLIBackend.predict clamped the neutral term to 0 without renormalising
# and returned e.g. (0.10, 0.00, 1.00), summing to 1.1. The whole suite stayed
# green because no test ever checked the sum.

_PREDICT_INPUTS = [
    # (premise, hypothesis, which branch of MockNLIBackend.predict it exercises)
    ("Sharks are not mammals; they are fish.", "Sharks are mammals.",
     "high overlap + negation"),
    ("The Eiffel Tower is located in Paris France.", "The Eiffel Tower is located in Paris.",
     "high overlap, no negation"),
    ("Springfield is a common place name.", "The mayor won an award in 1998.",
     "low overlap -> neutral"),
    ("The project deadline is not Friday it moved.", "The project deadline is Friday.",
     "negation"),
    ("Water boils at 100 degrees Celsius at sea level.", "Water boils at 100 degrees Celsius.",
     "full overlap"),
    ("Any premise at all.", "", "empty hypothesis"),
]


def _assert_is_distribution(backend, premise, hypothesis, label):
    e, n, c = backend.predict(premise, hypothesis)
    for name, v in (("e", e), ("n", n), ("c", c)):
        assert 0.0 <= v <= 1.0, f"{label}: {name}={v} outside [0, 1]"
    assert abs((e + n + c) - 1.0) < 1e-9, (
        f"{label}: predict() returned ({e:.3f}, {n:.3f}, {c:.3f}) "
        f"summing to {e + n + c:.3f}, not a distribution"
    )


@pytest.mark.parametrize("premise,hypothesis,label", _PREDICT_INPUTS)
def test_mock_backend_returns_a_distribution(premise, hypothesis, label):
    _assert_is_distribution(MockNLIBackend(), premise, hypothesis, label)


@pytest.mark.skipif(
    os.environ.get("NEUTRORAG_TEST_HF") != "1",
    reason="downloads/loads a real NLI model; opt in with NEUTRORAG_TEST_HF=1",
)
@pytest.mark.parametrize("premise,hypothesis,label", _PREDICT_INPUTS)
def test_hf_backend_returns_a_distribution(premise, hypothesis, label):
    from neutrorag.scoring import HFNLIBackend

    _assert_is_distribution(HFNLIBackend(), premise, hypothesis, label)
