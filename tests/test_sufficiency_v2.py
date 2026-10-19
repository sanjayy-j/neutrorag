"""Tests for r_v2, the Addendum 5 repair.

Two properties are load-bearing and are tested directly:

  1. support_frac is DEPTH-INVARIANT in form. support_count divides by a fixed
     target, so it rises mechanically as the gate deepens retrieval -- the
     circularity measured at eta^2 = 0.9144 in Phase 3. support_frac divides by
     the number of passages actually retrieved.
  2. The relevance floor actually BINDS on real retrieved relevances. The legacy
     floor of 0.3 never bound on any of the 1200 base-depth passages in the
     Phase 3 run, which is what made the count-based component a constant.

Offline: no NLI model, no reader, no retriever. The "real relevances" used
below are read from the committed Phase 3 artefact.
"""
import csv
from pathlib import Path

import pytest

from neutrorag.scoring import Evidence, MockNLIBackend, NLIRetrievalScorer
from neutrorag.sufficiency import (
    DEFAULT_SUPPORT_TARGET, RELEVANCE_FLOOR_V2, retrieval_sufficiency,
    retrieval_sufficiency_v2, support_count, support_frac,
)

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. depth invariance
# ---------------------------------------------------------------------------
def test_support_frac_is_invariant_to_depth_when_quality_is_constant():
    """Doubling the passage count at the same quality mix must not move it.

    This is the property support_count lacks and the whole point of the repair.
    """
    high, low = 0.95, 0.10
    for k in (2, 4, 6, 10):
        rel = [high] * k + [low] * k
        assert support_frac(rel) == pytest.approx(0.5), f"moved at k={k}"


def test_support_count_by_contrast_rises_with_depth():
    """The defect being repaired, pinned so the contrast is explicit."""
    rel2 = [0.95] * 2
    rel8 = [0.95] * 8
    assert support_count(rel2, floor=0.3, target=10) == pytest.approx(0.2)
    assert support_count(rel8, floor=0.3, target=10) == pytest.approx(0.8)
    # support_frac sees no change, because the quality did not change
    assert support_frac(rel2) == support_frac(rel8) == pytest.approx(1.0)


def test_support_frac_tracks_quality_not_quantity():
    assert support_frac([0.95, 0.95, 0.95]) == pytest.approx(1.0)
    assert support_frac([0.95, 0.95, 0.10]) == pytest.approx(2 / 3)
    assert support_frac([0.10, 0.10, 0.10]) == pytest.approx(0.0)


def test_support_frac_empty_is_zero_not_a_crash():
    assert support_frac([]) == 0.0


def test_adding_a_bad_passage_lowers_support_frac():
    """Depth-invariant is not the same as depth-blind: fetching junk should
    lower the proportion that was any good."""
    assert support_frac([0.95, 0.95]) > support_frac([0.95, 0.95, 0.10])


# ---------------------------------------------------------------------------
# 2. the floor binds on REAL relevances
# ---------------------------------------------------------------------------
def _real_relevances():
    """max_relevance per question from the committed Phase 3 diagnostics."""
    path = REPO / "results" / "r_component_diagnostics.csv"
    if not path.exists():
        pytest.skip("committed Phase 3 diagnostics not present")
    with open(path, encoding="utf-8") as fh:
        return [float(r["max_relevance"]) for r in csv.DictReader(fh)]


def test_the_legacy_floor_never_binds_on_real_relevances():
    """The defect. DenseRetriever remaps cosine to [0,1], so 0.3 means
    cos >= -0.4, which essentially every text pair satisfies."""
    rel = _real_relevances()
    assert min(rel) > 0.3
    assert support_frac(rel, floor=0.3) == pytest.approx(1.0)


def test_the_v2_floor_does_bind_on_real_relevances():
    rel = _real_relevances()
    frac = support_frac(rel, floor=RELEVANCE_FLOOR_V2)
    assert 0.0 < frac < 1.0, f"floor must partition the real data, got {frac}"
    assert min(rel) < RELEVANCE_FLOOR_V2 < max(rel)


def test_the_v2_floor_is_the_registered_value():
    """Fixed in PREREGISTRATION.md Addendum 5 and not to be adjusted."""
    assert RELEVANCE_FLOOR_V2 == 0.770580


# ---------------------------------------------------------------------------
# 3. r_v2 composition
# ---------------------------------------------------------------------------
def test_r_v2_is_the_mean_of_exactly_two_components():
    bd = retrieval_sufficiency_v2(
        "alpha beta", passages=["alpha beta gamma", "delta"],
        relevances=[0.95, 0.10])
    assert set(bd.components) == {"coverage", "support_frac"}
    assert bd.r == pytest.approx(sum(bd.components.values()) / 2)
    assert all(w == pytest.approx(0.5) for w in bd.weights_used.values())


def test_r_v2_drops_diversity_and_connectivity():
    """Dropped, not retargeted: distinct-sources/k is identically 1.0 at every
    depth on this corpus, so no target makes it informative."""
    bd = retrieval_sufficiency_v2("a claim", passages=["a claim here"],
                                  relevances=[0.9])
    assert "diversity" not in bd.components
    assert "connectivity" not in bd.components


def test_r_v2_can_exceed_half_which_is_the_point():
    """T = r*e is bounded by r, and I >= 1-r, so at r < 0.5 truth can never
    win. r_v2 must be able to clear 0.5 or the gate still cannot decline."""
    bd = retrieval_sufficiency_v2(
        "alpha beta gamma", passages=["alpha beta gamma delta"],
        relevances=[0.95])
    assert bd.r > 0.5


def test_r_v2_empty_evidence_is_zero():
    bd = retrieval_sufficiency_v2("c", passages=[], relevances=[])
    assert bd.r == 0.0 and bd.components == {}


# ---------------------------------------------------------------------------
# 4. the legacy path is untouched
# ---------------------------------------------------------------------------
def test_scorer_defaults_to_the_legacy_path():
    """Every figure already in the report must stay reproducible, so r_v2 must
    be opt-in and the default must not change."""
    s = NLIRetrievalScorer(nli=MockNLIBackend())
    assert s.use_r_v2 is False
    ev = [Evidence("alpha beta gamma", 0.9, source="A")]
    legacy = s.sufficiency("alpha beta", ev)
    direct = retrieval_sufficiency(
        "alpha beta", passages=["alpha beta gamma"], relevances=[0.9],
        sources=["A"], support_target=DEFAULT_SUPPORT_TARGET)
    assert legacy.r == pytest.approx(direct.r)
    assert "support_count" in legacy.components


def test_scorer_with_use_r_v2_takes_the_new_path():
    ev = [Evidence("alpha beta gamma", 0.9, source="A")]
    s = NLIRetrievalScorer(nli=MockNLIBackend(), use_r_v2=True)
    bd = s.sufficiency("alpha beta", ev)
    assert set(bd.components) == {"coverage", "support_frac"}


def test_r_override_still_wins_over_r_v2():
    """The r-ablation must keep working regardless of which path is selected."""
    s = NLIRetrievalScorer(nli=MockNLIBackend(), use_r_v2=True, r_override=0.9)
    bd = s.sufficiency("a", [Evidence("a b", 0.9)])
    assert bd.r == pytest.approx(0.9)


def test_r_v2_makes_no_nli_call():
    """Same guarantee as the legacy path: r must not depend on the NLI head."""
    class _Exploding:
        def predict(self, premise, hypothesis):  # pragma: no cover
            raise AssertionError("sufficiency must not call the NLI backend")

    s = NLIRetrievalScorer(nli=_Exploding(), use_r_v2=True)
    s.sufficiency("alpha", [Evidence("alpha beta", 0.9)])
