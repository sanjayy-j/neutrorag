"""First real exercise of eval.risk_coverage and eval.expected_calibration_error.

Both have been in the repository since before the First Review and had never
been run on anything. Task 3 of the Track B plan is the first use. These tests
record what they actually do.

Two defects are confirmed and pinned below as characterisation tests (they
PASS, documenting present behaviour), each paired with a strict xfail stating
the behaviour that would be correct. Fixing either changes reported numbers, so
neither is fixed here without a decision.

  D1  expected_calibration_error silently DROPS any item whose confidence is
      exactly 0.0, because the first bin is the half-open (0.0, 0.1]. The bin
      weights then sum to less than 1 and ECE is under-estimated. An abstaining
      system is exactly the kind that emits confidence 0.0, so this is not a
      corner case for Track B -- it is the main case.

  D2  risk_coverage breaks ties in confidence by array index, so AURC depends
      on the order the rows happen to arrive in. With every confidence tied,
      AURC ranges over 0.150 to 0.683 on the same multiset of outcomes.

A third, milder issue is recorded in
test_aurc_is_not_normalised_to_the_unit_interval: AURC integrates coverage over
[1/n, 1] rather than (0, 1], so its maximum is (n-1)/n, not 1. Comparisons at
equal n are unaffected; comparisons across different n are not.
"""
import pytest

from neutrorag.eval import expected_calibration_error as ece
from neutrorag.eval import risk_coverage


# ---------------------------------------------------------------------------
# D1 -- confidence exactly 0.0 is dropped from every bin
# ---------------------------------------------------------------------------
def test_ece_drops_confidence_exactly_zero():
    """Characterisation. Four items at confidence 0.0 that are all correct are
    maximally mis-calibrated: the true ECE is 1.0. The implementation reports
    0.0, i.e. perfectly calibrated, because no bin contains them."""
    assert ece([0.0, 0.0, 0.0, 0.0], [True, True, True, True]) == 0.0


def test_ece_is_correct_just_above_zero():
    """The defect is exactly at the boundary: nudging the confidence off 0.0
    recovers the right answer, which localises the bug to the bin edge."""
    assert ece([1e-9] * 4, [True] * 4) == pytest.approx(1.0)


def test_ece_underweights_when_some_confidences_are_zero():
    """Characterisation. Two of four items sit at 0.0 and are dropped, so the
    remaining weight is 0.5 and the reported ECE is halved -- here to 0.0,
    because the surviving items at confidence 1.0 are correct."""
    assert ece([0.0, 0.0, 1.0, 1.0], [True, True, True, True]) == 0.0


@pytest.mark.xfail(
    strict=True,
    reason="D1: expected_calibration_error uses (lo, hi] bins, so the first "
           "bin excludes confidence 0.0 and those items enter no bin at all. "
           "The weighted average is then under-normalised. The fix is to make "
           "the lowest bin closed on the left, but it changes every ECE this "
           "project will report, so it is a decision rather than a patch.",
)
def test_ece_should_count_every_item():
    # all correct at confidence 0.0 -> |acc - conf| = 1.0 over the whole sample
    assert ece([0.0] * 4, [True] * 4) == pytest.approx(1.0)
    # half at 0.0 (correct) and half at 1.0 (correct) -> 0.5
    assert ece([0.0, 0.0, 1.0, 1.0], [True] * 4) == pytest.approx(0.5)


def test_ece_is_sane_when_no_confidence_is_zero():
    """Guards the fix: this case must not move when D1 is addressed."""
    assert ece([1.0, 1.0], [True, True]) == pytest.approx(0.0)
    assert ece([1.0, 1.0], [False, False]) == pytest.approx(1.0)
    assert ece([0.5, 0.5], [True, False]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# D2 -- AURC depends on input order when confidences tie
# ---------------------------------------------------------------------------
_TIED = [0.5] * 6
_CORRECT_FIRST = [True, True, True, False, False, False]
_WRONG_FIRST = [False, False, False, True, True, True]


def test_aurc_depends_on_row_order_when_confidences_tie():
    """Characterisation. Same confidences, same multiset of outcomes, 4.5x
    different AURC -- decided entirely by which rows came first."""
    a = risk_coverage(_TIED, _CORRECT_FIRST)["aurc"]
    b = risk_coverage(_TIED, _WRONG_FIRST)["aurc"]
    assert a == pytest.approx(0.15, abs=1e-3)
    assert b == pytest.approx(0.6833, abs=1e-3)
    assert b > a * 4


@pytest.mark.xfail(
    strict=True,
    reason="D2: np.argsort(-conf) resolves ties by index, so AURC is a "
           "function of dataset order whenever confidences tie. A system that "
           "assigns one confidence to every abstention -- which is what an "
           "abstaining system does -- has a large tied block, so this affects "
           "the Track B comparison directly. A tie-aware AURC would average "
           "risk within each tied block.",
)
def test_aurc_should_be_invariant_to_order_within_a_tie():
    a = risk_coverage(_TIED, _CORRECT_FIRST)["aurc"]
    b = risk_coverage(_TIED, _WRONG_FIRST)["aurc"]
    assert a == pytest.approx(b)


def test_aurc_ranks_separable_cases_correctly():
    """AURC is not broken in general: given distinct confidences it orders a
    good ranking below a bad one, which is the property Task 3 relies on."""
    good = risk_coverage([0.9, 0.8, 0.2, 0.1], [True, True, False, False])["aurc"]
    bad = risk_coverage([0.9, 0.8, 0.2, 0.1], [False, False, True, True])["aurc"]
    assert good < bad
    assert risk_coverage([0.9, 0.8, 0.2, 0.1], [True] * 4)["aurc"] == 0.0


def test_aurc_is_not_normalised_to_the_unit_interval():
    """Characterisation. With every prediction wrong the risk is 1 everywhere,
    so a normalised AURC would be 1.0. It reports (n-1)/n because coverage is
    integrated over [1/n, 1]. Safe for same-n comparisons, wrong in absolute
    terms and not comparable across different n -- which matters here, because
    the answerable and unanswerable arms have different n."""
    assert risk_coverage([0.9, 0.8, 0.2, 0.1], [False] * 4)["aurc"] == pytest.approx(0.75)
    assert risk_coverage([0.9] * 10, [False] * 10)["aurc"] == pytest.approx(0.9)


def test_selective_accuracy_at_coverage_is_correct():
    """The other half of risk_coverage's output, which is not affected by
    either defect."""
    out = risk_coverage([0.9, 0.8, 0.2, 0.1], [True, True, False, False])
    assert out["sel_acc@0.5"] == pytest.approx(1.0)
    assert out["full_acc"] == pytest.approx(0.5)


def test_risk_coverage_single_item_does_not_crash():
    assert risk_coverage([0.5], [True])["aurc"] == 0.0
    assert risk_coverage([0.5], [False])["aurc"] == 1.0
