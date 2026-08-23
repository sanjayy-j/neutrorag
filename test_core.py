from neutrorag.core import TIF


def test_clamped_into_unit_interval():
    t = TIF(T=1.5, I=-0.2, F=0.3)
    assert t.T == 1.0 and t.I == 0.0 and t.F == 0.3


def test_dominant_channel():
    assert TIF(0.8, 0.1, 0.1).dominant == "T"
    assert TIF(0.1, 0.8, 0.1).dominant == "I"
    assert TIF(0.1, 0.1, 0.8).dominant == "F"


def test_independence_mass_can_exceed_one():
    # Support and contradiction co-occurring is the case a summed-to-one
    # representation cannot express; here mass > 1 proves independence.
    t = TIF(T=0.6, I=0.2, F=0.7)
    assert t.mass > 1.0
    assert t.is_contradiction


def test_confidence_monotonicity():
    strong = TIF(0.9, 0.05, 0.05).confidence
    weak = TIF(0.9, 0.5, 0.5).confidence
    assert strong > weak
    assert 0.0 <= weak <= strong <= 1.0


def test_collapse_merges_i_and_f():
    cs = TIF(T=0.5, I=0.3, F=0.4).collapse()
    assert cs.truth == 0.5
    assert abs(cs.uncertainty - 0.7) < 1e-9
