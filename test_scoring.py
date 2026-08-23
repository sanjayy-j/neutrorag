from neutrorag.scoring import Evidence, MockNLIBackend, NLIRetrievalScorer


def make_scorer():
    return NLIRetrievalScorer(nli=MockNLIBackend())


def test_no_evidence_is_pure_indeterminacy():
    tif = make_scorer().score("Some unverifiable claim about X.", [])
    assert tif.dominant == "I"
    assert tif.I == 1.0 and tif.T == 0.0 and tif.F == 0.0


def test_supporting_evidence_yields_truth():
    s = make_scorer()
    tif = s.score(
        "The Eiffel Tower is located in Paris.",
        [Evidence("The Eiffel Tower is located in Paris France.", 0.9)],
    )
    assert tif.dominant == "T"


def test_contradicting_evidence_yields_falsity():
    s = make_scorer()
    tif = s.score(
        "Sharks are mammals.",
        [Evidence("Sharks are not mammals sharks are fish.", 0.9)],
    )
    assert tif.dominant == "F"


def test_offtopic_evidence_yields_indeterminacy():
    s = make_scorer()
    tif = s.score(
        "The mayor won an award in 1998.",
        [Evidence("Springfield is a common place name in many countries.", 0.9)],
    )
    assert tif.dominant == "I"


def test_conflicting_sources_raise_mass_above_one():
    """One source supports, one contradicts -> T and F both non-trivial."""
    s = make_scorer()
    tif = s.score(
        "The project deadline is Friday.",
        [
            Evidence("The project deadline is Friday this week.", 0.9),
            Evidence("The project deadline is not Friday it moved.", 0.9),
        ],
    )
    assert tif.T > 0.2 and tif.F > 0.2  # genuine contradiction represented
