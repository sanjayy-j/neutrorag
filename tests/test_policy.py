from neutrorag.core import TIF
from neutrorag.policy import Action, NeutroPolicy, PolicyConfig
from neutrorag.scoring import Evidence


def test_answer_when_well_supported():
    p = NeutroPolicy()
    action, _ = p.decide_once(TIF(T=0.8, I=0.1, F=0.1))
    assert action is Action.ANSWER


def test_surface_contradiction_when_support_and_conflict():
    p = NeutroPolicy()
    action, _ = p.decide_once(TIF(T=0.5, I=0.1, F=0.6))
    assert action is Action.SURFACE_CONTRADICTION


def test_expand_when_indeterminate():
    p = NeutroPolicy()
    action, _ = p.decide_once(TIF(T=0.1, I=0.8, F=0.05))
    assert action is Action.EXPAND


def test_hedge_when_weak_and_mixed():
    p = NeutroPolicy()
    action, _ = p.decide_once(TIF(T=0.3, I=0.3, F=0.2))
    assert action is Action.HEDGE


def test_expansion_loop_recovers_answer():
    """High I first; the expand callback 'finds' good evidence -> ANSWER."""
    calls = {"n": 0}

    def score_fn(claim, evidence):
        # Before expansion: indeterminate. After: supported.
        if any(e.text == "good" for e in evidence):
            return TIF(T=0.85, I=0.1, F=0.05)
        return TIF(T=0.1, I=0.8, F=0.05)

    def expand_fn(claim, evidence, step):
        calls["n"] += 1
        return list(evidence) + [Evidence("good", 0.9)]

    decision = NeutroPolicy(PolicyConfig(max_expansions=2)).run(
        "claim", [Evidence("weak", 0.2)], score_fn, expand_fn)
    assert decision.action is Action.ANSWER
    assert calls["n"] == 1
    assert decision.expansions_used == 1


def test_abstain_when_indeterminacy_irreducible():
    """High I that expansion never fixes -> ABSTAIN after budget is spent."""
    def score_fn(claim, evidence):
        return TIF(T=0.1, I=0.85, F=0.05)

    def expand_fn(claim, evidence, step):
        return evidence  # never helps

    decision = NeutroPolicy(PolicyConfig(max_expansions=2)).run(
        "claim", [Evidence("x", 0.2)], score_fn, expand_fn)
    assert decision.action is Action.ABSTAIN
    assert decision.expansions_used == 2


def test_abstain_when_no_expand_fn_available():
    def score_fn(claim, evidence):
        return TIF(T=0.1, I=0.85, F=0.05)

    decision = NeutroPolicy().run("claim", [Evidence("x", 0.2)], score_fn, expand_fn=None)
    assert decision.action is Action.ABSTAIN
