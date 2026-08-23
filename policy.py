"""The NeutroRAG decision policy -- the core research contribution.

A scalar confidence can only choose between *answer* and *abstain*. NeutroRAG
reads the whole (T, I, F) triple and picks among five actions, treating missing
evidence and contradiction as different problems:

    High T, low I, low F         -> ANSWER
    High F (with some T)         -> SURFACE_CONTRADICTION   (conflict isn't fixed by more retrieval)
    High I (budget remaining)    -> EXPAND retrieval, then re-score
    High I (budget exhausted)    -> ABSTAIN
    weak / mixed                 -> HEDGE

The staged loop -- indeterminacy *triggers action*, not just refusal -- is what
distinguishes NeutroRAG from prior selective-prediction / abstention work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Sequence, Tuple

from .core import TIF
from .scoring import Evidence


class Action(str, Enum):
    ANSWER = "answer"
    EXPAND = "expand"
    SURFACE_CONTRADICTION = "surface_contradiction"
    HEDGE = "hedge"
    ABSTAIN = "abstain"


@dataclass
class PolicyConfig:
    tau_T: float = 0.50          # support needed to answer
    tau_I: float = 0.50          # indeterminacy that triggers expansion/abstention
    tau_F: float = 0.40          # contradiction that triggers surfacing
    tau_T_conflict: float = 0.30  # support level at which high F reads as genuine conflict
    max_expansions: int = 2      # retrieval-expansion budget


@dataclass
class Step:
    tif: TIF
    action: Action
    rationale: str


@dataclass
class Decision:
    action: Action
    rationale: str
    tif: TIF
    expansions_used: int = 0
    history: List[Step] = field(default_factory=list)


# Type aliases for the pluggable callbacks the controller drives.
ScoreFn = Callable[[str, Sequence[Evidence]], TIF]
ExpandFn = Callable[[str, Sequence[Evidence], int], Sequence[Evidence]]


class NeutroPolicy:
    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig()

    def decide_once(self, tif: TIF) -> Tuple[Action, str]:
        """Pure, stateless mapping from a (T, I, F) triple to an action."""
        c = self.config
        if tif.F >= c.tau_F and tif.T >= c.tau_T_conflict:
            return (Action.SURFACE_CONTRADICTION,
                    f"support (T={tif.T:.2f}) and conflict (F={tif.F:.2f}) co-occur")
        if tif.F >= c.tau_F:
            return (Action.SURFACE_CONTRADICTION, f"contradiction dominates (F={tif.F:.2f})")
        if tif.I >= c.tau_I:
            return (Action.EXPAND, f"evidence insufficient (I={tif.I:.2f})")
        if tif.T >= c.tau_T:
            return (Action.ANSWER, f"well supported (T={tif.T:.2f})")
        return (Action.HEDGE,
                f"weak/mixed signals (T={tif.T:.2f}, I={tif.I:.2f}, F={tif.F:.2f})")

    def run(
        self,
        claim: str,
        evidence: Sequence[Evidence],
        score_fn: ScoreFn,
        expand_fn: Optional[ExpandFn] = None,
    ) -> Decision:
        """Score, then apply the staged policy, expanding retrieval under high I."""
        history: List[Step] = []
        tif = score_fn(claim, evidence)

        for step in range(self.config.max_expansions + 1):
            action, rationale = self.decide_once(tif)
            history.append(Step(tif=tif, action=action, rationale=rationale))

            if action is Action.EXPAND:
                budget_left = step < self.config.max_expansions
                if expand_fn is not None and budget_left:
                    evidence = expand_fn(claim, evidence, step)
                    tif = score_fn(claim, evidence)
                    continue  # re-evaluate with the enlarged evidence set
                # Nothing more to retrieve -> indeterminacy is irreducible here.
                return Decision(
                    action=Action.ABSTAIN,
                    rationale="indeterminacy persists after exhausting retrieval budget",
                    tif=tif,
                    expansions_used=step,
                    history=history,
                )

            return Decision(action=action, rationale=rationale, tif=tif,
                            expansions_used=step, history=history)

        # Unreachable, but keep a well-defined return.
        return Decision(action=Action.ABSTAIN, rationale="budget exhausted",
                        tif=tif, expansions_used=self.config.max_expansions, history=history)
