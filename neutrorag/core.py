"""Core neutrosophic representation for NeutroRAG.

A single-valued neutrosophic value carries three *independent* memberships in
[0, 1] -- Truth (T), Indeterminacy (I) and Falsity (F). Crucially they need NOT
sum to 1. That independence is the whole theoretical justification for NeutroRAG:
probability and fuzzy membership collapse "unsupported" and "contradicted" onto a
single axis, whereas neutrosophy keeps *missing evidence* (I) and *active
contradiction* (F) separate so the system can act on each differently.
"""
from __future__ import annotations

from dataclasses import dataclass


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


@dataclass(frozen=True)
class TIF:
    """A single-valued neutrosophic triple (Truth, Indeterminacy, Falsity)."""

    T: float
    I: float
    F: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "T", clamp(self.T))
        object.__setattr__(self, "I", clamp(self.I))
        object.__setattr__(self, "F", clamp(self.F))

    # -- derived quantities -------------------------------------------------
    @property
    def dominant(self) -> str:
        """Return 'T', 'I' or 'F' -- the strongest channel (argmax)."""
        return max((("T", self.T), ("I", self.I), ("F", self.F)), key=lambda kv: kv[1])[0]

    @property
    def net_truth(self) -> float:
        """T - F in [-1, 1]: how much support outweighs contradiction."""
        return self.T - self.F

    @property
    def confidence(self) -> float:
        """Scalar answer-confidence in [0, 1] for risk-coverage ranking.

        Monotone: high only when support is high AND both indeterminacy and
        falsity are low. Used to order answers when deciding what to abstain on.
        """
        return clamp(self.T * (1.0 - self.I) * (1.0 - self.F))

    @property
    def mass(self) -> float:
        """T + I + F. Exceeds 1 exactly when support and conflict co-occur --
        the fingerprint of a genuine contradiction that a summed-to-one model
        cannot represent."""
        return self.T + self.I + self.F

    @property
    def is_contradiction(self) -> bool:
        """Both meaningful support and meaningful conflict are present."""
        return self.T >= 0.3 and self.F >= 0.3

    # -- ablation -----------------------------------------------------------
    def collapse(self) -> "CollapsedScore":
        """Fold I and F into a single uncertainty scalar.

        This is the *fuzzy-collapse ablation*: it deliberately discards the
        distinction NeutroRAG is built on, so experiments can show the drop.
        """
        return CollapsedScore(truth=self.T, uncertainty=clamp(self.I + self.F))

    def as_dict(self) -> dict:
        return {"T": self.T, "I": self.I, "F": self.F}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"TIF(T={self.T:.3f}, I={self.I:.3f}, F={self.F:.3f})"


@dataclass(frozen=True)
class CollapsedScore:
    """Baseline representation used only for the ablation: a single truth value
    and one merged uncertainty scalar (I and F are no longer distinguishable)."""

    truth: float
    uncertainty: float

    @property
    def confidence(self) -> float:
        return clamp(self.truth * (1.0 - self.uncertainty))
