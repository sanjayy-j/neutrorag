"""NeutroRAG: a neutrosophic trust layer for retrieval-augmented generation.

Separates *missing evidence* (indeterminacy) from *contradiction* (falsity) and
acts on each differently -- expanding retrieval, surfacing conflict, or
abstaining -- instead of hallucinating.
"""
from .core import TIF, CollapsedScore, clamp
from .scoring import (
    Evidence,
    NLIRetrievalScorer,
    MockNLIBackend,
    HFNLIBackend,
    NLIBackend,
)
from .policy import Action, PolicyConfig, Decision, Step, NeutroPolicy

__version__ = "0.1.0"

__all__ = [
    "TIF", "CollapsedScore", "clamp",
    "Evidence", "NLIRetrievalScorer", "MockNLIBackend", "HFNLIBackend", "NLIBackend",
    "Action", "PolicyConfig", "Decision", "Step", "NeutroPolicy",
    "__version__",
]
