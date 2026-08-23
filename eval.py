"""Evaluation metrics for NeutroRAG.

Includes the metric that is unique to this work -- *failure-type attribution*:
does the system correctly say WHY it is unsure (missing evidence vs.
contradiction)? -- and the *fuzzy-collapse ablation* that motivates the whole
design by showing what is lost when I and F are merged into one scalar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .core import TIF

# Gold failure channel labels.
#   T  <- claim is supported            (FEVER: SUPPORTS)
#   I  <- not enough evidence           (FEVER: NOT ENOUGH INFO)
#   F  <- claim is contradicted         (FEVER: REFUTES)
CHANNELS = ("T", "I", "F")


# ---------------------------------------------------------------------------
# Selective prediction / calibration
# ---------------------------------------------------------------------------
def risk_coverage(confidences: Sequence[float], correct: Sequence[bool]) -> Dict[str, float]:
    """Area under the risk-coverage curve (lower is better) + selective accuracy
    at 50%/80% coverage. Answers are taken in descending confidence order."""
    conf = np.asarray(confidences, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    order = np.argsort(-conf)
    ok_sorted = ok[order]
    n = len(ok_sorted)
    coverages = np.arange(1, n + 1) / n
    accuracies = np.cumsum(ok_sorted) / np.arange(1, n + 1)
    risks = 1.0 - accuracies
    aurc = float(np.trapezoid(risks, coverages)) if n > 1 else float(risks[0])

    def acc_at(cov: float) -> float:
        k = max(1, int(round(cov * n)))
        return float(ok_sorted[:k].mean())

    return {"aurc": aurc, "sel_acc@0.5": acc_at(0.5), "sel_acc@0.8": acc_at(0.8),
            "full_acc": float(ok.mean())}


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool],
                               n_bins: int = 10) -> float:
    conf = np.asarray(confidences, dtype=float)
    ok = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Failure-type attribution (the distinctive metric)
# ---------------------------------------------------------------------------
def predicted_channel(tif: TIF) -> str:
    return tif.dominant


def failure_type_attribution(tifs: Sequence[TIF], gold: Sequence[str]) -> Dict[str, object]:
    """Accuracy of argmax(T, I, F) against the gold T/I/F channel, plus a 3x3
    confusion matrix. This is what a collapsed model structurally cannot do."""
    idx = {ch: i for i, ch in enumerate(CHANNELS)}
    conf = np.zeros((3, 3), dtype=int)
    correct = 0
    for tif, g in zip(tifs, gold):
        p = predicted_channel(tif)
        conf[idx[g], idx[p]] += 1
        correct += int(p == g)
    return {"accuracy": correct / len(gold), "confusion": conf, "labels": CHANNELS}


def collapsed_attribution(tifs: Sequence[TIF], gold: Sequence[str],
                          uncertainty_threshold: float = 0.5) -> Dict[str, object]:
    """Best a fuzzy/probabilistic (I,F-merged) baseline can do at attribution.

    With only (truth, uncertainty), the model can separate 'supported' (T) from
    'unsure', but CANNOT tell missing-evidence (I) from contradiction (F); we
    give it the benefit of the doubt by letting it guess the more frequent
    unsure class, and still it caps out well below the full model.
    """
    gold_arr = list(gold)
    unsure_gold = [g for g in gold_arr if g != "T"]
    majority_unsure = max(("I", "F"), key=unsure_gold.count) if unsure_gold else "I"
    correct = 0
    for tif, g in zip(tifs, gold_arr):
        cs = tif.collapse()
        pred = "T" if cs.uncertainty < uncertainty_threshold else majority_unsure
        correct += int(pred == g)
    return {"accuracy": correct / len(gold_arr), "forced_unsure_class": majority_unsure}


@dataclass
class AblationResult:
    full_attribution: float
    collapsed_attribution: float

    @property
    def delta(self) -> float:
        return self.full_attribution - self.collapsed_attribution


def fuzzy_collapse_ablation(tifs: Sequence[TIF], gold: Sequence[str]) -> AblationResult:
    """The month-3 go/no-go experiment: full neutrosophic vs. I+F collapsed."""
    full = failure_type_attribution(tifs, gold)["accuracy"]
    collapsed = collapsed_attribution(tifs, gold)["accuracy"]
    return AblationResult(full_attribution=full, collapsed_attribution=collapsed)  # type: ignore[arg-type]


def format_confusion(conf: np.ndarray, labels: Sequence[str] = CHANNELS) -> str:
    header = "gold\\pred " + "  ".join(f"{l:>4}" for l in labels)
    rows = [header]
    for i, l in enumerate(labels):
        rows.append(f"{l:>8}  " + "  ".join(f"{conf[i, j]:>4d}" for j in range(len(labels))))
    return "\n".join(rows)
