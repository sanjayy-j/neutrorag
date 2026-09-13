"""Sensitivity sweep over the policy thresholds, lambda, and the sufficiency weights.

Memo F12: "Policy thresholds (0.50 / 0.50 / 0.40 / 0.30) are asserted with no
justification and are entangled with r's scale. Note that whenever r <= 0.5,
I >= 0.5 automatically and the policy fires EXPAND on everything."

Nothing here retunes anything. Every default stays exactly where it is; this
reports what would happen if it moved, so the choice of defaults becomes a
stated, inspectable decision rather than an unexamined one.

Four sweeps, all on the 200 FEVER claims with the NLI outputs replayed from
fever_run_200.csv (the model is never re-run, so every condition sees identical
model output):

  A. r swept directly via r_override -- answers "at what r does I take over?"
  B. the four policy thresholds, one at a time
  C. lambda (the weight of disagree in F)
  D. the four sufficiency weights, one at a time

    python experiments/run_sensitivity_sweep.py

Outputs results/sensitivity_sweep.csv and results/sensitivity_sweep.png.

Caveat (memo F9), which applies to sweeps A and D especially: on FEVER only the
coverage and support_count components of r are available, and support_count is
pinned near 1/3 because 185 of 200 rows carry a single passage. The connectivity
and diversity weights are therefore INERT here -- they are dropped and the
remaining weights renormalised. A meaningful weight sweep needs Step 9 data.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_results_table import CHANNELS, accuracy  # noqa: E402
from neutrorag.fever import load_fever_hf  # noqa: E402
from neutrorag.policy import Action, NeutroPolicy, PolicyConfig  # noqa: E402
from neutrorag.scoring import NLIRetrievalScorer  # noqa: E402
from neutrorag.sufficiency import SufficiencyWeights  # noqa: E402

ACTIONS = [a.value for a in Action]
DEFAULTS = PolicyConfig()


class _Replay:
    def __init__(self, triple, times):
        self._rows = [triple] * times
        self._i = 0

    def predict(self, premise, hypothesis):
        out = self._rows[self._i]
        self._i += 1
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="fever_run_200.csv")
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def load_corpus(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    examples = load_fever_hf(split="validation", max_examples=len(rows))
    corpus = []
    for ex, row in zip(examples, rows):
        if row["e"] == "":
            continue
        corpus.append((ex, (float(row["e"]), float(row["n"]), float(row["c"]))))
    return corpus


def score_all(corpus, *, r_override=None, lam=0.5, weights=None):
    """Return (gold, predicted_channel, tif) for every claim under one setting."""
    out = []
    for ex, triple in corpus:
        kw = {"nli": _Replay(triple, len(ex.evidence)), "lam": lam}
        if r_override is not None:
            kw["r_override"] = r_override
        if weights is not None:
            kw["sufficiency_weights"] = weights
        scorer = NLIRetrievalScorer(**kw)
        tif = scorer.score(ex.claim, ex.evidence)
        out.append((ex.gold_channel, tif.dominant, tif))
    return out


def action_mix(scored, config: PolicyConfig):
    policy = NeutroPolicy(config)
    counts = Counter(policy.decide_once(tif)[0].value for _, _, tif in scored)
    total = sum(counts.values())
    return {a: counts.get(a, 0) / total for a in ACTIONS}


def summarise(scored, config: PolicyConfig):
    acc = accuracy([(g, p) for g, p, _ in scored])
    pred_i = sum(1 for _, p, _ in scored if p == "I") / len(scored)
    mix = action_mix(scored, config)
    return acc, pred_i, mix


def main() -> None:
    args = parse_args()
    corpus = load_corpus(REPO / args.csv)
    print(f"corpus: {len(corpus)} claims with recorded NLI output")

    records = []

    def record(sweep, param, value, scored, config=DEFAULTS, extra=None):
        acc, pred_i, mix = summarise(scored, config)
        row = {"sweep": sweep, "parameter": param, "value": round(value, 4),
               "attribution_accuracy": round(acc, 4), "frac_predicted_I": round(pred_i, 4)}
        row.update({f"action_{a}": round(mix[a], 4) for a in ACTIONS})
        row.update(extra or {})
        records.append(row)
        return row

    # -- A. r swept directly -------------------------------------------------
    print("A. sweeping r via r_override ...")
    a_r, a_acc, a_predI, a_expand = [], [], [], []
    for i in range(0, 101, 2):
        r = i / 100
        scored = score_all(corpus, r_override=r)
        row = record("A_r_override", "r", r, scored)
        a_r.append(r); a_acc.append(row["attribution_accuracy"])
        a_predI.append(row["frac_predicted_I"]); a_expand.append(row["action_expand"])

    # the LARGEST r at which the condition still holds -- i.e. where the cliff is
    takeover = max((r for r, f in zip(a_r, a_predI) if f >= 0.999), default=None)
    half = max((r for r, f in zip(a_r, a_predI) if f >= 0.5), default=None)
    print(f"   I is argmax for >=50% of claims at every r <= {half}")
    print(f"   I is argmax for 100% of claims at every r <= {takeover}")

    # -- B. policy thresholds ------------------------------------------------
    print("B. sweeping policy thresholds ...")
    baseline_default_r = score_all(corpus)                 # r = sufficiency
    baseline_const_r = score_all(corpus, r_override=0.9)   # r = 0.9
    thresholds = {"tau_T": "tau_T", "tau_I": "tau_I", "tau_F": "tau_F",
                  "tau_T_conflict": "tau_T_conflict"}
    b_curves = {}
    for name in thresholds:
        xs, ys_suff, ys_const = [], [], []
        for i in range(0, 101, 5):
            v = i / 100
            cfg = PolicyConfig(**{**DEFAULTS.__dict__, name: v})
            r1 = record("B_threshold", name, v, baseline_default_r, cfg,
                        {"r_condition": "sufficiency"})
            r2 = record("B_threshold", name, v, baseline_const_r, cfg,
                        {"r_condition": "const_0.9"})
            xs.append(v); ys_suff.append(r1["action_expand"]); ys_const.append(r2["action_expand"])
        b_curves[name] = (xs, ys_suff, ys_const)

    # -- C. lambda -----------------------------------------------------------
    print("C. sweeping lambda ...")
    c_x, c_acc, c_surface = [], [], []
    for i in range(0, 21):
        lam = i / 10
        scored = score_all(corpus, lam=lam)
        row = record("C_lambda", "lambda", lam, scored, extra={"r_condition": "sufficiency"})
        c_x.append(lam); c_acc.append(row["attribution_accuracy"])
        c_surface.append(row["action_surface_contradiction"])

    # -- D. sufficiency weights ----------------------------------------------
    print("D. sweeping sufficiency weights ...")
    d_curves = {}
    for comp in ("coverage", "support_count", "connectivity", "diversity"):
        xs, ys = [], []
        for i in range(0, 101, 10):
            w = i / 100
            rest = (1.0 - w) / 3.0
            kwargs = {c: rest for c in
                      ("coverage", "support_count", "connectivity", "diversity")}
            kwargs[comp] = w
            scored = score_all(corpus, weights=SufficiencyWeights(**kwargs))
            row = record("D_weight", comp, w, scored, extra={"r_condition": "sufficiency"})
            xs.append(w); ys.append(row["attribution_accuracy"])
        d_curves[comp] = (xs, ys)
        if len(set(round(y, 6) for y in ys)) == 1:
            print(f"   {comp}: INERT on FEVER (accuracy constant at {ys[0]:.4f})")

    # -- write CSV -----------------------------------------------------------
    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cols = ["sweep", "parameter", "value", "r_condition", "attribution_accuracy",
            "frac_predicted_I"] + [f"action_{a}" for a in ACTIONS]
    with open(outdir / "sensitivity_sweep.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            rec.setdefault("r_condition", "")
            w.writerow(rec)

    # -- figure --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("NeutroRAG sensitivity sweep (200 FEVER claims, I = max(1-r, n))",
                 fontsize=13)

    ax = axes[0][0]
    ax.plot(a_r, a_predI, label="fraction predicted I", lw=2)
    ax.plot(a_r, a_expand, label="fraction action=EXPAND", lw=2, ls="--")
    ax.plot(a_r, a_acc, label="attribution accuracy", lw=2, ls=":")
    ax.axvline(0.5, color="crimson", lw=1)
    ax.text(0.52, 0.92, "r = 0.5", color="crimson", fontsize=9,
            transform=ax.transAxes)
    ax.set_xlabel("r (overridden)"); ax.set_ylabel("fraction")
    ax.set_title("A. I takes over as r falls"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0][1]
    for name, (xs, ys_s, ys_c) in b_curves.items():
        ax.plot(xs, ys_s, lw=1.8, label=f"{name} (r=suff)")
    ax.set_xlabel("threshold value"); ax.set_ylabel("fraction action=EXPAND")
    ax.set_title("B. thresholds vs EXPAND rate (r = sufficiency)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.plot(c_x, c_acc, lw=2, label="attribution accuracy")
    ax.plot(c_x, c_surface, lw=2, ls="--", label="fraction SURFACE_CONTRADICTION")
    ax.axvline(0.5, color="crimson", lw=1)
    ax.text(0.27, 0.92, "default lambda = 0.5", color="crimson", fontsize=9,
            transform=ax.transAxes)
    ax.set_xlabel("lambda"); ax.set_ylabel("value")
    ax.set_title("C. lambda"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1][1]
    for comp, (xs, ys) in d_curves.items():
        ax.plot(xs, ys, lw=1.8, marker="o", ms=3, label=comp)
    ax.set_xlabel("weight on that component (others share the remainder)")
    ax.set_ylabel("attribution accuracy")
    ax.set_title("D. sufficiency weights\n(connectivity/diversity inert on FEVER)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / "sensitivity_sweep.png", dpi=150)

    print(f"\nwrote {outdir / 'sensitivity_sweep.csv'} ({len(records)} rows)")
    print(f"wrote {outdir / 'sensitivity_sweep.png'}")

    # -- the F12 answer ------------------------------------------------------
    print("\n" + "=" * 70)
    print("F12: at what r does I take over?")
    print("=" * 70)
    print("  Analytically, I = max(1-r, n) >= 1-r and T = r*e <= r, so for r < 0.5")
    print("  we have I >= 1-r > r >= T: I beats T unconditionally. That is the SAME")
    print("  threshold as the old I = (1-r) + r*n, which also satisfies I >= 1-r.")
    print("  Changing I did not move the F12 regime boundary.")
    print(f"  Empirically on FEVER: I is the argmax for >=50% of claims at r <= {half},")
    print(f"  and for 100% of claims at r <= {takeover}.")
    for r_probe in (0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
        rec = next(x for x in records
                   if x["sweep"] == "A_r_override" and abs(x["value"] - r_probe) < 1e-9)
        print(f"    r={r_probe:.1f}: predicted-I {rec['frac_predicted_I']:.1%}, "
              f"EXPAND {rec['action_expand']:.1%}, accuracy {rec['attribution_accuracy']:.3f}")


if __name__ == "__main__":
    main()
