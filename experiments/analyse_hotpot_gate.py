"""Post-hoc analysis of results/hotpot_gate.csv.

Separated from the run so the expensive pass is never repeated to answer a new
question, and so the budget-equality check can be audited independently of the
script that produced the data.

    python experiments/analyse_hotpot_gate.py

The check that matters most here is EQUAL BUDGET. run_hotpot_gate.py equalises
the fraction of questions that expand AT ALL, by calibrating the scalar
threshold to the I-gate's rate on the base stage. It does not automatically
follow that the TOTAL number of expansions matches, because each gated
condition re-tests its own trigger after expanding and the scalar threshold was
calibrated on the base-stage distribution. If mean expansions per question
differ, the budgets are not equal and the comparison is void. This script
reports both numbers so that cannot pass unnoticed.
"""
from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CSV = REPO / "results" / "hotpot_gate.csv"


def wald(p: float, n: int):
    se = math.sqrt(p * (1 - p) / n) if n else 0.0
    return se, max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def mcnemar(a_ok, b_ok):
    b = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    c = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    m = b + c
    if m == 0:
        return b, c, 1.0
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(min(b, c) + 1)) / (2 ** m))
    return b, c, p


def main() -> None:
    if not CSV.exists():
        print(f"missing {CSV}; run experiments/run_hotpot_gate.py first")
        return
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    by_cond = defaultdict(dict)
    for r in rows:
        by_cond[r["condition"]][r["qid"]] = r
    conds = sorted(by_cond)
    qids = sorted(by_cond[conds[0]])
    n = len(qids)
    rec = [q for q in qids if by_cond[conds[0]][q]["in_recoverable"] == "True"]
    suf = [q for q in qids if q not in set(rec)]

    print("=" * 78)
    print("PHASE 3 ANALYSIS")
    print("=" * 78)
    print(f"n = {n}   recoverable-miss = {len(rec)}   already-sufficient = {len(suf)}")

    # ---- the budget-equality audit --------------------------------------
    print("\n" + "=" * 78)
    print("EQUAL-BUDGET AUDIT  (the comparison is void if these differ)")
    print("=" * 78)
    print(f"{'condition':24s} {'any-exp rate':>13s} {'mean exps':>11s} "
          f"{'total exps':>11s} {'mean final k':>13s}")
    for c in conds:
        used = [int(by_cond[c][q]["expansions_used"]) for q in qids]
        ks = [int(by_cond[c][q]["final_k"]) for q in qids]
        print(f"{c:24s} {statistics.mean(1 if u else 0 for u in used):>13.4f} "
              f"{statistics.mean(used):>11.4f} {sum(used):>11d} "
              f"{statistics.mean(ks):>13.2f}")

    # ---- headline --------------------------------------------------------
    print("\n" + "=" * 78)
    print("OUTCOMES")
    print("=" * 78)
    print(f"{'condition':24s} {'EM':>8s} {'95% CI':>16s} {'F1':>8s} "
          f"{'EM(recov)':>10s} {'recov':>8s} {'unnec':>8s}")
    stats = {}
    for c in conds:
        em = statistics.mean(int(by_cond[c][q]["em"]) for q in qids)
        f1 = statistics.mean(float(by_cond[c][q]["f1"]) for q in qids)
        emr = statistics.mean(int(by_cond[c][q]["em"]) for q in rec) if rec else 0.0
        gic = statistics.mean(int(by_cond[c][q]["gold_in_context"]) for q in rec) if rec else 0.0
        un = statistics.mean(1 if int(by_cond[c][q]["expansions_used"]) else 0
                             for q in suf) if suf else 0.0
        _se, lo, hi = wald(em, n)
        stats[c] = dict(em=em, f1=f1, emr=emr, gic=gic, un=un)
        print(f"{c:24s} {em:>8.4f} {f'[{lo:.3f}, {hi:.3f}]':>16s} {f1:>8.4f} "
              f"{emr:>10.4f} {gic:>8.4f} {un:>8.4f}")

    # ---- the gate --------------------------------------------------------
    print("\n" + "=" * 78)
    print("McNEMAR vs c_i_gate")
    print("=" * 78)
    ok = {c: [int(by_cond[c][q]["em"]) == 1 for q in qids] for c in conds}
    okr = {c: [int(by_cond[c][q]["em"]) == 1 for q in rec] for c in conds}
    for c in conds:
        if c == "c_i_gate":
            continue
        b, cc, p = mcnemar(ok["c_i_gate"], ok[c])
        br, cr, pr = mcnemar(okr["c_i_gate"], okr[c])
        print(f"  vs {c:24s} whole: b={b:3d} c={cc:3d} p={p:.4f}   "
              f"recoverable: b={br:3d} c={cr:3d} p={pr:.4f}")

    # ---- by type ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("BY QUESTION TYPE")
    print("=" * 78)
    types = sorted({by_cond[conds[0]][q]["qtype"] for q in qids})
    print(f"{'condition':24s} " + "  ".join(f"{t:>20s}" for t in types))
    for c in conds:
        cells = []
        for t in types:
            qs = [q for q in qids if by_cond[conds[0]][q]["qtype"] == t]
            em = statistics.mean(int(by_cond[c][q]["em"]) for q in qs)
            _se, lo, hi = wald(em, len(qs))
            cells.append(f"{em:.3f} [{lo:.2f},{hi:.2f}]")
        print(f"{c:24s} " + "  ".join(f"{x:>20s}" for x in cells))
    for t in types:
        qs = [q for q in qids if by_cond[conds[0]][q]["qtype"] == t]
        print(f"    n({t}) = {len(qs)}")

    # ---- verdict ---------------------------------------------------------
    prim = "b_scalar_max_enc"
    if prim in stats and "c_i_gate" in stats:
        b, cc, p = mcnemar(ok["c_i_gate"], ok[prim])
        print("\n" + "=" * 78)
        print("PRE-REGISTERED VERDICT")
        print("=" * 78)
        d = stats["c_i_gate"]["em"] - stats[prim]["em"]
        print(f"  (c) I-gate EM {stats['c_i_gate']['em']:.4f}  vs  "
              f"(b) scalar max(e,n,c) EM {stats[prim]['em']:.4f}   "
              f"difference {d:+.4f}")
        print(f"  McNemar b={b} c={cc} p={p:.4f}")
        print(f"  Did (c) beat (b)? {'YES' if d > 0 else 'NO'}"
              f"{' significantly' if d > 0 and p < 0.05 else ''}")
        if d <= 0:
            print("  -> STOP CONDITION FIRED (pre-registered). Do not proceed to")
            print("     Phases 4-5 and do not adjust thresholds.")


if __name__ == "__main__":
    main()
