"""Per-question-type breakdown of the Phase 3 gate run. Groupby only, no re-run.

Reads results/hotpot_gate.csv.

The question this answers: Phase 2 showed comparison questions already retrieve
well (Recall@3 = 0.873 against bridge's 0.659), and Phase 3 showed expansion
DEGRADES comparison below fixed-k. A gate that worked would therefore decline to
expand on comparison questions. Does it?

If the I-gate fires at roughly its pooled rate on comparison anyway, that is
direct evidence for the Addendum 2 degeneracy: with r = 0.463 at base depth,
T = r*e <= 0.463 < 1-r = 0.537, so T is structurally unreachable and the gate
cannot represent "this is already well supported, do not expand". It would be
firing because it has no way not to.

    python experiments/analyse_hotpot_by_type.py
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
BASE = "a_fixed_k"          # never expands, so its rows ARE the base stage


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
        print(f"missing {CSV}")
        return
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    by = defaultdict(dict)
    for r in rows:
        by[r["condition"]][r["qid"]] = r
    conds = sorted(by)
    qids = sorted(by[BASE])
    types = sorted({by[BASE][q]["qtype"] for q in qids})
    of_type = {t: [q for q in qids if by[BASE][q]["qtype"] == t] for t in types}

    print("=" * 78)
    print("PHASE 3 BY QUESTION TYPE  (groupby on the committed CSV, no re-run)")
    print("=" * 78)
    for t in types:
        print(f"  n({t}) = {len(of_type[t])}")

    # ---- r at base depth, by type ---------------------------------------
    print("\n" + "-" * 78)
    print("retrieval sufficiency r at BASE depth (k=3), from the fixed-k rows")
    print("-" * 78)
    print(f"  {'type':>12} {'mean r':>8} {'median r':>9} {'min':>7} {'max':>7} "
          f"{'1-r (mean)':>11} {'r<0.5':>7}")
    for t in types:
        rs = [float(by[BASE][q]["r"]) for q in of_type[t]]
        below = sum(1 for v in rs if v < 0.5)
        print(f"  {t:>12} {statistics.mean(rs):>8.4f} {statistics.median(rs):>9.4f} "
              f"{min(rs):>7.4f} {max(rs):>7.4f} {1 - statistics.mean(rs):>11.4f} "
              f"{f'{below}/{len(rs)}':>7}")
    print("\n  T = r*e is bounded above by r. Where mean r < 0.5, T cannot exceed")
    print("  1-r for any NLI output, so 'well supported' is unreachable and the")
    print("  gate has no way to decline to expand. (PREREGISTRATION Addendum 2.)")

    # ---- firing rate and depth, by type ---------------------------------
    print("\n" + "-" * 78)
    print("firing rate and mean final k, by type")
    print("-" * 78)
    head = f"  {'condition':24}"
    for t in types:
        head += f" {t + ' fire':>16} {t + ' k':>12}"
    print(head)
    for c in conds:
        line = f"  {c:24}"
        for t in types:
            qs = of_type[t]
            fire = statistics.mean(
                1 if int(by[c][q]["expansions_used"]) > 0 else 0 for q in qs)
            k = statistics.mean(int(by[c][q]["final_k"]) for q in qs)
            line += f" {fire:>16.4f} {k:>12.2f}"
        print(line)

    # ---- the direct test -------------------------------------------------
    print("\n" + "-" * 78)
    print("THE TEST: does the I-gate fire on comparison as often as on bridge?")
    print("-" * 78)
    if "comparison" in types and "bridge" in types:
        for c in conds:
            if c == BASE:
                continue
            fb = statistics.mean(1 if int(by[c][q]["expansions_used"]) > 0 else 0
                                 for q in of_type["bridge"])
            fc = statistics.mean(1 if int(by[c][q]["expansions_used"]) > 0 else 0
                                 for q in of_type["comparison"])
            print(f"  {c:24} bridge {fb:.4f}   comparison {fc:.4f}   "
                  f"difference {fc - fb:+.4f}")
        print("\n  Phase 2: comparison already retrieves at Recall@3 = 0.873 vs")
        print("  bridge 0.659, and Phase 3 showed expansion DEGRADES comparison")
        print("  below fixed-k. A gate that worked would fire LESS on comparison.")

    # ---- comparison-only McNemar vs fixed-k -----------------------------
    print("\n" + "=" * 78)
    print("COMPARISON QUESTIONS ONLY -- McNemar of each gated condition vs fixed-k")
    print("=" * 78)
    print("  n=79 is underpowered by design; the direction and the b/c counts are")
    print("  the point, not significance.\n")
    if "comparison" in types:
        qs = of_type["comparison"]
        base_ok = [int(by[BASE][q]["em"]) == 1 for q in qs]
        base_em = statistics.mean(int(by[BASE][q]["em"]) for q in qs)
        print(f"  fixed-k EM on comparison = {base_em:.4f} "
              f"({sum(base_ok)}/{len(qs)} correct)\n")
        print(f"  {'condition':24} {'EM':>8} {'b':>4} {'c':>4} {'p':>8}  direction")
        for c in conds:
            if c == BASE:
                continue
            ok = [int(by[c][q]["em"]) == 1 for q in qs]
            em = statistics.mean(int(by[c][q]["em"]) for q in qs)
            # b = gated right where fixed-k wrong; c = fixed-k right where gated wrong
            b, cc, p = mcnemar(ok, base_ok)
            direction = ("gated better" if b > cc else
                         "FIXED-K BETTER" if cc > b else "tied")
            print(f"  {c:24} {em:>8.4f} {b:>4} {cc:>4} {p:>8.4f}  {direction}")
        print("\n  b = gated condition right where fixed-k wrong")
        print("  c = fixed-k right where gated condition wrong")

    # ---- how much of the comparison loss is expansion? ------------------
    print("\n" + "-" * 78)
    print("comparison questions: EM split by whether the condition expanded")
    print("-" * 78)
    if "comparison" in types:
        qs = of_type["comparison"]
        print(f"  {'condition':24} {'expanded':>20} {'not expanded':>20}")
        for c in conds:
            if c == BASE:
                continue
            exp = [q for q in qs if int(by[c][q]["expansions_used"]) > 0]
            non = [q for q in qs if int(by[c][q]["expansions_used"]) == 0]
            e1 = statistics.mean(int(by[c][q]["em"]) for q in exp) if exp else float("nan")
            e0 = statistics.mean(int(by[c][q]["em"]) for q in non) if non else float("nan")
            print(f"  {c:24} {f'{e1:.4f} (n={len(exp)})':>20} "
                  f"{f'{e0:.4f} (n={len(non)})':>20}")


if __name__ == "__main__":
    main()
