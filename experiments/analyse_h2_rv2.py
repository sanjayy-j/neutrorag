"""Addendum 5 / H2 -- the single pre-registered analysis pass. Run ONCE.

H2, registered before the sample was drawn (PREREGISTRATION.md Addendum 5):

    With T reachable, the I-gate beats the best scalar-confidence gate on
    pooled exact match at equal expansion budget.

Order is fixed by the registration and is not cosmetic: the MANIPULATION CHECKS
come first and are reported separately, because they are VOIDING conditions. If
either fails, the repair did not take and the primary test is **void, not
negative** -- reporting a null from a still-crippled gate as evidence against
the decomposition is the error this registration exists to prevent.

  check (i)    r_v2 > 0.5 on at least 40% of questions at base depth.
               This is the condition that makes T reachable at all, since
               T = r*e <= r and I >= 1-r.
  check (ii)   eta^2 of delta-r_v2 from final_k, computed over ALL questions
               with final_k in {3, 5} and non-expanders contributing exactly 0,
               below 0.30. Defined this way in the addendum because the Phase 3
               form -- over expanders across two expansion depths -- is
               DEGENERATE on a single-rung ladder and would pass vacuously.
               A scale-free statistic that IS comparable to Phase 3 is reported
               alongside: |mean delta-r| / within-depth sd, which was 11.40 for
               the Phase 3 I-gate.

  PRIMARY      pooled EM, c_i_gate vs b_scalar_max_enc, exact McNemar,
               one-sided in the predicted direction, two-sided alongside.
               max_enc is primary because it is the condition that BEAT the
               I-gate in Phase 3 and produced the stop condition.
  SECONDARY    the other two scalars, Holm-corrected across the two.
  MECHANISTIC  the I-gate's firing-rate differential by question type should
               now be NEGATIVE (firing less on comparison, which retrieves at
               Recall@3 = 0.8734). Phase 3 gave +0.0249 with T unreachable.

    python experiments/analyse_h2_rv2.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RES = REPO / "results"
CSV = RES / "hotpot_gate_h2.csv"
META = RES / "hotpot_gate_h2_meta.json"
BASE = "a_fixed_k"
PRIMARY = "c_i_gate"
COMPARATOR = "b_scalar_max_enc"

CHECK_I_THRESHOLD = 0.40
CHECK_II_THRESHOLD = 0.30
PHASE3_SCALEFREE = 11.40
PHASE3_FIRING_DIFFERENTIAL = +0.0249


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def counts(gated_ok, base_ok):
    b = sum(1 for g, f in zip(gated_ok, base_ok) if g and not f)
    c = sum(1 for g, f in zip(gated_ok, base_ok) if f and not g)
    return b, c


def one_sided(wins: int, losses: int) -> float:
    """P(X <= losses) under Binomial(wins+losses, 0.5).

    H2 predicts the I-gate WINS more discordant pairs, so the evidence is
    strong when `losses` is small.
    """
    m = wins + losses
    if m == 0:
        return 1.0
    return min(1.0, sum(math.comb(m, i) for i in range(losses + 1)) / 2 ** m)


def two_sided(b: int, c: int) -> float:
    m = b + c
    if m == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(m, i)
                            for i in range(min(b, c) + 1)) / 2 ** m)


def wald(p: float, n: int):
    se = math.sqrt(p * (1 - p) / n) if n else 0.0
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def holm(pairs):
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    m, out, running = len(pairs), {}, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pairs[i][1]))
        out[pairs[i][0]] = running
    return out


def main() -> None:
    if not CSV.exists():
        print(f"missing {CSV}")
        return
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    by = defaultdict(dict)
    for r in rows:
        by[r["condition"]][r["qid"]] = r
    conds = sorted(by)
    gated = [c for c in conds if c != BASE]
    qids = sorted(by[BASE])
    n = len(qids)
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}

    print("=" * 78)
    print("ADDENDUM 5 / H2 -- pre-registered test of the repaired gate")
    print("=" * 78)
    print(f"realised n : {n}   seed {meta.get('seed')}   "
          f"ladder k={meta.get('k_base')} -> {meta.get('k_schedule')}")
    print(f"r_v2       : {meta.get('r_v2')}   floor "
          f"{meta.get('relevance_floor_v2')}")
    print(f"analysis at: commit {commit()}")

    for other in ("hotpot_gate.csv", "hotpot_gate_h1.csv"):
        pth = RES / other
        if pth.exists():
            prev = {r["qid"] for r in csv.DictReader(open(pth, encoding="utf-8"))}
            ov = set(qids) & prev
            print(f"disjoint from {other:22} "
                  f"{'CONFIRMED' if not ov else f'VIOLATED ({len(ov)})'}")

    rb = {q: float(by[BASE][q]["r"]) for q in qids}

    # =================================================== MANIPULATION CHECKS
    print("\n" + "#" * 78)
    print("# MANIPULATION CHECKS -- voiding conditions, reported FIRST")
    print("#" * 78)

    over = sum(1 for v in rb.values() if v > 0.5)
    frac = over / n
    check_i = frac >= CHECK_I_THRESHOLD
    print(f"\n(i) r_v2 > 0.5 at base depth on {over}/{n} = {frac:.4f}"
          f"   threshold >= {CHECK_I_THRESHOLD:.2f}   "
          f"{'PASS' if check_i else 'FAIL'}")
    print(f"    r_v2 at base depth: min {min(rb.values()):.4f} "
          f"max {max(rb.values()):.4f} mean {statistics.mean(rb.values()):.4f} "
          f"var {statistics.pvariance(list(rb.values())):.6f}")
    print(f"    (Phase 3 legacy r: mean 0.4546, var 0.001981, >0.5 on 14.8%)")

    # delta-r over ALL questions, from the I-gate's own trajectory
    dr_all, dr_exp, by_depth = [], [], defaultdict(list)
    for q in qids:
        d = float(by[PRIMARY][q]["r"]) - rb[q]
        k = int(by[PRIMARY][q]["final_k"])
        dr_all.append(d)
        by_depth[k].append(d)
        if int(by[PRIMARY][q]["expansions_used"]) > 0:
            dr_exp.append(d)
    tot = statistics.pvariance(dr_all) if len(dr_all) > 1 else 0.0
    within, nn = 0.0, 0
    for k, v in by_depth.items():
        if len(v) > 1:
            within += len(v) * statistics.pvariance(v)
            nn += len(v)
    eta2 = 1 - (within / nn) / tot if nn and tot > 0 else float("nan")
    check_ii = eta2 == eta2 and eta2 < CHECK_II_THRESHOLD
    print(f"\n(ii) eta^2 of delta-r_v2 from final_k over ALL questions "
          f"= {eta2:.4f}")
    print(f"     threshold < {CHECK_II_THRESHOLD:.2f}   "
          f"{'PASS' if check_ii else 'FAIL'}")
    for k in sorted(by_depth):
        v = by_depth[k]
        print(f"     final_k={k}: n={len(v):4d} mean delta-r {statistics.mean(v):+.4f} "
              f"var {statistics.pvariance(v) if len(v) > 1 else 0.0:.6f}")
    if dr_exp and len(dr_exp) > 1:
        sf = abs(statistics.mean(dr_exp))
        sd = statistics.pstdev(dr_exp)
        ratio = sf / sd if sd > 0 else float("inf")
        print(f"\n     scale-free statistic, comparable to Phase 3:")
        print(f"       |mean delta-r| / within-depth sd = {ratio:.3f}")
        print(f"       (Phase 3 I-gate: {PHASE3_SCALEFREE:.2f})")

    void = not (check_i and check_ii)
    print("\n" + "#" * 78)
    if void:
        print("# REPAIR DID NOT TAKE -- the primary test below is VOID, not")
        print("# negative. A null from a still-crippled gate is not evidence")
        print("# about the decomposition.")
    else:
        print("# BOTH CHECKS PASS -- the gate can express 'well supported, do")
        print("# not expand', so the primary test is interpretable.")
    print("#" * 78)

    # ================================================================ OUTCOMES
    em = {c: statistics.mean(int(by[c][q]["em"]) for q in qids) for c in conds}
    f1 = {c: statistics.mean(float(by[c][q]["f1"]) for q in qids) for c in conds}
    ok = {c: [int(by[c][q]["em"]) == 1 for q in qids] for c in conds}
    rate = {c: statistics.mean(1 if int(by[c][q]["expansions_used"]) else 0
                               for q in qids) for c in gated}

    print("\n" + "-" * 78)
    print("Exact match, token-F1 and expansion rate")
    print("-" * 78)
    print(f"  {'condition':24} {'EM':>8} {'95% CI':>16} {'F1':>8} {'exp.rate':>9}")
    for c in [BASE] + gated:
        lo, hi = wald(em[c], n)
        print(f"  {c:24} {em[c]:>8.4f} {f'[{lo:.3f}, {hi:.3f}]':>16} "
              f"{f1[c]:>8.4f} {rate.get(c, 0.0):>9.4f}")

    print("\n" + "=" * 78)
    print(f"PRIMARY (pre-registered): {PRIMARY} vs {COMPARATOR}, ONE-SIDED")
    print("=" * 78)
    b, c = counts(ok[PRIMARY], ok[COMPARATOR])
    p1, p2 = one_sided(b, c), two_sided(b, c)
    print(f"  I-gate EM        {em[PRIMARY]:.4f}")
    print(f"  max(e,n,c) EM    {em[COMPARATOR]:.4f}   difference "
          f"{em[PRIMARY] - em[COMPARATOR]:+.4f}")
    print(f"  expansion rates  {rate[PRIMARY]:.4f} vs {rate[COMPARATOR]:.4f}"
          f"   {'MATCHED' if abs(rate[PRIMARY] - rate[COMPARATOR]) <= 0.02 else 'MISMATCH'}")
    print(f"  discordant {b + c}/{n}   b={b} (I-gate only right)  "
          f"c={c} (scalar only right)")
    print(f"  one-sided p = {p1:.4f}      two-sided p = {p2:.4f}")
    beat = em[PRIMARY] > em[COMPARATOR]
    print()
    if void:
        print("  >>> VOID: a manipulation check failed. No inference is drawn.")
    elif beat and p1 < 0.05:
        print("  >>> H2 SUPPORTED: the I-gate beats the scalar, p < 0.05 one-sided.")
    elif beat:
        print(f"  >>> H2 NOT SUPPORTED: direction held but NULL (p = {p1:.4f}).")
    else:
        print("  >>> H2 FALSIFIED: direction REVERSED -- the scalar still wins.")

    print("\n" + "=" * 78)
    print("SECONDARY (pre-registered): the other two scalars, Holm-corrected")
    print("=" * 78)
    sec = [x for x in gated if x not in (PRIMARY, COMPARATOR)]
    raw, cc = [], {}
    for cond in sec:
        bb, c2 = counts(ok[PRIMARY], ok[cond])
        cc[cond] = (bb, c2)
        raw.append((cond, one_sided(bb, c2)))
    adj = holm(raw)
    print(f"  {'vs condition':24} {'its EM':>8} {'b':>4} {'c':>4} "
          f"{'p (1-sided)':>12} {'Holm':>8} {'sig':>5}")
    for cond, p in raw:
        bb, c2 = cc[cond]
        print(f"  {cond:24} {em[cond]:>8.4f} {bb:>4} {c2:>4} {p:>12.4f} "
              f"{adj[cond]:>8.4f} {'yes' if adj[cond] < 0.05 else 'no':>5}")

    print("\n" + "=" * 78)
    print("MECHANISTIC SECONDARY (pre-registered): firing differential by type")
    print("=" * 78)
    qtype = {q: by[BASE][q]["qtype"] for q in qids}
    types = sorted(set(qtype.values()))
    print(f"  {'condition':24} " + " ".join(f"{t:>12}" for t in types)
          + f" {'differential':>13}")
    for cond in gated:
        cells, vals = [], {}
        for t in types:
            qs = [q for q in qids if qtype[q] == t]
            v = statistics.mean(1 if int(by[cond][q]["expansions_used"]) else 0
                                for q in qs) if qs else float("nan")
            vals[t] = v
            cells.append(f"{v:>12.4f}")
        diff = (vals.get("comparison", float("nan"))
                - vals.get("bridge", float("nan")))
        print(f"  {cond:24} " + " ".join(cells) + f" {diff:>+13.4f}")
    for t in types:
        print(f"    n({t}) = {sum(1 for q in qids if qtype[q] == t)}")
    print(f"\n  Phase 3 I-gate differential was {PHASE3_FIRING_DIFFERENTIAL:+.4f} "
          f"with T unreachable.")
    print("  H2 predicts this flips NEGATIVE: firing less on comparison, which")
    print("  retrieves at Recall@3 = 0.8734 and is degraded by expansion.")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  manipulation check (i)  : {'PASS' if check_i else 'FAIL'} "
          f"({frac:.4f} vs >= {CHECK_I_THRESHOLD})")
    print(f"  manipulation check (ii) : {'PASS' if check_ii else 'FAIL'} "
          f"(eta^2 {eta2:.4f} vs < {CHECK_II_THRESHOLD})")
    if void:
        print("  H2: VOID -- the repair did not take.")
    else:
        print(f"  H2: {'SUPPORTED' if (beat and p1 < 0.05) else 'NOT SUPPORTED'}"
              f"   (difference {em[PRIMARY] - em[COMPARATOR]:+.4f}, "
              f"one-sided p = {p1:.4f})")


if __name__ == "__main__":
    main()
