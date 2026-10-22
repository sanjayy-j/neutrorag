"""Independent verification of the figures the final-review report states.

Recomputes each figure from the committed raw artefacts and compares it against
the value the report asserts. Prints PASS / MISMATCH / UNVERIFIABLE per item and
exits non-zero if anything mismatches, so this can gate a submission.

    python experiments/verify_report_claims.py

The report itself is NOT in this repository -- CITATIONS.md records that it is a
.docx edited outside it -- so the asserted values below are transcribed from the
audit brief and are the thing under test. The repository is the source of truth.
"""
from __future__ import annotations

import csv
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

RES = REPO / "results"
FAILS: list = []
ROWS: list = []


def check(item, asserted, computed, source, ok=None, note=""):
    if ok is None:
        ok = (abs(asserted - computed) < 5e-5
              if isinstance(asserted, float) else asserted == computed)
    status = "PASS" if ok else "MISMATCH"
    if not ok:
        FAILS.append(item)
    ROWS.append((item, asserted, computed, status, source, note))


def unverifiable(item, why, source=""):
    ROWS.append((item, "-", "-", "UNVERIFIABLE", source, why))


def mcnemar(a_ok, b_ok):
    b = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    c = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    m = b + c
    if m == 0:
        return b, c, 1.0
    return b, c, min(1.0, 2 * sum(math.comb(m, i)
                                  for i in range(min(b, c) + 1)) / 2 ** m)


def load_gate(name):
    by = defaultdict(dict)
    for r in csv.DictReader(open(RES / name, encoding="utf-8")):
        by[r["condition"]][r["qid"]] = r
    return by


def main() -> None:
    from analyse_rq2_auroc import auc_ci, delong_test, load_sample

    print("=" * 78)
    print("VERIFICATION OF REPORT-ASSERTED FIGURES against committed artefacts")
    print("=" * 78)
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                     cwd=REPO, text=True).strip()
    print(f"repo commit {commit}\n")

    # ---- A. main FEVER comparison ------------------------------------
    rows = list(csv.DictReader(open(REPO / "fever_run_200_current.csv",
                                    encoding="utf-8")))
    def raw_ch(r):
        v = {"T": float(r["e"]), "I": float(r["n"]), "F": float(r["c"])}
        return max(v, key=v.get)
    neu = [(r["gold_channel"], r["predicted_channel"]) for r in rows]
    raw = [(r["gold_channel"], raw_ch(r)) for r in rows]
    acc_n = sum(g == p for g, p in neu) / len(neu)
    acc_r = sum(g == p for g, p in raw) / len(raw)
    b, c, p = mcnemar([g == p for g, p in neu], [g == p for g, p in raw])
    check("A. FEVER argmax(T,I,F)", 0.6650, round(acc_n, 4),
          "fever_run_200_current.csv")
    check("A. FEVER raw argmax(e,n,c)", 0.7550, round(acc_r, 4),
          "fever_run_200_current.csv")
    check("A. FEVER exact McNemar p", 4.0e-5, p, "recomputed",
          ok=abs(p - 4.0e-5) < 1e-6,
          note=f"b={b} c={c}; exact p={p:.3e}, report rounds to 4.0e-5")
    check("A. FEVER n", 200, len(rows), "row count")

    # ---- B. recoverable-miss subset ----------------------------------
    g3 = load_gate("hotpot_gate.csv")
    q3 = sorted(g3["a_fixed_k"])
    rec = [q for q in q3 if g3["a_fixed_k"][q]["in_recoverable"] == "True"]
    em_fixed = statistics.mean(int(g3["a_fixed_k"][q]["em"]) for q in rec)
    em_igate = statistics.mean(int(g3["c_i_gate"][q]["em"]) for q in rec)
    bb, cc, pp = mcnemar([int(g3["c_i_gate"][q]["em"]) == 1 for q in rec],
                         [int(g3["a_fixed_k"][q]["em"]) == 1 for q in rec])
    check("B. recoverable-miss, fixed-k EM", 0.1486, round(em_fixed, 4),
          "hotpot_gate.csv")
    check("B. recoverable-miss, I-gate EM", 0.2432, round(em_igate, 4),
          "hotpot_gate.csv")
    check("B. recoverable-miss McNemar p", 0.0011, round(pp, 4), "recomputed",
          note=f"b={bb} c={cc}, n={len(rec)}")
    check("B. recoverable-miss n", 222, len(rec), "subset size")

    # ---- C. retrieval-success AUROC ----------------------------------
    p3 = load_sample(RES / "hotpot_gate.csv", RES / "r_component_diagnostics.csv")
    y = [r["gold_in_context"] for r in p3]
    a, _se, lo, hi, _m, _n = auc_ci([r["r"] for r in p3], y)
    a_mr, *_ = auc_ci([r["max_relevance"] for r in p3], y)
    z, pv = delong_test([r["r"] for r in p3],
                        [r["max_relevance"] for r in p3], y)
    check("C. AUROC r for gold_in_context", 0.5958, round(a, 4), "recomputed")
    check("C. AUROC max_relevance, same label", 0.5004, round(a_mr, 4),
          "recomputed")
    check("C. DeLong p, r vs max_relevance", 0.0149, round(pv, 4), "recomputed",
          note=f"z={z:.3f}; modest effect, CI [{lo:.4f}, {hi:.4f}]")

    # ---- D. hard-retrieval primary -----------------------------------
    h3 = load_gate("hotpot_gate_h3.csv")
    qh = sorted(h3["a_fixed_k"])
    yh = [int(h3["a_fixed_k"][q]["em"]) for q in qh]
    rh = [float(h3["a_fixed_k"][q]["r"]) for q in qh]
    ah, _s, loh, hih, _m2, _n2 = auc_ci(rh, yh)
    check("D. hard-retrieval AUROC r for em", 0.6032, round(ah, 4), "recomputed")
    check("D. hard-retrieval CI low", 0.5331, round(loh, 4), "recomputed")
    check("D. hard-retrieval CI high", 0.6733, round(hih, 4), "recomputed")
    check("D. hard-retrieval n", 400, len(qh), "row count")
    check("D. CI excludes 0.5", True, bool(loh > 0.5), "derived")

    # ---- E. construction / gate comparisons --------------------------
    h2 = load_gate("hotpot_gate_h2.csv")
    q2 = sorted(h2["a_fixed_k"])
    r2 = [float(h2["a_fixed_k"][q]["r"]) for q in q2]
    over = sum(1 for v in r2 if v > 0.5) / len(r2)
    em_i = statistics.mean(int(h2["c_i_gate"][q]["em"]) for q in q2)
    em_s = statistics.mean(int(h2["b_scalar_max_enc"][q]["em"]) for q in q2)
    b2, c2, p2_ = mcnemar([int(h2["c_i_gate"][q]["em"]) == 1 for q in q2],
                          [int(h2["b_scalar_max_enc"][q]["em"]) == 1 for q in q2])
    check("E. H2 manipulation check (i), r_v2>0.5", 0.7775, round(over, 4),
          "hotpot_gate_h2.csv")
    check("E. H2 I-gate EM", 0.2900, round(em_i, 4), "hotpot_gate_h2.csv")
    check("E. H2 max(e,n,c) EM", 0.3075, round(em_s, 4), "hotpot_gate_h2.csv")
    check("E. H2 I-gate did NOT beat the scalar", True, bool(em_i < em_s),
          "derived", note=f"two-sided McNemar p={p2_:.4f}, b={b2} c={c2}: "
                          "non-significant and must stay so in prose")

    # ---- F. fourth disjoint sample, paired ---------------------------
    pq = {r["qid"]: r for r in csv.DictReader(
        open(RES / "h3_paired_perquestion.csv", encoding="utf-8"))}
    shared = [q for q in qh if q in pq]
    agree = [q for q in shared
             if int(pq[q]["em"]) == int(h3["a_fixed_k"][q]["em"])]
    ya = [int(h3["a_fixed_k"][q]["em"]) for q in agree]
    zz, pvv = delong_test([float(h3["a_fixed_k"][q]["r"]) for q in agree],
                          [float(pq[q]["r"]) for q in agree], ya)
    check("F. paired DeLong p (label-agreeing subset)", 0.0844, round(pvv, 4),
          "recomputed", note=f"z={zz:.3f}, n={len(agree)} of {len(shared)}")
    # disjointness of all four samples
    sets = {}
    for nm, f in (("Phase3", "hotpot_gate.csv"), ("H1", "hotpot_gate_h1.csv"),
                  ("H2", "hotpot_gate_h2.csv"), ("H3", "hotpot_gate_h3.csv")):
        sets[nm] = {r["qid"] for r in csv.DictReader(open(RES / f, encoding="utf-8"))}
    pairs_ok = True
    overlaps = []
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ov = sets[names[i]] & sets[names[j]]
            if ov:
                pairs_ok = False
                overlaps.append(f"{names[i]}&{names[j]}={len(ov)}")
    check("F. four samples pairwise disjoint", True, pairs_ok, "qid sets",
          note="; ".join(overlaps) if overlaps else
               "sizes " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()))

    # ---- CONFLICTS denominator (Blocker F of the brief) --------------
    from neutrorag.conflicts import CONFLICT_TYPES, TYPE_TO_EXPECTED_ACTION
    cr = list(csv.DictReader(open(RES / "conflicts_pairwise.csv", encoding="utf-8")))
    elig = [r for r in cr if r["conflict_type"] in TYPE_TO_EXPECTED_ACTION]
    okc = sum(1 for r in elig
              if r["action"] == TYPE_TO_EXPECTED_ACTION[r["conflict_type"]])
    mis = sum(1 for r in cr if "misinformation" in r["conflict_type"].lower())
    check("CONFLICTS release total", 458, sum(CONFLICT_TYPES.values()),
          "conflicts.CONFLICT_TYPES")
    check("CONFLICTS scored sample", 100, len(cr), "conflicts_pairwise.csv")
    check("CONFLICTS policy-test denominator", 80, len(elig), "derived",
          note="100 minus Complementary(10) and Outdated(10), which carry NO "
               "expected action; NOT minus the 5 Misinformation instances")
    check("CONFLICTS Misinformation in the scored sample", 0, mis,
          "conflicts_pairwise.csv",
          note="the 5 Misinformation instances were never sampled, so "
               "'100 minus 5' would wrongly imply 95")
    check("CONFLICTS expected-action accuracy", 0.2000,
          round(okc / len(elig), 4), "recomputed", note=f"{okc}/{len(elig)}")

    # ---- manifest + reference counts ---------------------------------
    man = (RES / "claims.md").read_text(encoding="utf-8")
    import re
    m = re.search(r"^(\d+) claims\. \*\*(\d+) pre-registered", man, re.M)
    check("manifest total claims", 66, int(m.group(1)), "results/claims.md")
    check("manifest pre-registered", 29, int(m.group(2)), "results/claims.md")
    cit = (REPO / "CITATIONS.md").read_text(encoding="utf-8")
    checked = int(re.search(r"\| checked \| (\d+) \|", cit).group(1))
    correct = int(re.search(r"\| correct as written \| (\d+) \|", cit).group(1))
    errs = int(re.search(r"\| \*\*carried errors\*\* \| \*\*(\d+)\*\*", cit).group(1))
    check("references checked", 18, checked, "CITATIONS.md")
    check("references reconcile (correct + errors == checked)", True,
          correct + errs == checked, "arithmetic",
          note=f"{correct} + {errs} = {correct + errs}")

    # ---- things the brief asserts that this repo cannot support -------
    unverifiable("RQ5 / RQ6 exist", "PREREGISTRATION.md defines RQ1-RQ4 only; "
                 "no RQ5 or RQ6 appears anywhere in the repository",
                 "PREREGISTRATION.md")
    unverifiable("simulated-cascade findings",
                 "no 'simulated cascade' artefact, script or result exists in "
                 "the repository", "repo-wide grep")
    unverifiable("Table 6.1 / Table 10.1, pages 20 / 49",
                 "the report is a .docx edited outside this repository "
                 "(CITATIONS.md line 3); it has never been committed on any "
                 "branch", "git log --all")

    # ---- report ------------------------------------------------------
    print(f"{'item':52} {'asserted':>10} {'computed':>10}  status")
    print("-" * 95)
    for item, a_, c_, st, src, note in ROWS:
        af = f"{a_:.4g}" if isinstance(a_, float) else str(a_)
        cf = f"{c_:.4g}" if isinstance(c_, float) else str(c_)
        print(f"{item:52} {af:>10} {cf:>10}  {st}")
        if note:
            print(f"{'':52} {'':>10} {'':>10}  -> {note}")
    n_pass = sum(1 for r in ROWS if r[3] == "PASS")
    n_mis = sum(1 for r in ROWS if r[3] == "MISMATCH")
    n_unv = sum(1 for r in ROWS if r[3] == "UNVERIFIABLE")
    print("-" * 95)
    print(f"PASS {n_pass}   MISMATCH {n_mis}   UNVERIFIABLE {n_unv}")
    if FAILS:
        print("\nMISMATCHED ITEMS:")
        for f in FAILS:
            print(f"  - {f}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
