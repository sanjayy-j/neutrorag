"""Addendum 6 -- the single pre-registered analysis pass. Run ONCE.

PRIMARY, two-sided, registered before the sample was drawn:

    Under hard retrieval, AUROC of LEGACY r for ANSWER CORRECTNESS (em at base
    depth) is above 0.5.

Reference values under easy retrieval, both legacy r on three passages:
    Phase 3, n=400 : 0.5483  95% CI [0.4840, 0.6127]
    H1,      n=600 : 0.4689  95% CI [0.4073, 0.5305]

LEGACY r, not r_v2, because the references are legacy-r figures and the
comparison must be like-for-like. r_v2's AUROC is computed alongside as a free
secondary and is labelled NOT comparable to those references.

SECONDARY
  - AUROC of legacy r for retrieval success (gold_in_context at base depth);
    reference 0.5958.
  - DeLong against max_relevance for BOTH labels, closing the gap Addendum 4's
    analysis left open, where max_relevance was unavailable for the H1 sample.
  - The five-condition gate comparison: a_fixed_k plus the four gated
    conditions, exact McNemar, I-gate vs max(e,n,c), as in Phase 3.

This reconstructs max_relevance and r_v2 by replaying the SAME pooled index the
run used (19,294 passages, superset seed 20261020), which needs the embedder but
no NLI model and no reader.

    python experiments/analyse_h3_hard_retrieval.py
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
sys.path.insert(0, str(REPO / "experiments"))

RES = REPO / "results"
CSV = RES / "hotpot_gate_h3.csv"
META = RES / "hotpot_gate_h3_meta.json"
BASE = "a_fixed_k"
PRIMARY_GATE = "c_i_gate"
COMPARATOR = "b_scalar_max_enc"

REF_EM = {"Phase 3 (n=400)": 0.5483, "H1 (n=600)": 0.4689}
REF_GOLD = 0.5958


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def mcnemar(a_ok, b_ok):
    b = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    c = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    m = b + c
    if m == 0:
        return b, c, 1.0
    return b, c, min(1.0, 2 * sum(math.comb(m, i)
                                  for i in range(min(b, c) + 1)) / 2 ** m)


def wald(p, n):
    se = math.sqrt(p * (1 - p) / n) if n else 0.0
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def main() -> None:
    if not CSV.exists():
        print(f"missing {CSV}")
        return
    from analyse_rq2_auroc import auc_ci, delong_test  # reuse, do not duplicate

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
    print("ADDENDUM 6 -- does r predict answer correctness once retrieval binds?")
    print("=" * 78)
    print(f"realised n : {n}   seed {meta.get('seed')}   "
          f"ladder k={meta.get('k_base')} -> {meta.get('k_schedule')}")
    print(f"pooled idx : pool_n {meta.get('pool_n')}  "
          f"pool_seed {meta.get('pool_seed')}")
    print(f"analysis at: commit {commit()}")
    for other in ("hotpot_gate.csv", "hotpot_gate_h1.csv", "hotpot_gate_h2.csv"):
        pth = RES / other
        if pth.exists():
            prev = {r["qid"] for r in csv.DictReader(open(pth, encoding="utf-8"))}
            ov = set(qids) & prev
            print(f"disjoint from {other:22} "
                  f"{'CONFIRMED' if not ov else f'VIOLATED ({len(ov)})'}")

    # ---- reconstruct max_relevance and r_v2 on the SAME pooled index -----
    print("\nreplaying the pooled index to recover max_relevance and r_v2 ...")
    from neutrorag.hotpot import load_hotpot_hf, stratified_sample
    from neutrorag.retrieval import DenseRetriever, Document
    from neutrorag.sufficiency import retrieval_sufficiency_v2

    superset = stratified_sample(load_hotpot_hf(), int(meta.get("pool_n", 2000)),
                                 seed=int(meta.get("pool_seed", 20261020)))
    t2t = {}
    for e in superset:
        for pr in e.paragraphs:
            t2t.setdefault(pr.title, pr.text)
    retr = DenseRetriever()
    retr.index([Document(t, x) for t, x in t2t.items()])
    print(f"  index ready, {len(t2t)} passages")
    ex_by_id = {e.qid: e for e in superset}
    kb = int(meta.get("k_base", 3))

    extra = {}
    for q in qids:
        e = ex_by_id.get(q)
        if e is None:
            continue
        got = retr.retrieve(e.question, k=kb)
        hyp = f"{e.question} {by[BASE][q]['answer']}".strip()
        bd = retrieval_sufficiency_v2(hyp, passages=[g.text for g in got],
                                      relevances=[g.relevance for g in got])
        extra[q] = {"max_relevance": max((g.relevance for g in got), default=0.0),
                    "r_v2": bd.r}
    print(f"  recovered for {len(extra)}/{n} questions")

    keep = [q for q in qids if q in extra]
    data = []
    for q in keep:
        data.append({
            "qid": q,
            "r": float(by[BASE][q]["r"]),
            "r_v2": extra[q]["r_v2"],
            "max_relevance": extra[q]["max_relevance"],
            "em": int(by[BASE][q]["em"]),
            "gold_in_context": int(by[BASE][q]["gold_in_context"]),
        })

    # ============================================================= PRIMARY
    print("\n" + "#" * 78)
    print("# PRIMARY (registered, two-sided): AUROC of LEGACY r for ANSWER")
    print("# CORRECTNESS (em at base depth) is above 0.5")
    print("#" * 78)
    y = [d["em"] for d in data]
    npos = sum(y)
    print(f"\n  label em: {npos} positives of {len(y)} "
          f"({100 * npos / len(y):.1f}%)")
    if not (0 < npos < len(y)):
        print("  DEGENERATE -- one class empty, AUROC undefined")
        return
    auc, se, lo, hi, _m, _nn = auc_ci([d["r"] for d in data], y)
    excludes = lo > 0.5 or hi < 0.5
    print(f"\n  legacy r AUROC = {auc:.4f}   SE {se:.4f}   "
          f"95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  CI excludes 0.5 : {'YES' if excludes else 'NO'}")
    print("\n  references, legacy r on three passages under EASY retrieval:")
    for k, v in REF_EM.items():
        print(f"    {k:18} {v:.4f}")
    verdict = ("POSITIVE" if (auc > 0.5 and excludes) else
               "REVERSED" if (auc < 0.5 and excludes) else "NULL")
    print(f"\n  >>> PRIMARY IS {verdict}")
    if verdict == "NULL":
        print("      AUROC does not differ from chance at the 95% level, so")
        print("      making retrieval the binding constraint did NOT make r")
        print("      predictive of answer correctness.")

    a2, s2, l2, h2, _, _ = auc_ci([d["r_v2"] for d in data], y)
    print(f"\n  [free secondary] r_v2 AUROC = {a2:.4f} 95% CI [{l2:.4f}, {h2:.4f}]")
    print("      NOT COMPARABLE to the references above: those are legacy-r")
    print("      figures and r_v2 is a different construction.")

    # =========================================================== SECONDARY
    print("\n" + "=" * 78)
    print("SECONDARY: retrieval success, and DeLong vs max_relevance")
    print("=" * 78)
    for lab, ref in (("gold_in_context", REF_GOLD), ("em", None)):
        yy = [d[lab] for d in data]
        if not (0 < sum(yy) < len(yy)):
            print(f"\n  label {lab}: DEGENERATE, skipped")
            continue
        print(f"\n  label {lab}: {sum(yy)} positives of {len(yy)}"
              + (f"   reference for legacy r: {ref:.4f}" if ref else ""))
        print(f"    {'predictor':16} {'AUROC':>8} {'95% CI':>20} {'above 0.5?':>11}")
        cache = {}
        for pred in ("r", "r_v2", "max_relevance"):
            s = [d[pred] for d in data]
            cache[pred] = s
            a, _se, l, h, _, _ = auc_ci(s, yy)
            v = "yes" if l > 0.5 else ("INVERTED" if h < 0.5 else "no")
            print(f"    {pred:16} {a:>8.4f} {f'[{l:.4f}, {h:.4f}]':>20} {v:>11}")
        print("    DeLong, two correlated ROC curves on the same cases:")
        for pred in ("r", "r_v2"):
            z, pv = delong_test(cache[pred], cache["max_relevance"], yy)
            a1, *_ = auc_ci(cache[pred], yy)
            a3, *_ = auc_ci(cache["max_relevance"], yy)
            sig = "SIGNIFICANT" if pv == pv and pv < 0.05 else "not significant"
            print(f"      {pred} ({a1:.4f}) vs max_relevance ({a3:.4f}): "
                  f"{a1 - a3:+.4f}, z = {z:.3f}, p = {pv:.4f}  [{sig}]")

    # ================================================= gate comparison
    print("\n" + "=" * 78)
    print("SECONDARY: five-condition gate comparison")
    print("=" * 78)
    em = {c: statistics.mean(int(by[c][q]["em"]) for q in qids) for c in conds}
    f1 = {c: statistics.mean(float(by[c][q]["f1"]) for q in qids) for c in conds}
    ok = {c: [int(by[c][q]["em"]) == 1 for q in qids] for c in conds}
    rate = {c: statistics.mean(1 if int(by[c][q]["expansions_used"]) else 0
                               for q in qids) for c in gated}
    print(f"  {'condition':24} {'EM':>8} {'95% CI':>18} {'F1':>8} {'exp.rate':>9}")
    for c in [BASE] + gated:
        lo_, hi_ = wald(em[c], n)
        print(f"  {c:24} {em[c]:>8.4f} {f'[{lo_:.3f}, {hi_:.3f}]':>18} "
              f"{f1[c]:>8.4f} {rate.get(c, 0.0):>9.4f}")
    b, c, p = mcnemar(ok[PRIMARY_GATE], ok[COMPARATOR])
    print(f"\n  exact McNemar, I-gate vs max(e,n,c): b={b} c={c} p={p:.4f}"
          f"   difference {em[PRIMARY_GATE] - em[COMPARATOR]:+.4f}")
    for other in [x for x in gated if x not in (PRIMARY_GATE, COMPARATOR)] + [BASE]:
        bb, cc, pp = mcnemar(ok[PRIMARY_GATE], ok[other])
        print(f"  I-gate vs {other:22} b={bb:3d} c={cc:3d} p={pp:.4f}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  PRIMARY: legacy r AUROC for em = {auc:.4f} "
          f"95% CI [{lo:.4f}, {hi:.4f}]  -> {verdict}")
    print(f"  references under easy retrieval: 0.5483 (Phase 3), 0.4689 (H1)")


if __name__ == "__main__":
    main()
