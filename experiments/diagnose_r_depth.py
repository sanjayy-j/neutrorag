"""Is r counting retrieval DEPTH rather than sufficiency?

Diagnostics only. Nothing is recalibrated and no accuracy figure is recomputed.
No NLI model, no reader and no retriever is loaded: everything comes from the
committed CSVs plus the HotpotQA parse, which is pure text.

The derivation under test, at the committed targets (support=diversity=10):

    k=3   support_count = 3/10 = 0.3,  diversity = 3/10 = 0.3
          => r = coverage/3 + 0.2000
    k=10  retrieves ALL paragraphs of a 10-paragraph instance, so
          support_count = diversity = 1.0 (the 0.3 relevance floor never binds)
          => r = coverage/3 + 0.6667

    => dr = (coverage10 - coverage3)/3 + 0.4667

If that holds, 0.4667 of every expansion's gain in r is mechanical -- a function
of how many passages were fetched, not of whether any of them helped. The gate
`I > T` is then pushed toward TRUE at base depth and toward FALSE after
expanding, by algebra rather than by evidence.

    python experiments/diagnose_r_depth.py
"""
from __future__ import annotations

import csv
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neutrorag.hotpot import load_hotpot_hf, stratified_sample  # noqa: E402
from neutrorag.sufficiency import coverage as coverage_fn  # noqa: E402

GATE_CSV = REPO / "results" / "hotpot_gate.csv"
DIAG_CSV = REPO / "results" / "r_component_diagnostics.csv"
BASE = "a_fixed_k"
TARGET = 10
SAMPLE_SEED, SAMPLE_N = 20261019, 400


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def spearman(x, y):
    from scipy import stats
    if len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan"), float("nan")
    r = stats.spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def describe(vals):
    return (min(vals), max(vals), statistics.mean(vals),
            statistics.pvariance(vals))


def main() -> None:
    rows = list(csv.DictReader(open(GATE_CSV, encoding="utf-8")))
    by = defaultdict(dict)
    for r in rows:
        by[r["condition"]][r["qid"]] = r
    conds = sorted(by)
    gated = [c for c in conds if c != BASE]
    qids = sorted(by[BASE])
    n = len(qids)
    print("=" * 78)
    print("IS r COUNTING DEPTH?  (committed CSVs only; no model loaded)")
    print("=" * 78)
    print(f"n = {n}   commit {git_commit()}\n")

    # ---------------------------------------------------------------- (4)
    print("-" * 78)
    print("(4) The 0.7775 firing rate: same COUNT, or the same QUESTIONS?")
    print("-" * 78)
    fired = {c: {q for q in qids if int(by[c][q]["expansions_used"]) > 0}
             for c in gated}
    for c in gated:
        print(f"  {c:24} fired on {len(fired[c]):3d}/{n}  "
              f"rate {len(fired[c]) / n:.4f}")
    allsame = len({len(v) for v in fired.values()}) == 1
    print(f"\n  identical COUNT across all four? {allsame}")
    ident = all(fired[c] == fired[gated[0]] for c in gated)
    print(f"  identical QUESTION SET across all four? {ident}")
    inter = set.intersection(*fired.values())
    union = set.union(*fired.values())
    print(f"  fired by ALL four : {len(inter)}   by AT LEAST one : {len(union)}")
    print("\n  The equal count is BY CONSTRUCTION, not a coincidence: the")
    print("  experiment calibrates each scalar threshold to the quantile of its")
    print("  own score distribution that reproduces the I-gate's rate, which")
    print("  forces the counts equal. That is the pre-registered equal-budget")
    print("  design. The SETS are what carries information, and they differ.")

    # ---------------------------------------------------------------- (1)
    print("\n" + "-" * 78)
    print("(1) r at base vs final depth, and the empirical dr")
    print("-" * 78)
    r_base = {q: float(by[BASE][q]["r"]) for q in qids}
    lo, hi, mu, var = describe(list(r_base.values()))
    over = sum(1 for v in r_base.values() if v > 0.5)
    print(f"  {'base (k=3)':26} min {lo:.4f} max {hi:.4f} mean {mu:.4f} "
          f"var {var:.6f}   r>0.5: {over}/{n}")
    print()
    print(f"  {'condition':26} {'min':>7} {'max':>7} {'mean':>7} {'var':>9} "
          f"{'r>0.5':>9} {'mean dr':>9}")
    drs = {}
    for c in gated:
        rf = {q: float(by[c][q]["r"]) for q in qids}
        lo, hi, mu, var = describe(list(rf.values()))
        over = sum(1 for v in rf.values() if v > 0.5)
        d = {q: rf[q] - r_base[q] for q in fired[c]}   # expanders only
        drs[c] = d
        print(f"  {c:26} {lo:>7.4f} {hi:>7.4f} {mu:>7.4f} {var:>9.6f} "
              f"{f'{over}/{n}':>9} {statistics.mean(d.values()):>9.4f}")

    print("\n  dr distribution over questions that expanded:")
    print(f"  {'condition':26} {'n':>5} {'min':>8} {'median':>8} {'max':>8} "
          f"{'var':>9} {'dr<0':>6}")
    for c in gated:
        d = list(drs[c].values())
        print(f"  {c:26} {len(d):>5} {min(d):>8.4f} {statistics.median(d):>8.4f} "
              f"{max(d):>8.4f} {statistics.pvariance(d):>9.6f} "
              f"{sum(1 for v in d if v < 0):>6}")

    # ---------------------------------------------------------------- (3)
    print("\n" + "-" * 78)
    print("(3) Backing out the components at final depth k=10")
    print("-" * 78)
    examples = {e.qid: e for e in
                stratified_sample(load_hotpot_hf(), SAMPLE_N, seed=SAMPLE_SEED)}
    cov3 = {r["qid"]: float(r["coverage"])
            for r in csv.DictReader(open(DIAG_CSV, encoding="utf-8"))}

    # At k=10 a 10-paragraph instance retrieves ALL its paragraphs, so coverage
    # there is computable from the parse alone -- ranking cannot matter.
    k10 = [q for q in qids
           if any(int(by[c][q]["final_k"]) == 10 for c in gated)]
    full = [q for q in k10 if len(examples[q].paragraphs) == 10]
    print(f"  questions reaching k=10 under some condition : {len(k10)}")
    print(f"  of which hold exactly 10 paragraphs          : {len(full)}")
    print("  (for those, k=10 retrieves every paragraph, so coverage at k=10 is")
    print("   determined by the parse and needs no retriever)\n")

    # The experiment scored sufficiency on the HYPOTHESIS, not the question:
    # hyp = question + " " + reader_answer. Coverage must be backed out on the
    # same string or it is a different quantity.
    bad_cov, implied_div = 0, []
    cov10_vals = []
    for q in full:
        ex = examples[q]
        cond = next(c for c in gated if int(by[c][q]["final_k"]) == 10)
        hyp = f"{ex.question} {by[cond][q]['answer']}".strip()
        c10 = coverage_fn(hyp, [p.text for p in ex.paragraphs])
        cov10_vals.append(c10)
        rq = float(by[cond][q]["r"])
        # r = (cov + support + diversity)/3, support = 1.0 (floor never binds)
        implied = 3 * rq - c10 - 1.0
        implied_div.append(implied)
        if not (-1e-6 <= implied <= 1 + 1e-6):
            bad_cov += 1

    lo, hi, mu, var = describe(implied_div)
    sat = sum(1 for v in implied_div if v >= 1 - 1e-6)
    print(f"  implied diversity at k=10: min {lo:.4f} max {hi:.4f} "
          f"mean {mu:.4f} var {var:.8f}")
    print(f"  saturated at 1.0 : {sat}/{len(implied_div)} "
          f"({100 * sat / len(implied_div):.1f}%)")
    worst = max(abs(v - 1.0) for v in implied_div)
    print(f"  max |implied - 1.0| : {worst:.3e}")
    print(f"  deviating by more than 1e-3 : "
          f"{sum(1 for v in implied_div if abs(v - 1.0) > 1e-3)}/{len(implied_div)}")
    print("    The CSV stores r rounded to 6 decimals, so backing a component")
    print("    out of it carries up to 3 * 5e-7 = 1.5e-6 of error. Deviations at")
    print("    that scale are rounding, not evidence against saturation.")
    lo, hi, mu, _ = describe(cov10_vals)
    print(f"\n  coverage at k=10, on the hypothesis : min {lo:.4f} max {hi:.4f} "
          f"mean {mu:.4f}")
    print(f"\n  => mechanical part of dr = 2 * (1.0 - 0.3)/3 = "
          f"{(1.0 - 0.3) / 3 * 2:+.4f}")

    # ---------------------------------------------------------------- (2)
    print("\n" + "-" * 78)
    print("(2) Is dr larger when expansion actually HELPED?")
    print("-" * 78)
    print("  helped  = gold_in_context 0 -> 1     (expansion fetched the gold)")
    print("  failed  = gold_in_context stayed 0   (expansion fetched nothing useful)")
    print("  already = gold_in_context already 1 at base depth\n")
    print("  CONFOUND, stated first: dr is mechanically a function of final_k")
    print("  (+0.2333 at k=6, +0.4667 at k=10, before any coverage gain), and")
    print("  k=10 retrieves EVERY paragraph so gold_in_context is guaranteed")
    print("  there. A raw dr-vs-helped correlation is therefore partly a")
    print("  dr-vs-depth correlation. The stratified table below is the test.\n")
    print(f"  {'condition':24} {'helped':>18} {'failed':>18} {'diff':>8} "
          f"{'rho':>7} {'p':>8}")
    for c in gated:
        helped, failed = [], []
        flags, vals = [], []
        for q in fired[c]:
            g0 = int(by[BASE][q]["gold_in_context"])
            g1 = int(by[c][q]["gold_in_context"])
            if g0 == 1:
                continue
            d = drs[c][q]
            vals.append(d)
            flags.append(g1)
            (helped if g1 == 1 else failed).append(d)
        if not helped or not failed:
            print(f"  {c:24} insufficient data")
            continue
        mh, mf = statistics.mean(helped), statistics.mean(failed)
        rho, p = spearman(vals, flags)
        print(f"  {c:24} {f'{mh:.4f} (n={len(helped)})':>18} "
              f"{f'{mf:.4f} (n={len(failed)})':>18} {mh - mf:>+8.4f} "
              f"{rho:>7.3f} {p:>8.4f}")
    # ---- stratified by final depth: the actual test ------------------
    print("\n" + "-" * 78)
    print("(2b) dr STRATIFIED BY final_k -- is dr a function of depth alone?")
    print("-" * 78)
    print(f"  {'condition':24} {'final_k':>8} {'n':>5} {'mean dr':>9} "
          f"{'var dr':>10} {'gold_in_ctx':>12}")
    for c in gated:
        for k in (6, 10):
            qs = [q for q in fired[c] if int(by[c][q]["final_k"]) == k]
            if not qs:
                continue
            d = [drs[c][q] for q in qs]
            g = statistics.mean(int(by[c][q]["gold_in_context"]) for q in qs)
            print(f"  {c:24} {k:>8} {len(qs):>5} {statistics.mean(d):>9.4f} "
                  f"{statistics.pvariance(d):>10.6f} {g:>12.4f}")
    print("")
    print("  variance of dr explained by final_k alone (eta-squared):")
    for c in gated:
        tot = statistics.pvariance([drs[c][q] for q in fired[c]])
        within, nn = 0.0, 0
        for k in (6, 10):
            qs = [q for q in fired[c] if int(by[c][q]["final_k"]) == k]
            if len(qs) < 2:
                continue
            within += len(qs) * statistics.pvariance([drs[c][q] for q in qs])
            nn += len(qs)
        eta = 1 - (within / nn) / tot if nn and tot else float("nan")
        print(f"    {c:24} total var {tot:.6f}  within {within / nn:.6f}  "
              f"eta^2 = {eta:.4f}")
    print("\n  If mean dr is near-constant WITHIN each final_k and the gap")
    print("  between depths is ~0.2333, then dr is determined by how many")
    print("  passages were fetched. gold_in_ctx at k=10 is 1.0 by construction,")
    print("  because k=10 retrieves every paragraph -- so the earlier")
    print("  dr-vs-helped correlation cannot be separated from depth here.")


if __name__ == "__main__":
    main()
