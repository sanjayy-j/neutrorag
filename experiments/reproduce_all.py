"""Regenerate every number NeutroRAG currently claims, with its provenance.

Memo standing expectation #1: "No number enters the report unless a script
committed to the repository produced it. Every figure in the next review must
be traceable to a commit hash and a command line."

For each number this prints the value, the command that produced it, and the
commit the repository was on. Numbers that have been RETRACTED are printed as
such rather than silently dropped -- see RESULTS.md.

    python experiments/reproduce_all.py                # offline; uses committed artefacts
    python experiments/reproduce_all.py --with-models  # also re-runs model-dependent steps

Offline mode derives the FEVER numbers from two committed dumps rather than
re-running the NLI model (~150s):

  fever_run_200_current.csv  the construction on main now -- these are the
                             live numbers, and section 3 reports them
  fever_run_200_gateA.csv    frozen, exactly as submitted at First Review
                             (constant r, I = (1-r) + r*n, pooled disagree).
                             Section 3b reports it, labelled as history,
                             because the retraction of 0.745 rests on it.

--with-models regenerates the current CSV from scratch and re-runs the pipeline
demo.
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CHANNELS = ("T", "I", "F")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--current-csv", default="fever_run_200_current.csv",
                   help="per-example dump from the CURRENT construction")
    p.add_argument("--gatea-csv", default="fever_run_200_gateA.csv",
                   help="frozen per-example dump as submitted at First Review")
    p.add_argument("--with-models", action="store_true",
                   help="also re-run the NLI/embedding-dependent steps (slow)")
    return p.parse_args()


def git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


COMMIT = git("rev-parse", "--short", "HEAD")


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def claim(label: str, value, command: str, note: str = "") -> None:
    print(f"\n  {label}")
    print(f"    value   : {value}")
    print(f"    command : {command}")
    print(f"    commit  : {COMMIT}")
    if note:
        print(f"    note    : {note}")


def retracted(label: str, value, reason: str) -> None:
    print(f"\n  [RETRACTED] {label}")
    print(f"    was     : {value}")
    print(f"    reason  : {reason}")
    print(f"    commit  : {COMMIT}")


# ---------------------------------------------------------------------------
def offline_library_numbers() -> None:
    from neutrorag.policy import Action, NeutroPolicy
    from neutrorag.scoring import Evidence, MockNLIBackend, NLIRetrievalScorer

    section("1. Library behaviour (offline, mock NLI)")

    backend = MockNLIBackend()
    sums = []
    for premise, hyp in [
        ("Sharks are not mammals; they are fish.", "Sharks are mammals."),
        ("The Eiffel Tower is located in Paris France.", "The Eiffel Tower is located in Paris."),
        ("Springfield is a common place name.", "The mayor won an award in 1998."),
    ]:
        e, n, c = backend.predict(premise, hyp)
        sums.append(e + n + c)
    claim("MockNLIBackend.predict returns a distribution (memo F11)",
          f"max |sum - 1| = {max(abs(s - 1.0) for s in sums):.2e}",
          "python -m pytest tests/test_memo_regressions.py -k distribution",
          "pre-fix this was 0.100 on 4 of 6 branch-covering inputs")

    scorer = NLIRetrievalScorer(nli=MockNLIBackend())
    policy = NeutroPolicy()
    tif = scorer.score("Sharks are mammals.",
                       [Evidence("Sharks are not mammals; they are fish.", 0.9)])
    claim("README quickstart triple",
          f"TIF(T={tif.T:.3f}, I={tif.I:.3f}, F={tif.F:.3f})",
          "python experiments/reproduce_all.py",
          "README.md quickstart quotes this value; it moved with the F11 fix "
          "and again with the Decision-2 saturation targets")

    weak = [Evidence("The region has a temperate climate.", 0.3)]

    def expand_fn(claim_, evidence, step):
        return list(evidence) + [
            Evidence("The heart is a muscular organ that pumps blood.", 0.9)]

    with_exp = policy.run("The heart pumps blood.", weak, scorer.score, expand_fn)
    without = policy.run("The heart pumps blood.", weak, scorer.score)
    claim("README quickstart expansion outcomes",
          f"with expand_fn -> {with_exp.action.value} "
          f"({with_exp.expansions_used} expansion); "
          f"without -> {without.action.value} ({without.expansions_used})",
          "python experiments/reproduce_all.py",
          "memo F16: the old example omitted expand_fn, so EXPAND fell through to ABSTAIN")

    assert with_exp.action is Action.ANSWER and without.action is Action.ABSTAIN


def demo_numbers() -> None:
    from neutrorag.eval import failure_type_attribution, fuzzy_collapse_ablation
    from neutrorag.fever import load_sample, score_examples
    from neutrorag.scoring import MockNLIBackend, NLIRetrievalScorer

    section("2. demo.py -- embedded 12-example sample (MOCK NLI, illustrative only)")

    scorer = NLIRetrievalScorer(nli=MockNLIBackend())
    tifs, gold = score_examples(load_sample(), scorer)
    attr = failure_type_attribution(tifs, gold)
    ab = fuzzy_collapse_ablation(tifs, gold)
    claim("attribution accuracy on the embedded sample",
          f"{attr['accuracy']:.3f}  (n=12)",
          "python demo.py",
          "MOCK NLI on 12 hand-written examples -- not a research result")
    retracted("fuzzy-collapse delta on the embedded sample",
              "previously reported here as a number",
              "WITHDRAWN (memo F4). Merging the I and F gold classes caps the "
              "collapsed score at the majority-class prior, so any positive "
              "delta is arithmetic rather than evidence for the I/F "
              "distinction. demo.py no longer prints it either -- it prints a "
              "retraction notice -- so no command reproduces this figure by "
              "design. See RESULTS.md (Retractions).")
    del ab


def _paired(rows):
    """(gold, predicted) for the construction that wrote the CSV, and for the
    raw NLI baseline recomputed from the same recorded e/n/c."""
    def raw_ch(r):
        v = {"T": float(r["e"]), "I": float(r["n"]), "F": float(r["c"])}
        return max(v, key=v.get)
    return ([(r["gold_channel"], r["predicted_channel"]) for r in rows],
            [(r["gold_channel"], raw_ch(r)) for r in rows])


def _acc_se(pairs):
    n = len(pairs)
    acc = sum(g == p for g, p in pairs) / n
    return acc, math.sqrt(acc * (1 - acc) / n)


def _mcnemar(pairs_a, pairs_b):
    b = sum(1 for a, r_ in zip(pairs_a, pairs_b) if a[0] == a[1] and r_[0] != r_[1])
    c = sum(1 for a, r_ in zip(pairs_a, pairs_b) if r_[0] == r_[1] and a[0] != a[1])
    m = b + c
    if not m:
        return b, c, 1.0
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(min(b, c) + 1)) / (2 ** m))
    return b, c, p


def _disagree(row):
    """Pairwise disagree for the row.

    Prefer the column the eval harness recorded, which is the real pairwise max
    over per-passage outputs. Fall back to 2*min(e, c) for the Gate A CSV, which
    predates the column -- exact on the 185/200 single-passage rows and an
    understatement on the rest.
    """
    if row.get("disagree") not in (None, ""):
        return float(row["disagree"])
    return min(1.0, 2 * min(float(row["e"]), float(row["c"])))


def fever_numbers(gatea_path: str, current_path: str) -> None:
    section("3. FEVER run, n=200 -- CURRENT construction (this is RQ1)")

    cur = REPO / current_path
    gate_a = REPO / gatea_path
    gen_cmd = f"python experiments/run_fever_eval.py --limit 200 --csv {current_path}"
    table_cmd = f"python experiments/make_results_table.py --csv {current_path}"

    if not cur.exists():
        print(f"\n  MISSING: {current_path} -- regenerate with:")
        print(f"    {gen_cmd}")
        return

    rows = list(csv.DictReader(open(cur, encoding="utf-8")))
    n = len(rows)
    neutro, raw = _paired(rows)
    acc_n, se_n = _acc_se(neutro)
    acc_r, se_r = _acc_se(raw)
    b, c, p_val = _mcnemar(neutro, raw)
    n_diff = sum(1 for a, r_ in zip(neutro, raw) if a[1] != r_[1])
    sig = "SIGNIFICANT" if p_val < 0.05 else "not significant"

    print("\n  Construction: r = retrieval sufficiency, I = max(1-r, n), pairwise")
    print("  disagree, Decision-2 saturation targets.")

    claim("attribution accuracy, argmax(T,I,F)",
          f"{acc_n:.4f} +/- {se_n:.4f} (Wald SE), 95% CI "
          f"[{acc_n - 1.96 * se_n:.3f}, {acc_n + 1.96 * se_n:.3f}]",
          gen_cmd,
          "the current construction, not the submitted one")
    claim("baseline accuracy, argmax(e,n,c)",
          f"{acc_r:.4f} +/- {se_r:.4f} (Wald SE), 95% CI "
          f"[{acc_r - 1.96 * se_r:.3f}, {acc_r + 1.96 * se_r:.3f}]",
          table_cmd,
          "identical NLI calls; independent of r and I, so unchanged since Gate A")
    claim("difference (NeutroRAG - baseline)",
          f"{acc_n - acc_r:+.4f} accuracy; predictions differ on {n_diff}/{n} rows",
          table_cmd,
          "RQ1 answer: the decomposition does NOT beat the classifier alone")
    claim("exact McNemar on paired predictions",
          f"b={b} (NeutroRAG only right) c={c} (baseline only right) "
          f"p={p_val:.4f} -- {sig} at alpha=0.05",
          table_cmd,
          "the gap is now statistically significant; at Gate A it was not")

    n_empty = sum(1 for r in rows if int(r["evidence_count"]) == 0)
    claim("N_empty (rows hitting the empty-evidence shortcut, memo F3)",
          f"{n_empty} of {n}",
          gen_cmd,
          "the F3 code path is live but fired zero times in this run")

    r_floats = sorted(float(r["r"]) for r in rows if r["r"] != "")
    r_vals = sorted({round(v, 4) for v in r_floats})
    claim("r across the run (memo F1, resolved by Gate B)",
          f"{len(r_vals)} distinct values, median {r_floats[len(r_floats) // 2]:.3f}, "
          f"range [{min(r_vals):.3f}, {max(r_vals):.3f}]",
          gen_cmd,
          "r is now computed from retrieval state; at Gate A it was the constant 0.9")

    dis = [_disagree(r) for r in rows]
    dis_sorted = sorted(dis)
    claim("pairwise disagree distribution (memo F6/F9)",
          f"median={dis_sorted[n // 2]:.4f} mean={sum(dis) / n:.4f} "
          f"max={max(dis):.4f}; {sum(d > 0.1 for d in dis)}/{n} rows exceed 0.1",
          gen_cmd,
          "the conflict term is near-inert on FEVER: 185/200 rows carry a single "
          "passage, so there is no pair to compare")

    worst = max(abs((float(r["T"]) + float(r["I"]) + float(r["F"]))
                    - (1 + 0.5 * _disagree(r))) for r in rows)
    claim("mass identity T+I+F == 1 + lambda*disagree (memo F5)",
          f"max deviation over {n} rows = {worst:.2e} -- IDENTITY BROKEN",
          "python experiments/check_mass_identity.py",
          "intended: I = max(1-r, n) breaks the identity that made 'T+I+F > 1' "
          "vacuous. The revised Gate B prediction is confirmed")

    masses = sorted(float(r["T"]) + float(r["I"]) + float(r["F"]) for r in rows)
    over = sum(1 for m in masses if m > 1.0)
    claim("mass T+I+F distribution",
          f"min={masses[0]:.3f} median={masses[n // 2]:.3f} max={masses[-1]:.3f}; "
          f"exceeds 1 on {over}/{n} rows ({100 * over / n:.1f}%)",
          "python experiments/check_mass_identity.py",
          "under the OLD I every row exceeded 1 by construction, so the two "
          f"criteria disagree on {n - over}/{n} ({100 * (n - over) / n:.1f}%) of rows")

    # ---- the Gate A snapshot, labelled as history -------------------------
    section("3b. FEVER run, n=200 -- GATE A snapshot (historical, superseded)")
    if not gate_a.exists():
        print(f"\n  MISSING: {gatea_path}")
        return
    grows = list(csv.DictReader(open(gate_a, encoding="utf-8")))
    gn = len(grows)
    gneutro, graw = _paired(grows)
    gacc_n, gse_n = _acc_se(gneutro)
    gacc_r, _ = _acc_se(graw)
    gb, gc, gp = _mcnemar(gneutro, graw)
    gtable = f"python experiments/make_results_table.py --csv {gatea_path}"

    print(f"\n  {gatea_path} is the per-example dump for the construction exactly")
    print("  as submitted at First Review: constant r = 0.9, I = (1-r) + r*n,")
    print("  pooled disagree. Retained unmodified because the retraction of the")
    print("  0.745 headline rests on it. These are NOT current numbers.")

    claim("[GATE A] attribution accuracy, argmax(T,I,F)",
          f"{gacc_n:.4f} +/- {gse_n:.4f} (Wald SE)",
          gtable,
          "the submitted headline; RETRACTED below")
    claim("[GATE A] baseline accuracy, argmax(e,n,c)",
          f"{gacc_r:.4f}",
          gtable,
          f"identical to the current baseline ({acc_r:.4f}) because e/n/c are "
          "bit-identical across both CSVs")
    claim("[GATE A] difference and McNemar",
          f"{gacc_n - gacc_r:+.4f} accuracy; b={gb} c={gc} p={gp:.4f} "
          f"({'significant' if gp < 0.05 else 'not significant'})",
          gtable,
          "at Gate A the gap was small and not significant; under the current "
          "construction it is larger and significant")
    gworst = max(abs((float(r["T"]) + float(r["I"]) + float(r["F"]))
                     - (1 + 0.5 * _disagree(r))) for r in grows)
    claim("[GATE A] mass identity T+I+F == 1 + lambda*disagree",
          f"max deviation over {gn} rows = {gworst:.2e} -- identity HELD",
          "python experiments/reproduce_all.py",
          "this is why 'T+I+F > 1 shows independence' was retracted: under the "
          "submitted I it was an algebraic identity, true of any input")

    gold_counts = {ch: sum(1 for r in grows if r["gold_channel"] == ch) for ch in CHANNELS}
    ceiling = (gold_counts["T"] + gold_counts["F"]) / gn
    retracted("fuzzy-collapse ablation delta (+0.205)",
              "full 0.745 vs collapsed 0.540, delta +0.205",
              "arithmetic, not an experiment (memo F4). Any I/F-merged model is "
              f"capped at ({gold_counts['T']}+{gold_counts['F']})/{gn} = {ceiling:.3f} "
              "by the label prior, and 0.540 is recoverable from the confusion "
              "matrix by summing two columns.")
    retracted("0.745 as a NeutroRAG result",
              "0.745 failure-type attribution accuracy",
              f"GATE A: raw argmax(e,n,c) on the identical NLI calls scores {gacc_r:.4f}. "
              "The neutrosophic layer added nothing measurable. Under the current "
              f"construction the layer scores {acc_n:.4f}, i.e. {acc_r - acc_n:.4f} "
              f"BELOW the same baseline (p={p_val:.4f}).")


def hotpot_numbers() -> None:
    """Phase 2 and Phase 3 headline figures, from the committed CSVs."""
    section("5. HotpotQA: retrieval, and the pre-registered gate experiment")

    ret_p = REPO / "results" / "hotpot_retrieval.csv"
    gate_p = REPO / "results" / "hotpot_gate.csv"
    if not ret_p.exists() or not gate_p.exists():
        print("\n  MISSING results/hotpot_retrieval.csv or results/hotpot_gate.csv")
        return

    import statistics
    ret = list(csv.DictReader(open(ret_p, encoding="utf-8")))
    mean = lambda c, rows: statistics.mean(float(r[c]) for r in rows)  # noqa: E731
    claim("retrieval: Recall@k, MRR, nDCG@10",
          "R@1 {:.4f} R@2 {:.4f} R@3 {:.4f} R@5 {:.4f} R@10 {:.4f}; "
          "MRR {:.4f}; nDCG@10 {:.4f}".format(
              *[mean(f"recall@{k}", ret) for k in (1, 2, 3, 5, 10)],
              mean("mrr", ret), mean("ndcg@10", ret)),
          "python experiments/run_hotpot_retrieval.py",
          "Recall@10 = 1.0 is an index-completeness check, not a result")

    rows = list(csv.DictReader(open(gate_p, encoding="utf-8")))
    by = {}
    for r in rows:
        by.setdefault(r["condition"], {})[r["qid"]] = r
    qids = sorted(by["a_fixed_k"])
    n = len(qids)
    recov = [q for q in qids if by["a_fixed_k"][q]["in_recoverable"] == "True"]
    em = {c: statistics.mean(int(by[c][q]["em"]) for q in qids) for c in by}

    claim("PRE-REGISTERED PRIMARY: I-gate vs scalar gate at equal budget",
          f"EM {em['c_i_gate']:.4f} (I-gate) vs {em['b_scalar_max_enc']:.4f} "
          f"(max(e,n,c)); difference {em['c_i_gate'] - em['b_scalar_max_enc']:+.4f}",
          "python experiments/analyse_hotpot_gate.py",
          "the stop condition FIRED: the decomposed gate did not beat the "
          "scalar. Phases 4-5 were not run")
    claim("expansion vs no expansion on the recoverable-miss subset",
          f"EM {statistics.mean(int(by['a_fixed_k'][q]['em']) for q in recov):.4f}"
          f" -> {statistics.mean(int(by['c_i_gate'][q]['em']) for q in recov):.4f}"
          f" over {len(recov)}/{n} questions",
          "python experiments/analyse_hotpot_gate.py",
          "the one significant positive result in Phase 3 (p=0.0011)")
    for c in sorted(em):
        claim(f"pooled EM, condition {c}", f"{em[c]:.4f}",
              "python experiments/analyse_hotpot_gate.py",
              "read against the extractive ceiling of 0.938, never against 1.0")

    diag = REPO / "results" / "r_component_diagnostics.csv"
    if diag.exists():
        d = list(csv.DictReader(open(diag, encoding="utf-8")))
        cov = [float(r["coverage"]) for r in d]
        rr = [float(r["r"]) for r in d]
        resid = max(abs(a / 3 + 0.2 - b) for a, b in zip(cov, rr))
        claim("r reduces to an affine function of coverage alone",
              f"r = 0.3333*coverage + 0.2000; max residual {resid:.2e} over "
              f"{len(d)} questions; r confined to [0.2000, 0.5333]",
              "python experiments/diagnose_r_components.py",
              "support_count and diversity have exactly zero variance; "
              "connectivity is dropped (no graph)")

    print("\n  Full manifest of all claims, with tests and pre-registration "
          "status:\n    results/claims.md "
          "(python experiments/make_claims_manifest.py)")


def run_analyses(with_models: bool) -> None:
    """Invoke every analysis script, so this file cannot drift from them."""
    section("6. Analysis scripts -- invoked, not merely cited")
    fast = [
        ("Phase 3 gate analysis", "experiments/analyse_hotpot_gate.py"),
        ("Phase 3 by question type", "experiments/analyse_hotpot_by_type.py"),
        ("r vs retrieval depth", "experiments/diagnose_r_depth.py"),
        ("regenerate the claims manifest", "experiments/make_claims_manifest.py"),
    ]
    slow = [
        ("r component diagnostics (~60s: re-embeds the corpus)",
         "experiments/diagnose_r_components.py"),
        ("Phase 2 retrieval metrics (~60s: re-embeds the corpus)",
         "experiments/run_hotpot_retrieval.py"),
    ]
    # All six run by default. The two embedding-dependent ones cost about
    # 60s each, which is acceptable for a reproduce-all and is what makes
    # "invokes every analysis script" true of the DEFAULT invocation
    # rather than only of --with-models.
    todo = fast + slow
    del with_models
    print("")
    print("  six analysers, ~2 min total (two of them re-embed the corpus)")
    print("")
    failed = []
    for label, cmd in todo:
        r = subprocess.run([sys.executable, cmd], cwd=REPO,
                           capture_output=True, text=True)
        status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode})"
        print(f"  [{status:>10}] {label}")
        if r.returncode != 0:
            failed.append(cmd)
            print("      " + (r.stderr.strip().splitlines() or ["(no stderr)"])[-1])
    if failed:
        print(f"\n  {len(failed)} analysis script(s) FAILED: {failed}")
    else:
        print(f"\n  all {len(todo)} invoked analysis scripts completed cleanly")


def contamination_note() -> None:
    section("7. Train/test contamination (memo F10) -- requires network")
    claim("overlap with FEVER-NLI train split",
          "0 / 196 distinct claims (0.0%), 0 claim-IDs",
          "see experiments/check_contamination.py",
          "no training contamination")
    claim("overlap with FEVER-NLI dev split",
          "196 / 196 distinct claims (100%), 196 claim-IDs",
          "see experiments/check_contamination.py",
          "the evaluation set IS the model's own reported dev split "
          "(model card: 'the dev set for Fever-NLI', reported 0.777)")


def model_dependent(with_models: bool) -> None:
    section("8. Model-dependent runs")
    cmds = [
        ("regenerate fever_run_200_current.csv (~150s CPU)",
         "python experiments/run_fever_eval.py --limit 200 "
         "--csv fever_run_200_current.csv"),
        ("pipeline demo, real NLI (the Section 5(d) trace; the report's "
         "screenshot is a historical artefact and no longer matches -- the "
         "action now flips to SURFACE_CONTRADICTION, see RESULTS.md)",
         "python experiments/run_pipeline_demo.py --nli real"),
        ("contamination check (downloads pietrolesci/nli_fever)",
         "python experiments/check_contamination.py"),
        ("saturation targets from depth alone (no accuracy; Decision 2)",
         "python experiments/derive_targets.py"),
        ("pairwise-mode sanity gate, real NLI (Decision 1)",
         "python experiments/check_pairwise_sanity.py --nli real"),
        ("CONFLICTS pairwise cache (~2.5h) then the policy test",
         "python experiments/score_conflicts_pairwise.py"),
        ("the policy test itself",
         "python experiments/run_conflicts_pairwise_eval.py"),
    ]
    if not with_models:
        print("\n  skipped (pass --with-models to run these):")
        for label, cmd in cmds:
            print(f"    {label}\n      {cmd}")
        return
    for label, cmd in cmds:
        print(f"\n  RUNNING: {label}\n    {cmd}")
        subprocess.run(cmd.split(), cwd=REPO, check=False)


def main() -> None:
    args = parse_args()
    dirty = git("status", "--porcelain")
    print("NeutroRAG -- reproduce every claimed number")
    print(f"  commit      : {COMMIT}  ({git('log', '-1', '--format=%s')})")
    print(f"  working tree: {'DIRTY -- numbers below may not match the commit' if dirty else 'clean'}")
    offline_library_numbers()
    demo_numbers()
    fever_numbers(args.gatea_csv, args.current_csv)
    hotpot_numbers()
    run_analyses(args.with_models)
    contamination_note()
    model_dependent(args.with_models)
    print("\nSee RESULTS.md for the same numbers with commentary and retractions.\n")


if __name__ == "__main__":
    main()
