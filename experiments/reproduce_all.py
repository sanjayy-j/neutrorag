"""Regenerate every number NeutroRAG currently claims, with its provenance.

Memo standing expectation #1: "No number enters the report unless a script
committed to the repository produced it. Every figure in the next review must
be traceable to a commit hash and a command line."

For each number this prints the value, the command that produced it, and the
commit the repository was on. Numbers that have been RETRACTED are printed as
such rather than silently dropped -- see RESULTS.md.

    python experiments/reproduce_all.py                # offline; uses committed artefacts
    python experiments/reproduce_all.py --with-models  # also re-runs model-dependent steps

Offline mode derives every FEVER number from the committed fever_run_200.csv
rather than re-running the NLI model (~150s). --with-models regenerates that
CSV from scratch and re-runs the pipeline demo.
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
    p.add_argument("--csv", default="fever_run_200.csv")
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
          "changed by the F11 fix; README.md quickstart quotes this value")

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
    claim("fuzzy-collapse delta on the embedded sample",
          f"{ab.delta:+.3f}",
          "python demo.py",
          "same caveat; see RESULTS.md for why the delta is not evidence of anything")


def fever_numbers(csv_path: str) -> None:
    section("3. FEVER run, n=200 (derived from the committed CSV)")

    path = REPO / csv_path
    if not path.exists():
        print(f"\n  MISSING: {csv_path} -- regenerate with:")
        print("    python experiments/run_fever_eval.py --limit 200 --csv fever_run_200.csv")
        return

    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    n = len(rows)
    gen_cmd = "python experiments/run_fever_eval.py --limit 200 --csv fever_run_200.csv"

    def raw_ch(r):
        v = {"T": float(r["e"]), "I": float(r["n"]), "F": float(r["c"])}
        return max(v, key=v.get)

    neutro = [(r["gold_channel"], r["predicted_channel"]) for r in rows]
    raw = [(r["gold_channel"], raw_ch(r)) for r in rows]
    acc_n = sum(g == p for g, p in neutro) / n
    acc_r = sum(g == p for g, p in raw) / n
    se_n = math.sqrt(acc_n * (1 - acc_n) / n)
    se_r = math.sqrt(acc_r * (1 - acc_r) / n)

    b = sum(1 for a, r_ in zip(neutro, raw) if a[0] == a[1] and r_[0] != r_[1])
    c = sum(1 for a, r_ in zip(neutro, raw) if r_[0] == r_[1] and a[0] != a[1])
    m = b + c
    p_val = min(1.0, 2 * sum(math.comb(m, i) for i in range(min(b, c) + 1)) / (2 ** m)) if m else 1.0

    claim("attribution accuracy, argmax(T,I,F)",
          f"{acc_n:.4f} +/- {se_n:.4f} (Wald SE), 95% CI [{acc_n-1.96*se_n:.3f}, {acc_n+1.96*se_n:.3f}]",
          gen_cmd,
          "NOT attributable to the neutrosophic layer -- see the baseline below")
    claim("GATE A baseline accuracy, argmax(e,n,c)",
          f"{acc_r:.4f} +/- {se_r:.4f} (Wald SE), 95% CI [{acc_r-1.96*se_r:.3f}, {acc_r+1.96*se_r:.3f}]",
          "python experiments/make_results_table.py",
          "identical NLI calls; the bare classifier scores HIGHER")
    claim("difference (NeutroRAG - baseline)",
          f"{acc_n - acc_r:+.4f} accuracy; predictions differ on {sum(1 for a, r_ in zip(neutro, raw) if a[1] != r_[1])}/{n} rows",
          "python experiments/make_results_table.py")
    claim("exact McNemar on paired predictions",
          f"b={b} c={c} p={p_val:.4f} -- not significant at alpha=0.05",
          "python experiments/make_results_table.py")

    n_empty = sum(1 for r in rows if int(r["evidence_count"]) == 0)
    claim("N_empty (rows hitting the empty-evidence shortcut, memo F3)",
          f"{n_empty} of {n}",
          gen_cmd,
          "the F3 code path is live but fired zero times in this run")

    claim("distinct values of r across the run (memo F1)",
          sorted({r["r"] for r in rows}),
          gen_cmd,
          "r is a constant, not a signal; Gate B decision pending")

    dis = [min(1.0, 2 * min(float(r["e"]), float(r["c"]))) for r in rows]
    dis_sorted = sorted(dis)
    claim("disagree = 2*min(e,c) distribution (memo F9)",
          f"median={dis_sorted[len(dis)//2]:.4f} mean={sum(dis)/len(dis):.4f} "
          f"max={max(dis):.4f}; {sum(d > 0.1 for d in dis)}/{n} rows exceed 0.1",
          gen_cmd,
          "the conflict term is near-inert on FEVER; 185/200 rows have a single passage")

    worst = max(abs((float(r["T"]) + float(r["I"]) + float(r["F"]))
                    - (1 + 0.5 * min(1.0, 2 * min(float(r["e"]), float(r["c"])))))
                for r in rows)
    claim("mass identity T+I+F == 1 + lambda*disagree (memo F5)",
          f"max deviation over {n} rows = {worst:.2e}",
          gen_cmd,
          "holds to machine precision on real data, not just random draws")

    gold_counts = {ch: sum(1 for r in rows if r["gold_channel"] == ch) for ch in CHANNELS}
    ceiling = (gold_counts["T"] + gold_counts["F"]) / n
    retracted("fuzzy-collapse ablation delta (+0.205)",
              "full 0.745 vs collapsed 0.540, delta +0.205",
              f"arithmetic, not an experiment (memo F4). Any I/F-merged model is capped at "
              f"({gold_counts['T']}+{gold_counts['F']})/{n} = {ceiling:.3f} by the label prior, "
              f"and 0.540 is recoverable from the confusion matrix by summing two columns.")
    retracted("0.745 as a NeutroRAG result",
              "0.745 failure-type attribution accuracy",
              "GATE A: raw argmax(e,n,c) on the identical NLI calls scores "
              f"{acc_r:.4f}. The neutrosophic layer adds nothing measurable.")


def contamination_note() -> None:
    section("4. Train/test contamination (memo F10) -- requires network")
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
    section("5. Model-dependent runs")
    cmds = [
        ("regenerate fever_run_200.csv (~150s CPU)",
         "python experiments/run_fever_eval.py --limit 200 --csv fever_run_200.csv"),
        ("pipeline demo, real NLI (reproduces the report's Section 5(d) screenshot)",
         "python experiments/run_pipeline_demo.py --nli real"),
        ("contamination check (downloads pietrolesci/nli_fever)",
         "python experiments/check_contamination.py"),
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
    fever_numbers(args.csv)
    contamination_note()
    model_dependent(args.with_models)
    print("\nSee RESULTS.md for the same numbers with commentary and retractions.\n")


if __name__ == "__main__":
    main()
