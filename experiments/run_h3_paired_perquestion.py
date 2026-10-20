"""Addendum 6 conditional secondary: the SAME 400 qids under per-question retrieval.

Runs only if the registered primary was positive. See PREREGISTRATION.md,
"Addendum 6, conditional secondary", which fixed that condition before the
primary was read.

Purpose. The primary's references (0.5483 Phase 3, 0.4689 H1) come from
different SAMPLES as well as a different retrieval setting, so comparing against
them confounds the retrieval change with sample variation. This re-runs the
identical 400 questions under per-question retrieval so the comparison becomes
paired: same questions, same retriever, same reader, same r formula, same base
depth, and the single difference is whether retrieval draws from the pooled
19,294-passage index or from the question's own 10 paragraphs.

Reader ONLY -- no NLI backend is constructed, no gate condition is evaluated and
no expansion happens. Records legacy r and em per question.

    python experiments/run_h3_paired_perquestion.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from neutrorag.hotpot import (  # noqa: E402
    DIVERSITY_TARGET, SUPPORT_TARGET, exact_match, load_hotpot_hf,
    stratified_sample, token_f1,
)
from neutrorag.reader import ExtractiveReader  # noqa: E402
from neutrorag.retrieval import DenseRetriever, Document  # noqa: E402
from neutrorag.sufficiency import retrieval_sufficiency  # noqa: E402

RES = REPO / "results"
H3 = RES / "hotpot_gate_h3.csv"
META = RES / "hotpot_gate_h3_meta.json"
OUT = RES / "h3_paired_perquestion.csv"


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print("ADDENDUM 6 conditional secondary -- same 400 qids, PER-QUESTION retrieval")
    print("=" * 78)
    if not H3.exists():
        raise SystemExit(f"ABORT: {H3} missing; the pooled run must exist first")
    meta = json.loads(META.read_text(encoding="utf-8"))
    want = {r["qid"] for r in csv.DictReader(open(H3, encoding="utf-8"))}
    kb = int(meta.get("k_base", 3))
    print(f"commit {commit()}   qids from the pooled run: {len(want)}   k={kb}")

    # Reconstruct the SAME sample deterministically, then assert identity.
    allex = load_hotpot_hf()
    superset = stratified_sample(allex, int(meta["pool_n"]),
                                 seed=int(meta["pool_seed"]))
    ex_by_id = {e.qid: e for e in superset}
    missing = want - set(ex_by_id)
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} qid(s) not in the superset")
    sample = [ex_by_id[q] for q in sorted(want)]
    print(f"matched all {len(sample)} qids   ASSERT identical sample: OK")

    reader = ExtractiveReader()
    reader.read("warm?", ["warm up the loader."])
    print(f"reader ready ({time.time() - t0:.0f}s)\n")

    rows = []
    for i, ex in enumerate(sample):
        # per-question index: the 10 paragraphs HotpotQA supplies
        r = DenseRetriever()
        r.index([Document(p.title, p.text) for p in ex.paragraphs])
        got = r.retrieve(ex.question, k=kb)
        title_of = {p.text: p.title for p in ex.paragraphs}
        titles = [title_of[g.text] for g in got]
        out = reader.read(ex.question, [g.text for g in got])
        hyp = f"{ex.question} {out.answer}".strip()
        bd = retrieval_sufficiency(
            hyp,
            passages=[g.text for g in got],
            relevances=[g.relevance for g in got],
            sources=titles,
            support_target=SUPPORT_TARGET,
            diversity_target=DIVERSITY_TARGET,
        )
        rows.append({
            "qid": ex.qid, "qtype": ex.qtype,
            "r": round(bd.r, 6),
            "max_relevance": round(max((g.relevance for g in got), default=0.0), 6),
            "answer": out.answer, "gold": ex.answer,
            "em": int(exact_match(out.answer, ex.answer)),
            "f1": round(token_f1(out.answer, ex.answer), 6),
            "gold_in_context": int(ex.gold_titles.issubset(set(titles))),
        })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(sample)}  ({time.time() - t0:.0f}s)")

    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}  ({len(rows)} rows)")
    print(f"wall-clock {(time.time() - t0) / 60:.1f} min")
    print("\nNo accuracy is printed here. The paired comparison is computed in")
    print("experiments/analyse_h3_paired.py as a single pass.")


if __name__ == "__main__":
    main()
