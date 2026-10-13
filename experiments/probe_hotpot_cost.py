"""Pre-flight cost probe for the Track B HotpotQA experiment.

Times COMPLETE instances end to end -- retrieval, all three readers, the NLI
trust layer and the policy -- not a sample of passages. A previous probe on
CONFLICTS timed six passages and under-projected the full run by 5x (projected
2.49 h, actual 12.8 h) because the probe was not representative of the unit of
work. The unit of work here is one question processed by all three systems.

    python experiments/probe_hotpot_cost.py --n 5 --k 5

Reports per-stage medians and a projected wall-clock for a given sample size.
Produces no experimental result and no accuracy figure.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

READER_MODEL = "deepset/roberta-base-squad2"
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=5, help="complete instances to time")
    p.add_argument("--k", type=int, default=5, help="top-k passages for the reader")
    p.add_argument("--target-hours", type=float, default=3.0)
    p.add_argument("--unanswerable", action="store_true",
                   help="strip the gold supporting paragraphs, leaving only "
                        "distractors (the Task 3 condition)")
    return p.parse_args()


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        return _Span(self, name)

    def add(self, name, dt):
        self.t.setdefault(name, []).append(dt)

    def med(self, name):
        return statistics.median(self.t.get(name, [0.0]))

    def total_per_instance(self):
        return sum(statistics.median(v) for v in self.t.values())


class _Span:
    def __init__(self, timer, name):
        self.timer, self.name = timer, name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.timer.add(self.name, time.perf_counter() - self.t0)
        return False


def main() -> None:
    args = parse_args()

    print("=" * 74)
    print("Track B HotpotQA -- PRE-FLIGHT COST PROBE (no results produced)")
    print("=" * 74)
    print("condition: " + ("UNANSWERABLE (gold paragraphs removed)"
                           if args.unanswerable else "answerable (full corpus)"))

    t0 = time.perf_counter()
    from datasets import load_dataset

    from neutrorag.policy import NeutroPolicy, PolicyConfig
    from neutrorag.reader import ExtractiveReader
    from neutrorag.retrieval import DenseRetriever, Document
    from neutrorag.scoring import Evidence, HFNLIBackend, NLIRetrievalScorer

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    print(f"dataset ready               : {time.perf_counter() - t0:.1f}s  (n={len(ds)})")

    t0 = time.perf_counter()
    reader = ExtractiveReader(model=READER_MODEL, device="cpu")
    reader.read("warm up?", ["warm up the lazy loader."])
    print(f"reader loaded               : {time.perf_counter() - t0:.1f}s  {READER_MODEL}")

    t0 = time.perf_counter()
    nli = HFNLIBackend(model=NLI_MODEL, device="cpu")
    nli.predict("warm up the lazy loader.", "warm up.")
    print(f"NLI loaded + warmed         : {time.perf_counter() - t0:.1f}s  {NLI_MODEL}")

    t0 = time.perf_counter()
    retr = DenseRetriever()
    retr.index([Document("w", "warm up the embedding model.")])
    retr.retrieve("warm", k=1)
    print(f"embedder loaded + warmed    : {time.perf_counter() - t0:.1f}s")

    scorer = NLIRetrievalScorer(nli=nli)
    policy = NeutroPolicy(PolicyConfig(max_expansions=2))
    T = Timer()
    n_nli_calls, n_reads, tok_lens = [], [], []

    print(f"\ntiming {args.n} complete instances at k={args.k} ...\n")

    for i in range(args.n):
        row = ds[i]
        q = row["question"]
        paras = [(t, " ".join(s)) for t, s in
                 zip(row["context"]["title"], row["context"]["sentences"])]
        if args.unanswerable:
            gold_titles = set(row["supporting_facts"]["title"])
            paras = [(t, b) for t, b in paras if t not in gold_titles]

        # ---- retrieval: index this question's 10 paragraphs, then query -----
        with T("1_index_10_paras"):
            r = DenseRetriever()
            r.index([Document(t, f"{t}. {body}") for t, body in paras])
        with T("2_retrieve"):
            got = r.retrieve(q, k=args.k)

        ctx = " ".join(e.text for e in got)
        tok_lens.append(len(reader._load()[0](ctx)["input_ids"]))

        # ---- (i) closed book -------------------------------------------------
        with T("3_read_closedbook"):
            reader.read(q, [])

        # ---- (ii) vector RAG -------------------------------------------------
        with T("4_read_vectorrag"):
            out = reader.read(q, [e.text for e in got])

        # ---- (iii) NeutroRAG: score the answer against the context, then gate
        # Hypothesis = question + predicted answer. THIS IS A FRAMING DECISION
        # and is flagged in the report, not settled here; it is timed because
        # its cost is what the projection needs.
        hyp = f"{q} {out.answer}".strip()
        ev = [Evidence(e.text, e.relevance, source=e.source) for e in got]
        calls = {"n": 0}
        inner = nli.predict

        def counting(premise, hypothesis):
            calls["n"] += 1
            return inner(premise, hypothesis)

        scorer.nli = type("C", (), {"predict": staticmethod(counting)})()
        reads = {"n": 0}

        def expand_fn(_c, evidence, step):
            wider = r.retrieve(q, k=min(len(paras), args.k + 3 * (step + 1)))
            reads["n"] += 1
            reader.read(q, [e.text for e in wider])
            return [Evidence(e.text, e.relevance, source=e.source) for e in wider]

        with T("5_neutrorag_nli_and_policy"):
            policy.run(hyp, ev, scorer.score, expand_fn)

        scorer.nli = nli
        n_nli_calls.append(calls["n"])
        n_reads.append(reads["n"])
        print(f"  instance {i}: nli_calls={calls['n']:3d}  extra_reads={reads['n']}  "
              f"ctx_tokens={tok_lens[-1]:4d}  total={sum(T.t[k][-1] for k in T.t):.2f}s")

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 74)
    print("PER-STAGE MEDIAN, one complete instance")
    print("=" * 74)
    for name in sorted(T.t):
        vals = T.t[name]
        print(f"  {name:28s} median {statistics.median(vals):6.3f}s   "
              f"min {min(vals):6.3f}  max {max(vals):6.3f}")

    wall = [sum(T.t[k][i] for k in T.t) for i in range(args.n)]
    per = statistics.mean(wall)          # <- THE projection statistic
    med = statistics.median(wall)
    n_exp = sum(1 for v in n_reads if v > 0)
    print(f"\n  per-instance wall-clock : mean {per:.2f}s   median {med:.2f}s   "
          f"min {min(wall):.2f}s   max {max(wall):.2f}s")
    print(f"  mean / median            : {per / med:.2f}x")
    print(f"  instances that expanded  : {n_exp}/{args.n} "
          f"({100 * n_exp / args.n:.0f}%)")
    print("\n  PROJECTING FROM THE MEAN. The expansion loop is a long tail: a")
    print("  non-expanding instance costs ~{:.0f}s and an expanding one ~{:.0f}s, so".format(
        med, max(wall)))
    print("  the median under-projects total wall-clock. This is the error that")
    print("  turned a 2.49 h CONFLICTS projection into a 12.8 h run.")
    print(f"  NLI calls per instance               : median "
          f"{statistics.median(n_nli_calls):.0f}  max {max(n_nli_calls)}")
    print(f"  extra reader passes (expansions)     : median "
          f"{statistics.median(n_reads):.0f}  max {max(n_reads)}")
    print(f"  context tokens at k={args.k}                : median "
          f"{statistics.median(tok_lens):.0f}  max {max(tok_lens)}  (reader limit 512)")

    print("\n" + "=" * 74)
    print("PROJECTION")
    print("=" * 74)
    budget = args.target_hours * 3600
    print(f"  {'sample':>8} {'projected':>12}   fits {args.target_hours:g} h?")
    for n in (200, 300, 400, 500, 750, 1000, 2000, len(ds)):
        secs = per * n
        print(f"  {n:>8} {secs / 3600:>10.2f} h   {'yes' if secs <= budget else 'NO'}")
    print(f"\n  largest sample inside {args.target_hours:g} h: "
          f"{int(budget / per)} instances  (mean {per:.2f}s each)")
    print(f"  the median would wrongly suggest {int(budget / med)} instances")
    print("\n  Unaccounted for in the above: the unanswerable condition (Task 3)")
    print("  re-runs every system on a modified corpus for half the sample, and")
    print("  Task 6 sweeps k over three values. Both multiply this figure.")


if __name__ == "__main__":
    main()
