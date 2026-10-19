# NeutroRAG

A training-free trust layer for retrieval-augmented generation. Retrieved
evidence is scored into three independent channels — **T**ruth,
**I**ndeterminacy, **F**alsity — and a five-way policy acts on them, so that
*"I found no evidence"* and *"I found evidence that disagrees with itself"*
trigger different behaviour instead of collapsing into one confidence score.

**The central hypothesis was pre-registered and refuted.**

A gate conditioned on the decomposed evidence state did **not** beat a gate
conditioned on a single scalar confidence at equal expansion budget. The scalar
that beat it is **`max(e, n, c)` — the NLI head's own maximum class
probability**, which this project had registered *in advance* as the weak
baseline.

| HotpotQA distractor, n=400 | pooled EM | comparison EM | fires less where expansion hurts |
|---|---:|---:|---:|
| **I-gate (decomposed)** | 0.3025 | 0.1139 | **+0.0249** — fires *more* |
| **`max(e,n,c)` (scalar)** | **0.3225** | **0.1392** | **−0.1486** |
| fixed-k, no expansion | 0.2675 | 0.1646 | — |

Exact McNemar on the primary comparison: b=14, c=22, **p=0.2430** — not
significant, and in the wrong direction. The registered condition was that the
decomposed gate must *beat* the scalar. It beat none of three scalars tested.
**Phases 4 and 5 were not run**, as the pre-registration requires.

## Two findings that are positive and significant

1. **Expansion helps on recoverable misses.** On the 222/400 questions where a
   gold paragraph sits outside the top-k, Exact Match rises **0.1486 → 0.2432**,
   exact McNemar b=30, c=9, **p=0.0011**. This reproduces FLARE / CRAG /
   Adaptive-RAG in a training-free setting rather than extending them.
2. **Lexical coverage outpredicts the retriever's own relevance score.** For
   whether retrieval actually succeeded, `coverage` reaches Spearman
   **rho = 0.2232 (p<0.0001)** — and 0.4338 on comparison questions — while the
   retriever's own `max_relevance` reaches **0.0196 (p=0.6956)**, no measurable
   relation at all.

## Why the gate failed: it reads back its own action

Tracing the null produced the project's most substantial result. On this corpus
the retrieval-sufficiency term reduces to

&nbsp;&nbsp;&nbsp;&nbsp;**`r = 0.3333 · coverage + 0.2000`**&nbsp;&nbsp;(max residual **1.11e-16** over 400 questions)

because two of its three live components have **exactly zero variance** —
`support_count` and `diversity` are both pinned at 0.3000 — and `connectivity`
is dropped entirely with no graph. That confines `r` to **[0.2000, 0.5333]**.

Expanding retrieval then moves `r` mechanically: at full depth both count-based
components saturate, so `r = coverage/3 + 0.6667` and **+0.4667 of every
expansion's gain in `r` is arithmetic**. Retrieval depth alone explains
**η² = 0.9144** of the I-gate's change in `r`, and the fraction of questions
with `r > 0.5` goes from **14.8% to 80–81% purely by expanding**. Since
`T = r·e ≤ r` and `I ≥ 1−r`, the gate `I > T` is pushed toward *true* before
expanding and toward *false* after — by arithmetic on the depth the gate itself
chose.

> **Design rule.** A sufficiency signal must not be a monotone function of the
> retrieval depth that the gate reading it controls.

Three further requirements for a successor construction — saturation target
strictly below the candidate pool, `r` kept above 0.5 with real variance, and an
outcome label that still varies at maximum depth — are in
[`PREREGISTRATION.md`](PREREGISTRATION.md) Addendum 3(e). **A construction
satisfying them has not been tested**, so this is a verdict on this construction
at this operating point, not on decomposed gating in general.

## Reproduce everything

```bash
pip install -e ".[nli]" && python experiments/reproduce_all.py
```

That prints every claimed number with the command and commit that produced it,
offline, and lists retracted figures as `[RETRACTED]` rather than dropping them.

```bash
python -m pytest -q                            # 141 passed, 6 skipped, 3 xfailed
python experiments/make_claims_manifest.py     # regenerate results/claims.md
```

## Where to look

| file | what it is |
|---|---|
| [`results/claims.md`](results/claims.md) | **authoritative manifest** — all 46 claims, figures, tests, n, regenerating script, and whether pre-registered. Every figure recomputed from the artefacts, none transcribed. |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | the four research questions, predictions and stop conditions, registered before the data was seen; three dated addenda recording a wiring bug, a structural limit and the outcome |
| [`RESULTS.md`](RESULTS.md) | full narrative, every table citing its script, with withdrawn claims collected at the end |
| [`CITATIONS.md`](CITATIONS.md) | citation audit trail; all 18 post-2023 references verified against publisher records |
| [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) | runbook for a 3–5 minute recording of the diagnosis |

## Read the numbers with these caveats

- **Only 9 of 46 claims are pre-registered.** The by-type split, the component
  diagnostics and the depth analysis were chosen *after* seeing the null. They
  are the most interesting findings here and they are **exploratory**; p-values
  are not corrected for that selection.
- **The reader is extractive**, so Exact Match is capped at **0.938** overall and
  **0.692** on comparison questions — 458 of 7405 HotpotQA validation answers
  are yes/no, which no span reader can produce. Never read an EM here against
  1.0.
- **Scope** is one retriever (`all-MiniLM-L6-v2`), one reader
  (`deepset/roberta-base-squad2`), one corpus (HotpotQA distractor validation),
  n=400, seed 20261019.
- **Earlier FEVER results are negative and retained**, not removed: the
  decomposition scores 0.6650 against a bare NLI classifier's 0.7550 on
  identical model calls (p=0.0000). Three figures from the First Review are
  **retracted** and listed as such in [`RESULTS.md`](RESULTS.md).
- The 200 FEVER claims **are** the NLI model's own reported dev split (196/196
  overlap), so those figures are not held-out.

## Not measured, not claimed

Faithfulness, citation accuracy, AURC, ECE, Brier, the RQ4 cost comparison
against a published LLM gate, and RQ2 proper (AUROC of `r` for answerability).
All were specified for phases that the stop condition prevented. No value for
any of them appears anywhere in this repository.

## Status

| milestone | state |
|---|---|
| core library: TIF, scorer, five-way policy, tests | done |
| real NLI backend, real FEVER data, dense + graph retrieval, generation | done |
| pre-registered HotpotQA gate experiment | done — **hypothesis refuted** |
| Phases 4–5 (answerability AUROC, selective prediction, cost) | **not run** — stop condition fired |
| successor construction meeting the four requirements | not attempted |

## Use of AI assistance

Development of this repository, including code, experiment design review and
documentation, was assisted by an AI coding assistant. All experiments were run
locally by the author; every reported figure is regenerated by a committed
script in `experiments/` and traceable to a commit hash. The author is
responsible for the content and for the decisions recorded in
[`PREREGISTRATION.md`](PREREGISTRATION.md). A full declaration appears in
Section 12 of the project report.

## License

MIT.
