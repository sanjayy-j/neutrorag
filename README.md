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

## Three further registered tests, run after that refutation

Two more pre-registered hypotheses failed, on samples disjoint from the first
and from each other. Both are reported because they are the strongest evidence
against the project's own idea.

- **H1 — the comparison-question finding did not replicate** (Addendum 4,
  n = 600, disjoint). Phase 3's exploratory by-type result said expansion
  degrades comparison questions below fixed `k`. Registered as a directional
  hypothesis and retested, the effect shrank from **−0.0506 to −0.0033**, about
  fifteenfold, one-sided **p = 0.4388**, and the 4-of-4 uniformity that made it
  look real became **1 of 4**. **Withdrawn as a finding.** What survives is only
  that expansion gives no measurable benefit there either.
- **H2 — the repaired construction was falsified** (Addendum 5, n = 400,
  disjoint). Phase 3's gate was crippled: `r` could not exceed 0.5333, so
  "well supported, do not expand" was unrepresentable. `r_v2` fixed that, and
  **both pre-registered manipulation checks confirmed the repair took effect** —
  `r_v2 > 0.5` on **77.75%** of questions at base depth against 14.8% for legacy
  `r`, and the depth-circularity η² down from **0.9144 to 0.1172**. The gate
  still lost: **0.2900 vs 0.3075** for `max(e,n,c)`, direction reversed,
  one-sided **p = 0.9461**. So the failure was not that `r` lacked range — that
  defect was real, it was fixed, and fixing it did not help.
- **H3 — positive** (Addendum 6, n = 400, pooled index, disjoint). See finding 3
  above. This is the one registered test of the central idea that succeeded.

**Three of ten registered tests are positive.** Ten registered, **nine
executed** (RQ4 was never run), three positive: RQ1's dilution invariance,
expansion on recoverable misses, and Addendum 6. The full inventory, with the
definition of "positive", is in
[`submission/TEST_INVENTORY.md`](submission/TEST_INVENTORY.md).

## Three findings that are positive and significant

1. **Expansion helps on recoverable misses.** On the 222/400 questions where a
   gold paragraph sits outside the top-k, Exact Match rises **0.1486 → 0.2432**,
   exact McNemar b=30, c=9, **p=0.0011**. This reproduces FLARE / CRAG /
   Adaptive-RAG in a training-free setting rather than extending them.
2. **Lexical coverage outpredicts the retriever's own relevance score.** For
   whether retrieval actually succeeded, `coverage` reaches Spearman
   **rho = 0.2232 (p<0.0001)** — and 0.4338 on comparison questions — while the
   retriever's own `max_relevance` reaches **0.0196 (p=0.6956)**, no measurable
   relation at all.
3. **`r` predicts answer correctness once retrieval is the binding constraint**
   — the one registered test of the central idea that came out positive. Under a
   pooled 19,294-passage index, AUROC of `r` for Exact Match is **0.6032**, 95%
   CI **[0.5331, 0.6733]**, which excludes 0.5 (Addendum 6, n=400). Under the
   easy per-question retrieval of the earlier phases it does not: 0.5483 and
   0.4689. Its secondary is the project's strongest single comparison — for
   *retrieval success* `r` reaches AUROC **0.7195** [0.6671, 0.7719] against the
   retriever's own score at 0.5624, a DeLong margin of **+0.1570, z = 4.698,
   p < 0.0001**.

**RQ2, as registered in the proposal, was also computed.** AUROC of `r` for
retrieval success is **0.5958** [0.5404, 0.6511] against `max_relevance` at
**0.5004**, DeLong **z = 2.435, p = 0.0149** — so `r` significantly beats the
retriever's own relevance score, while the retriever's score is itself
indistinguishable from chance at predicting whether its own retrieval worked.
The effect is modest and the earlier Spearman figure above is the same finding
in a weaker metric.

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
| [`results/claims.md`](results/claims.md) | **authoritative manifest** — all 66 claims, figures, tests, n, regenerating script, and whether pre-registered. Every figure recomputed from the artefacts, none transcribed. |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | the **four** research questions (RQ1–RQ4) and **three** further registered hypotheses (H1, H2, and Addendum 6's primary), with their predictions and stop conditions, all recorded before the data was seen; **Addenda 1–6**, dated, covering a wiring bug, a structural limit, the Phase 3 outcome, and the three later tests |
| [`RESULTS.md`](RESULTS.md) | full narrative, every table citing its script, with withdrawn claims collected at the end |
| [`CITATIONS.md`](CITATIONS.md) | citation audit trail; all 18 post-2023 references verified against publisher records |
| [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) | runbook for a 3–5 minute recording of the diagnosis |

## Read the numbers with these caveats

- **Only 29 of 66 claims are pre-registered.** The by-type split, the component
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

Faithfulness, citation accuracy, AURC, ECE, Brier, and the RQ4 cost
comparison against a published LLM gate. All were specified for phases that the
stop condition prevented. No value for any of them appears anywhere in this
repository.

## Status

| milestone | state |
|---|---|
| core library: TIF, scorer, five-way policy, tests | done |
| real NLI backend, real FEVER data, dense + graph retrieval, generation | done |
| pre-registered HotpotQA gate experiment | done — **hypothesis refuted** |
| Phase 4 — answerability AUROC (RQ2) | **run later**, after the stop condition: AUROC 0.5958 vs 0.5004, DeLong p = 0.0149 |
| Phase 5 — selective prediction (AURC, ECE, Brier), faithfulness, citation accuracy | **not run** — stop condition fired |
| RQ4 — cost comparison against a published LLM gate | **not run** — stop condition fired |
| successor construction meeting the four requirements | **built and tested, then falsified** — `r_v2` (Addendum 5) passed both manipulation checks and still lost to `max(e,n,c)` |
| hard-retrieval test of `r` (Addendum 6) | done — **positive**, the one registered success |

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
