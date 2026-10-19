# NeutroRAG — results and provenance

Every number NeutroRAG claims, with the command that produced it and its status.
Regenerate all of it with:

```bash
python experiments/reproduce_all.py                # offline, from committed artefacts
python experiments/reproduce_all.py --with-models  # also re-runs the model-dependent steps
```

**Standing rule (supervisory memo, expectation #1): no number enters the report
unless a script committed to this repository produced it.** If a number is not in
this file, it does not go in the report.

---

## Status summary

| | |
|---|---|
| **Gate A** | **FAILED** — the raw NLI baseline scores higher than NeutroRAG |
| **Gate B** (definition of `r`) | signed off, revised; `r` implemented AND `I = max(1-r, n)`. **Prediction confirmed — the mass identity is broken** |
| **Decision 1** (pairwise mode) | implemented; sanity gate passed, but **both CONFLICTS stop conditions fired** |
| **Decision 2** (saturation targets) | re-derived from depth alone and committed before any accuracy run (`53d305e`) |
| **Gate C** (expansion loop not a no-op) | test written and **passing** — see below |

> **Step 6 landed** (pairwise `disagree`), and **Step 9 (CONFLICTS) ran and was
> halted** on its own stop conditions — but for a measurement error, not a
> finding about the method. See the Step 9 section.

---

## Gate B (revised): `I = max(1-r, n)` — the mass identity is broken

### The change

`scoring.py:229`: `I = clamp(1.0 - r + r * n)` becomes `I = clamp(max(1.0 - r, n))`.
`T`, `F`, `lambda` and `disagree` are byte-identical; `policy.py` untouched.
Indeterminacy is now the **worse** of two independent doubts — thin retrieval
(`1-r`) or on-topic but uninformative evidence (`n`) — rather than a
multiplicative blend of them.

The first Gate B attempt (redefining `r` alone) could not have worked, and this
is why: `r` multiplied the whole NLI simplex across the three channels and then
cancelled against `I`'s `(1-r)` term, giving `T+I+F = 1 + λ·disagree` for **any**
`r`. Breaking it required changing `I`, which memo F5 route (i) actually asked
for ("break the identity by *deriving I* from a retrieval-sufficiency signal").

### The prediction test — CONFIRMED

```bash
python experiments/check_mass_identity.py
```

| check | max \|deviation\| | before |
|---|---:|---:|
| random draws, `r ~ Uniform(0,1)` | **2.499e-01** | 4.441e-16 |
| random draws, `r` computed by `sufficiency` | **2.496e-01** | 4.441e-16 |
| real run: pipeline corpus + graph | **1.696e-01** | 2.220e-16 |
| real run: 200 FEVER claims | **2.275e-01** | 2.220e-16 |

This matches the analytic bound exactly. Deviation is `-r·n` when `1-r > n` and
`-(1-n)(1-r)` otherwise; both are maximised at `n = 1-r`, where each equals
`r(1-r)` — i.e. **0.25 at r = 0.5**.

### The mass now carries independent information

| corpus | min | median | max | mass > 1 | mass < 1 |
|---|---:|---:|---:|---:|---:|
| pipeline (4 claims) | 0.930 | 1.100 | 1.471 | 75% | 25% |
| FEVER (200 claims) | 0.775 | 0.988 | 1.277 | 24.5% | 75.5% |

Under the old `I`, **all 200** FEVER rows had mass > 1, because "mass > 1" was
identically "disagree > 0". Under the new `I`, 49/200 do, and the two criteria
**disagree on 75.5% of rows**. The mass can now also fall *below* 1, which it
previously never could.

## The r-ablation — computed `r` is much worse, on FEVER

```bash
python experiments/run_r_ablation.py    # -> results/r_ablation.{md,csv}
```

Same data, same replayed NLI outputs, `I` held at `max(1-r, n)`; only `r` varies.

| condition | accuracy | macro-F1 |
|---|---:|---:|
| (a) `r = 0.9` constant | 0.7550 | 0.7571 |
| (b) `r = sufficiency` | **0.4600** | 0.4043 |
| raw `argmax(e,n,c)` [reference] | 0.7550 | 0.7574 |

**(b) − (a) = −0.2950.** Differ on 85/200 rows; exact McNemar b=13, c=72,
**p = 0.0000**. Large and highly significant.

**This measures memo F9, not the sufficiency design.** FEVER gives one gold
passage per claim: 185/200 rows carry exactly one, and with no graph and no
provenance `connectivity` and `diversity` are unavailable. Only `coverage` can
really vary, so `r` is a near-proxy for claim/passage token overlap on this
corpus rather than a measure of retrieval depth.

Under the Decision-2 target (`support_target = 1`, FEVER's own p75 depth)
`support_count` saturates on every row and `r` spans [0.500, 1.000] with a median
of **0.750** and 26 distinct values; **17/200** rows land at `r ≤ 0.5`, the memo
F12 regime where `I = max(1-r, n) ≥ 0.5` automatically. Condition (b) loses on
precisely those low-`r` rows: F-recall falls 0.853 → **0.662** and T-recall
0.631 → **0.554**, while I-recall is flat at 0.776 and I-*precision* falls to
0.500 — it is predicting I more often and being wrong when it does. No parameter
was retuned; the targets were committed before this run.

*(These figures were 0.417 and 143/200 before the Decision-2 calibration. The
earlier reading — that the whole corpus sat inside the F12 regime — no longer
holds, and the current explanation is the low-`r` tail, not the corpus median.)*

## Sensitivity sweep (memo F12)

```bash
python experiments/run_sensitivity_sweep.py   # -> results/sensitivity_sweep.{csv,png}
```

**The F12 boundary did not move.** `I = max(1-r, n) ≥ 1-r` and `T = r·e ≤ r`, so
for `r < 0.5`, `I ≥ 1-r > r ≥ T` — I beats T unconditionally. The old
`I = (1-r) + r·n` also satisfies `I ≥ 1-r`, so the regime begins at `r = 0.5`
under **both** formulas. Empirically the cliff is sharp:

| r | predicted I | action = EXPAND | attribution accuracy |
|---:|---:|---:|---:|
| 0.50 | 97.5% | 72.5% | 0.350 |
| 0.52 | 62.5% | 40.0% | 0.680 |
| 1.00 | 41.0% | 37.0% | 0.765 |

I is the argmax for *every* claim at any `r ≤ 0.42`, and for at least half of
them at any `r ≤ 0.58`.

**The median-`r` figure here moved with the Decision-2 calibration and the
conclusion inverted.** Before calibration FEVER's median `r` was 0.417, which put
the whole corpus inside the degenerate regime. Under the committed targets the
median is **0.750**, the range is [0.500, 1.000], and **0 of 200** rows fall at
`r ≤ 0.42`. The corpus is no longer inside that regime, so the F12 cliff is not
the explanation for the current FEVER result — the mechanism below is.

**Why the computed `r` loses to a constant one.** The sweep pins `r`, so its
rows are directly comparable to the live system, which computes it. Every pinned
value in the observed range beats the computed one: `r = 1.0` scores 0.765,
`r = 0.8` scores 0.745, `r = 0.6` scores 0.735 — while the computed, *varying*
`r` scores **0.665**. It is therefore not the level of `r` that costs accuracy
but its spread. The 20 rows carrying the lowest `r` get an `I` floor of `1-r`
large enough to win the argmax, and the live system pays for them while any
constant `r ≥ 0.6` does not.

That is the whole of the −0.0900 gap, and it is unusually clean:

- The two systems differ on exactly **20 of 200** rows.
- On **all 20**, NeutroRAG predicts **`I`** — not one is a T↔F disagreement.
- The baseline is correct on **19** of those 20 (McNemar c=19, b=1).
- **17 of the 20** are rows with `r ≤ 0.58`.

So the layer's only effect on FEVER is to overwrite a correct, confident NLI
judgement with "not enough info" on the rows where retrieval sufficiency scores
low — and on FEVER, `r` scoring low is mostly an artefact of the corpus (one gold
passage, constant relevance, no graph, no source labels) rather than a fact about
the evidence. That is a measurement-validity problem for `r` on this corpus, and
it is the same conclusion the r-ablation reached from the other direction.

Other sweep observations, none acted on: accuracy is highest at `r = 1.0`
(0.765), and the small edge there over raw `argmax(e,n,c)` (0.755) comes from
`λ·disagree` in F. `lambda` is nearly flat for
attribution (0.455–0.470 over 0–2) but monotonically raises the
SURFACE_CONTRADICTION rate. `connectivity` and `diversity` weights are inert on
FEVER; at weight 1.0 for either, every *available* component has zero weight and
`r` falls to 0.

## Phase 3 / RQ3 — the gate FAILED. Pre-registered stop condition fired.

```bash
python experiments/run_hotpot_gate.py          # N=400, seed 20261019, 150.5 min
python experiments/analyse_hotpot_gate.py
```

Pre-registered in [`PREREGISTRATION.md`](PREREGISTRATION.md) as the question that
decides the project. **The answer is: indeterminacy-triggered expansion recovers
answers that fixed-`k` misses, and does not beat a scalar-confidence trigger at
equal expansion budget.** The novelty claim does not survive.

Conditions: identical retriever, identical reader, identical expansion schedule
(`k = 3 → 6 → 10`). (b) and (c) are computed from the **same per-passage NLI
outputs**, so the only thing that differs between them is the decision function.

| condition | exp. rate | recovery | unnecessary exp. | **EM** | token-F1 |
|---|---:|---:|---:|---:|---:|
| (a) FIXED-k | 0.0000 | 0.0000 | 0.0000 | 0.2675 | 0.3636 |
| **(c) I-GATE** | 0.7775 | 0.7027 | 0.7865 | **0.3025** | 0.4032 |
| (b) scalar `max(e,n,c)` | 0.7775 | 0.7342 | 0.7584 | **0.3225** | 0.4247 |
| (b) scalar `1-n` | 0.7775 | 0.7477 | 0.7921 | **0.3075** | 0.4086 |
| (b) scalar `e` | 0.7775 | 0.8153 | 0.7247 | **0.3100** | 0.4066 |

Recoverable-miss subset: **222/400 (55.5%)** — five times the pre-registered
power floor of ~40. Top-3 already sufficient on 178/400 (44.5%).

### RQ3, first half: YES — expansion recovers misses

On the recoverable-miss subset, EM rises from **0.1486** (fixed-`k`) to **0.2432**
(I-gate), exact McNemar **b=30, c=9, p=0.0011**. Indeterminacy-triggered
expansion genuinely recovers answers that fixed-`k` retrieval misses, and the
effect is significant.

### RQ3, second half: NO — and this is the claim

**All three scalar gates beat the I-gate on end-to-end EM.** The I-gate is last
of the four gated conditions:

| comparison | McNemar | p |
|---|---|---:|
| (c) vs (a) FIXED-k, whole sample | b=34, c=20 | 0.0759 |
| (c) vs (a) FIXED-k, recoverable-miss | **b=30, c=9** | **0.0011** |
| (c) vs (b) `max(e,n,c)` | b=14, c=22 | 0.2430 |
| (c) vs (b) `1-n` | b=6, c=8 | 0.7905 |
| (c) vs (b) `e` | b=7, c=10 | 0.6291 |

No scalar beats the I-gate *significantly*, but the I-gate does not beat any of
them, and the pre-registered condition was that it must. **Conditioning on `I`
specifically adds nothing over conditioning on low confidence generally.**

### The null is NOT explained by the two triggers choosing the same questions

This was the obvious alternative reading and it is ruled out. Equal firing rates
do not imply equal decisions: with `|A| = |B| = 311` of 400, pigeonhole forces
overlap `≥ 2(311) − 400 = 222`, i.e. **≥ 71.4%** of the firing set, and equal
rates make the two discordant sets exactly the same size.

| gate | \|A∩B\| | overlap | c-only | b-only | decision agreement | effective n |
|---|---:|---:|---:|---:|---:|---:|
| `max(e,n,c)` | 225 | **72.3%** | 86 | 86 | 57.0% | **172** |
| `e` | 251 | 80.7% | 60 | 60 | 70.0% | 120 |
| `1-n` | 277 | 89.1% | 34 | 34 | 83.0% | 68 |

`max(e,n,c)` overlaps the I-gate on **72.3%, barely above the 71.4% floor** — its
decisions are close to maximally different given the rate constraint, on 172
discordant questions. So the two triggers select substantially *different*
questions and the scalar still wins. Restricted to the discordant set, where all
the signal is (a concordant question runs an identical pipeline and returns an
identical answer, contributing a noise-free tie):

| gate | n_discordant | (c) I-gate EM | (b) scalar EM | difference | McNemar p |
|---|---:|---:|---:|---:|---:|
| `max(e,n,c)` | 172 | 0.2674 | 0.2907 | −0.0233 | 0.5847 |
| `e` | 120 | 0.3083 | 0.3167 | −0.0083 | 1.0000 |
| `1-n` | 68 | 0.3235 | 0.3382 | −0.0147 | 1.0000 |

The I-gate is behind on every discordant set. This is not an untested
equivalence; it is a genuine difference in which the decomposition is **not
better**.

### Addendum 2's prediction is falsified

Registered in advance: *"(c) will beat (b) on recovering gold into context."*
It did not. The I-gate recovered the **least** gold of all four gated conditions
— 0.7027 against 0.7342, 0.7477 and 0.8153. The entailment scalar `e` recovered
the most gold (0.8153), used the **fewest** unnecessary expansions (0.7247) and
scored higher EM (0.3100). It beats the I-gate on all three axes at once.

### One budget caveat, stated rather than buried

The pre-registered criterion — the fraction of questions that expand at all —
is matched **exactly**: 0.7775 for every gated condition, 311/400. Total
expansions are *not* identical, and the I-gate uses **fewer**:

| condition | mean expansions | total | mean final k |
|---|---:|---:|---:|
| (c) I-GATE | 1.3550 | **542** | 7.64 |
| (b) `max(e,n,c)` | 1.4575 | 583 | 8.05 |
| (b) `1-n` | 1.5075 | 603 | 8.25 |
| (b) `e` | 1.5275 | 611 | 8.33 |

The I-gate consumes 7–13% less total retrieval, because it stops re-firing
sooner after expanding. So it is **cheaper and less accurate**, and a reading in
which the gap narrows at equal *total* consumption cannot be excluded from this
data. It cannot rescue the claim, though: the `e` scalar beats the I-gate on EM,
on recovery and on unnecessary expansions simultaneously, and no budget
adjustment reverses a three-way loss. Per the pre-registration, no threshold was
touched after calibration.

### Two findings that outlast the verdict

**1. Recovering the evidence is not recovering the answer.** Expansion puts the
gold paragraph into context 70–82% of the time on the recoverable-miss subset,
yet EM there reaches only 0.24–0.27. The gate's job is largely being done; the
reader then fails anyway. An extractive span reader handed more passages has
more distractors to choose between, so retrieval recovery and answer recovery
come apart. **This caps what any gating signal can deliver on this pipeline**,
and it means RQ3's second half was being measured through a lossy downstream
stage.

**2. Expansion helps bridge questions and hurts comparison questions.**

| condition | bridge (n=321) | comparison (n=79) |
|---|---|---|
| (a) FIXED-k | 0.293 [0.24, 0.34] | **0.165** [0.08, 0.25] |
| (c) I-GATE | 0.349 [0.30, 0.40] | 0.114 [0.04, 0.18] |
| (b) `max(e,n,c)` | **0.368** [0.31, 0.42] | 0.139 [0.06, 0.22] |
| (b) `1-n` | 0.358 [0.31, 0.41] | 0.101 [0.03, 0.17] |
| (b) `e` | 0.358 [0.31, 0.41] | 0.114 [0.04, 0.18] |

Every gated condition improves bridge and **degrades comparison below
fixed-`k`**. This is consistent with Phase 2: comparison questions already
retrieve well (R@3 = 0.873), so expansion adds only distractors. The comparison
arm is underpowered (n=79, with ~30.8% yes/no unreachable by a span reader), so
the intervals overlap — but the direction is uniform across all four gated
conditions, which makes it unlikely to be noise.

### Why the gate cannot decline: `r` does not distinguish the two types

```bash
python experiments/analyse_hotpot_by_type.py     # groupby, no re-run
```

Phase 2 established that comparison questions already retrieve well
(Recall@3 = 0.873 against bridge's 0.659) and Phase 3 that expansion **degrades**
comparison below fixed-`k`. A working gate would therefore fire *less* on
comparison. The I-gate fires **more**.

| condition | fires on bridge | fires on comparison | difference |
|---|---:|---:|---:|
| (b) `max(e,n,c)` | 0.8069 | 0.6582 | **−0.1486** |
| (b) `1-n` | 0.7882 | 0.7342 | −0.0540 |
| **(c) I-GATE** | 0.7726 | 0.7975 | **+0.0249** |
| (b) `e` | 0.7664 | 0.8228 | +0.0564 |

**The pre-registered "weak" scalar is the one that correctly declines**, firing
15 points less where expansion hurts — and it has both the best comparison EM
(0.1392) and the best pooled EM (0.3225). The decomposed gate cannot decline.

The mechanism is `r`, and it is unambiguous. At base depth `r` is **flat across
question types** despite retrieval quality differing by 0.21 Recall@3:

| type | mean r | median r | min | max | mean `1-r` | `r < 0.5` |
|---|---:|---:|---:|---:|---:|---:|
| bridge | 0.4562 | 0.4593 | 0.3282 | 0.5333 | 0.5438 | 265/321 (82.6%) |
| comparison | 0.4483 | 0.4500 | 0.3111 | 0.5333 | 0.5517 | 65/79 (82.3%) |

`r` differs by **0.008** between a type retrieved at R@3 = 0.873 and one
retrieved at 0.659. It is not measuring retrieval adequacy in any way that
tracks retrieval success. And because `r < 0.5` on **82% of the whole sample**,
`T = r·e ≤ r < 1-r ≤ I` there: *"well supported, do not expand" is
unrepresentable on four questions in five.* This is the Addendum 2 degeneracy
measured directly, and it is the proximate cause of the RQ3 null.

### Comparison questions only: fixed-`k` beats every gated condition

n = 79, underpowered by design — the direction and counts are the point.
Fixed-`k` EM = 0.1646 (13/79).

| condition | EM | b | c | p | direction |
|---|---:|---:|---:|---:|---|
| (b) `max(e,n,c)` | 0.1392 | 2 | 4 | 0.6875 | fixed-`k` better |
| (c) I-GATE | 0.1139 | 2 | 6 | 0.2891 | fixed-`k` better |
| (b) `e` | 0.1139 | 3 | 7 | 0.3438 | fixed-`k` better |
| (b) `1-n` | 0.1013 | 2 | 7 | 0.1797 | fixed-`k` better |

*(b = gated right where fixed-`k` wrong; c = the reverse.)* Every one is null
individually, but `c > b` in **all four**, totalling b=9 against c=24. Four
conditions failing in the same direction is weak evidence individually and
consistent evidence jointly: on comparison questions, expanding at all is the
mistake.

Splitting comparison EM by whether the condition actually expanded (small
cells, indicative only):

| condition | expanded | did not expand |
|---|---|---|
| (b) `max(e,n,c)` | 0.1154 (n=52) | **0.1852** (n=27) |
| (b) `e` | 0.1077 (n=65) | 0.1429 (n=14) |
| **(c) I-GATE** | **0.1270** (n=63) | **0.0625** (n=16) |
| (b) `1-n` | 0.1207 (n=58) | 0.0476 (n=21) |

For `max(e,n,c)` and `e`, declining was associated with *higher* EM — the gate
withheld expansion where withholding helped. For the I-gate the relation is
**inverted**: its 16 non-firing decisions scored 0.0625 against 0.1270 for the
ones it expanded. Its selectivity is not merely absent but mildly
anti-correlated with where selectivity pays. At n=16 this is indicative, not
established.

### What this means for the claim

The narrowed claim registered in Phase 0 was *expansion conditioned on a
decomposed evidence state, where indeterminacy specifically triggers
expansion*. The decomposition **selects different questions** from a scalar
trigger — demonstrably, at 72.3% overlap against a 71.4% floor — and selecting
them differently does **not** select them better. Against the pre-registered
stop condition, the claim fails.

What survives is weaker and must be stated as such: **expansion triggered on
the evidence state beats no expansion** (p=0.0011 on the recoverable-miss
subset), which is the FLARE/CRAG/Adaptive-RAG result reproduced in a
training-free setting, not a new one.

**Phases 4 and 5 were not run**, as the pre-registration requires. No threshold,
weight, lambda or target was adjusted after the verdict.

## Phase 2: Track B retrieval metrics (HotpotQA distractor)

```bash
python experiments/run_hotpot_retrieval.py      # N=400, seed 20261019
```

Properties of the retriever alone — no reader, no NLI, no policy — so they are
shared by every condition in Phases 3 and 5 and are reported once. N = 400
stratified sample (321 bridge / 79 comparison), paragraph-level retrieval with
`all-MiniLM-L6-v2`. Gold is always exactly 2 paragraphs, so per-instance recall
takes only the values 0, 0.5 and 1.

| k | Recall@k | 95% CI | context precision@k | sentence recall@k |
|---:|---:|---:|---:|---:|
| 1 | 0.4088 | [0.361, 0.457] | 0.8175 | 0.4122 |
| 2 | 0.6150 | [0.567, 0.663] | 0.6150 | 0.6187 |
| 3 | 0.7013 | [0.656, 0.746] | 0.4675 | 0.7033 |
| 5 | 0.8137 | [0.776, 0.852] | 0.3255 | 0.8177 |
| 10 | 1.0000 | [1.000, 1.000] | 0.2005 | 1.0000 |

**MRR = 0.8904** (first gold paragraph) · **nDCG@10 = 0.8431**

Two of these are not results and must not be read as any:

- **Recall@10 = 1.0 on 400/400 instances is an index-completeness check.** Every
  paragraph an instance owns is indexed, so retrieving ten of at most ten
  retrieves the gold necessarily. It confirms the harness indexes what it
  claims to and says nothing about the retriever.
- **Context precision@k falls as k grows** (0.8175 → 0.2005) purely because the
  numerator is capped at 2 gold paragraphs while the denominator is k. It is
  arithmetic, not degradation.

The informative figures are the middle rows. At k=2 — two retrieved for two gold,
the fairest single comparison — **both** gold paragraphs are retrieved on only
**30.0%** of instances, and **neither** on **7.0%**. MRR of 0.8904 against
Recall@1 of 0.4088 says the retriever reliably puts *one* gold paragraph near the
top and routinely misses the second, which is the characteristic multi-hop
failure: the bridge entity is not mentioned in the question.

### Sentence-level recall is reported separately, on purpose

Supporting facts are published per **sentence**; retrieval here operates on
**paragraphs**. Sentence recall tracks paragraph recall closely (0.6187 vs 0.6150
at k=2) because gold sentences are concentrated in the gold paragraphs, but the
two are different quantities and conflating them would overstate what retrieval
delivered. Both are given at every k.

### By question type

| type | n | R@2 | R@3 | R@5 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| bridge | 321 | 0.5779 | 0.6589 | 0.7773 | 0.8717 | 0.8217 |
| comparison | 79 | 0.7658 | 0.8734 | 0.9620 | 0.9662 | 0.9303 |

**Comparison questions are much easier to retrieve for**, by a wide margin at
every k (R@3 0.873 vs 0.659). The reason is structural: a comparison question
names both entities explicitly ("Were Scott Derrickson and Ed Wood of the same
nationality?"), so both gold paragraphs are lexically and semantically close to
the query. A bridge question names only the first entity and the second must be
reached through it, so the dense retriever has nothing to match on. This is worth
stating because the same two types invert on **answering**: comparison questions
are far easier to retrieve and far harder to answer, 30.8% of them being yes/no
and unreachable by a span reader.

### A defect this exposed in the retriever

`DenseRetriever.retrieve` returns `Evidence(text, relevance)` and **does not
propagate the `Document` id**. Document identity is therefore lost at the
retrieval boundary, which has two consequences:

1. The experiment has to recover each paragraph's title by reverse-mapping its
   text, which is what `run_hotpot_retrieval.py` does.
2. More seriously, the scorer receives `sources=None`, so the **`diversity`
   component of `r` silently degrades** rather than being computed. Combined
   with `connectivity` being unavailable (no graph is built for HotpotQA), only
   `coverage` and `support_count` can vary on this corpus — two of four
   components, against one of four on FEVER.

Phase 3 re-attaches sources explicitly when constructing `Evidence` so the
component is not silently zeroed. The underlying retriever is left unchanged:
`retrieval.py` is not modified mid-experiment.

## RQ1: pairwise `disagree` resists dilution — the proof

Pre-registered as RQ1 in [`PREREGISTRATION.md`](PREREGISTRATION.md). The
measurement already existed (below); this is the two-direction argument for why
it comes out that way, which is what makes it a property of the operator rather
than a fact about one evidence set.

Notation: passages `i = 1..n` with relevance weights `w_i > 0` and per-passage
NLI outputs `(e_i, n_i, c_i)`. Write `W = Σ w_i`.

### Direction 1 — mean-pooled aggregation is dilution-sensitive by construction

The pooled form first collapses the per-passage outputs to weighted means,

&nbsp;&nbsp;&nbsp;&nbsp;`ē = Σ wᵢeᵢ / W`,&nbsp;&nbsp;`c̄ = Σ wᵢcᵢ / W`,

and then takes `disagree = 2·min(ē, c̄)`.

Add one passage `j` that is neutral: `e_j ≈ 0`, `c_j ≈ 0`, `w_j > 0`. The new
means are

&nbsp;&nbsp;&nbsp;&nbsp;`ē' = (Σ wᵢeᵢ + w_j e_j) / (W + w_j)`,&nbsp;&nbsp;
`c̄' = (Σ wᵢcᵢ + w_j c_j) / (W + w_j)`.

The denominator strictly increases by `w_j`. Neither numerator increases, since
`w_j e_j ≈ 0` and `w_j c_j ≈ 0`. Hence `ē' < ē` and `c̄' < c̄` strictly, provided
the corresponding numerator was non-zero. `min` is monotone non-decreasing in
each argument, so

&nbsp;&nbsp;&nbsp;&nbsp;`2·min(ē', c̄') < 2·min(ē, c̄)`.

Every neutral passage added therefore strictly lowers the conflict signal. The
decay is monotone and unbounded below: adding enough neutral passages drives
`disagree` to 0 regardless of how sharply two passages contradict each other.
**This is not a tuning artefact — it is what averaging means.**

### Direction 2 — pairwise aggregation is invariant

The pairwise form never pools. It takes

&nbsp;&nbsp;&nbsp;&nbsp;`disagree = 2 · max_{(i,j)} min(eᵢ, c_j)`

over ordered pairs of distinct passages. Let `M = max_{(i,j)} min(eᵢ, c_j)` over
the original set.

Adding passage `j*` with `e_{j*} ≈ 0`, `c_{j*} ≈ 0` does two things and only two:

1. **It leaves every existing pair intact.** The pairs `(i,j)` among the original
   passages are unchanged, because no term in `min(eᵢ, c_j)` depends on any other
   passage or on `n`. So `M' ≥ M`: the max is taken over a superset.
2. **The pairs it adds cannot raise the max.** A new pair is either `(j*, j)`,
   contributing `min(e_{j*}, c_j) ≤ e_{j*} ≈ 0`, or `(i, j*)`, contributing
   `min(eᵢ, c_{j*}) ≤ c_{j*} ≈ 0`. Both are `≈ 0 ≤ M`. So `M' ≤ max(M, 0) = M`.

Together `M' = M`. The signal is **invariant** to neutral passages, exactly:
not approximately, and not because of clamping. `max` is the right aggregator
here precisely because conflict is an **existential** property of a set — "some
pair disagrees" — whereas a mean answers a **universal** question, "how much
does the set disagree on average", which is not what the policy needs to know.

### The measurement this predicts

```bash
python experiments/check_conflict_dilution.py --nli real
```

| filler passages added | 0 | 1 | 2 | 3 | 4 | spread |
|---|---:|---:|---:|---:|---:|---:|
| `disagree` OLD (pooled) | 0.847 | 0.643 | 0.525 | 0.452 | 0.400 | **0.446** |
| `disagree` NEW (pairwise) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **0.000** |
| raw, unclamped | 1.739 | 1.739 | 1.739 | 1.739 | 1.739 | 0.000 |

Both directions are confirmed. The pooled row decays monotonically, as Direction
1 requires. The pairwise row is flat, as Direction 2 requires — and the **raw,
unclamped** row is flat at 1.739 at every filler count, which rules out the one
alternative explanation: the flatness is genuine invariance, not the clamp at
1.0 hiding a decay.

### Scope — what this does and does not distinguish

**It does distinguish the operator from mean-pooled NLI aggregation.** That is
the standard construction in NLI-based RAG verification, and it is SURE-RAG's
own weak baseline, so the comparison is against a real and current practice
rather than a straw man.

**It does not distinguish it from Dempster-Shafer methods, and must not be
claimed to.** In a DS framework a neutral passage contributes a *vacuous* mass
assignment, and vacuous mass is the **neutral element of the combination rule by
construction** — so dilution-invariance is automatic there and is not a result.
EvidentialRAG (arXiv 2607.10491) is such a method. Against ERAG the honest
statement is that both are dilution-invariant, by different routes, and this
proof buys nothing. What survives as a distinction is in RQ3, not here.

A second limit: the proof assumes `e_j ≈ 0` *and* `c_j ≈ 0`, i.e. genuinely
neutral filler. A passage that is partly entailing or partly contradicting is
not filler, and *should* move the signal. Invariance is claimed only for the
neutral case.

## Step 6: pairwise `disagree` — fixes F6 and F8

`scoring.py`: `disagree = 2·min(e, c)` on the pooled means becomes
`2·max over pairs (i,j) of min(e_i, c_j)` on the per-passage outputs. `T`, `F`,
`I` and `lambda` byte-identical.

```bash
python experiments/check_conflict_dilution.py --nli real
```

| filler passages added | 0 | 1 | 2 | 3 | 4 | spread |
|---|---:|---:|---:|---:|---:|---:|
| `disagree` OLD (pooled) | 0.847 | 0.643 | 0.525 | 0.452 | 0.400 | **0.446** |
| `disagree` NEW (pairwise) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **0.000** |
| raw, unclamped | 1.739 | 1.739 | 1.739 | 1.739 | 1.739 | 0.000 |

The raw value is identical at every filler count, so the flatness is genuine
invariance, not clipping. `F` itself still drifts 0.906 → 0.692 (spread 0.214)
because `F = r·c + λ·disagree` and the pooled `c` falls as filler is added; only
the `disagree` term was ever expected flat. After the Decision-2 target
calibration `r` no longer falls with it — it is flat at **0.958** across the
whole sweep, because `support_count` saturates at the first passage — so the
residual drift in `F` is now attributable to the pooled `c` alone.

**F8 is fixed.** The flagship drug-trial case now returns **SURFACE_CONTRADICTION
with F = 0.906** (real NLI) against `tau_F = 0.40`, versus the memo's F = 0.347
and ABSTAIN. It still fires with 4 filler passages added (F = 0.692).

Mass identity still broken (2.499e-01). FEVER `e`/`n`/`c` **bit-identical** — with
one caveat: `fever_run_200_gateA.csv` stores the *pooled* triple per claim, so on
replay every passage carries the same values and pairwise ≡ pooled by
construction for **all** row sizes, not only the 185 single-passage ones.
`fever_run_200_current.csv` records a per-row `disagree` column taken from the
real per-passage outputs, so figures derived from it do not share that defect.

## Step 9: CONFLICTS — run, halted, and not interpretable as intended

Dataset: Cattan et al., arXiv 2506.08500,
[google-research-datasets/rag_conflicts](https://github.com/google-research-datasets/rag_conflicts).
458 instances, 9.20 passages each, 8.33 domains each. **The release publishes no
retrieval score** — relevance is derived as question–passage cosine similarity
(see `conflicts.RELEVANCE_SOURCE`). Scored a seeded stratified sample of 130
instances / 1210 passages (~107 min CPU).

```bash
python experiments/score_conflicts.py && python experiments/run_conflicts_eval.py
python experiments/diagnose_conflicts_hypothesis.py
```

### The headline numbers, and why they must not be read at face value

| | result | stop condition |
|---|---|---|
| 3b `disagree` by type | Conflicting opinions median **0.222** vs No conflict median **0.556**; Mann-Whitney p = 2.14e-03 — **reversed** | triggered |
| 3c policy test | exact-match **0.267** (CI 0.175–0.358, n=90); SURFACE_CONTRADICTION recall **0.200**, firing on 20/45 *No conflict* vs 9/45 *Conflicting opinions* | triggered |
| 3a/3d r-ablation | **null**: r=0.9 and r=sufficiency agree on **128/130** | uninformative |

**The cause is a measurement error.** CONFLICTS is a QA dataset; its instances
are *questions* ("what is the most common blood type"), and each question was
passed to the NLI model as the hypothesis. Entailment against an interrogative
is not a well-posed problem, and the output shows it: median entailment 0.709
(59.6% above 0.5), **median contradiction 0.017**. The model marks any on-topic
passage "entailing" and emits arbitrary contradiction scores.

Concretely: the highest-`disagree` *No conflict* instance pairs a blood-type
frequency table (e=0.985) against a generic "Blood Types — as you probably
know…" intro (c=0.993) — neither contradicts anything. The lowest-`disagree`
*Conflicting opinions* instance has max contradiction **0.014** across 7
passages, because none of them contradicts a question. `correct_answer` is
non-empty for only 237/458 instances, so this cannot be patched by forming
question+answer claims.

**So 3b and 3c are not evidence that `disagree` or the policy are wrong.** They
are evidence that the scorer was applied outside its task definition.

### What the run does establish

`support_count` and `diversity` **saturate at 1.0** on every instance (≥5
passages, ≥3 domains, frozen target 3); `r` spans only 0.867–1.000 with 0/130
below 0.5. FEVER spans [0.500, 1.000] with a median of 0.750 and 17/200 rows at
`r ≤ 0.5`; CONFLICTS pins it high. **Neither venue exercises `r` across its full
range**, which is a property of the components' targets, not of either dataset.
Reported, not retuned.

*(Before the Decision-2 calibration FEVER's median was 0.417 with 143/200 below
0.5, and the sentence above read "FEVER pinned `r` low". Calibrating the targets
moved FEVER into the upper half of the range; it did not widen it.)*

## Decision 2: saturation targets, derived from depth alone

```bash
python experiments/derive_targets.py     # reads depth only; prints no accuracy
```

**No fixed target can work, and on FEVER none works at all.** FEVER's passage
count is a point mass — p25 = median = p75 = p90 = **1**, with 92.5% of
instances at k=1 — so a saturating function of k cannot manufacture variance the
input does not contain. CONFLICTS is concentrated at 8–10, and at the old target
of 3 both count components were **100% saturated, std exactly 0.000**. The two
requirements are mutually exclusive.

The real defect in `target=3` was **bias, not lost discrimination**: FEVER's
`support_count` sat at the constant 0.333 and subtracted from `r` on every
instance, which is what put 143/200 below the F12 cliff.

**Rule adopted: target = p75 of that corpus's own depth distribution.** Not max
(outlier-driven), not median (saturates half the corpus by construction), and
explicitly *not* the spread-maximising value — that is still fitting, even to
depth. `fever.SUPPORT_TARGET = 1`, `conflicts.SUPPORT_TARGET = 10`,
`conflicts.DIVERSITY_TARGET = 9`, module fallback 1 (unbiased).

| prediction, committed in `53d305e` before any run | outcome |
|---|---|
| FEVER r median 0.417 → **0.750** | **0.750** ✓ |
| FEVER instances at r ≤ 0.5: 143/200 → **17/200** | **17/200** ✓ |
| CONFLICTS r median 1.000 → **0.930**, pstdev 0.0297 → **0.0856** | **0.930 / 0.0856** ✓ |

## Decision 1: pairwise mode, and its sanity gate

```bash
python experiments/check_pairwise_sanity.py --nli real
```

`T` is **corroboration** in this mode — mutual agreement between sources, never
"support for the claim". The query is used only for retrieval sufficiency and is
never passed to the NLI model as a hypothesis (asserted structurally by
`test_query_is_never_passed_to_the_nli_model`).

| case | corroboration | contradiction | neutrality | dominant | expected |
|---|---:|---:|---:|---|---|
| contradicting | 0.000 | **1.000** | 0.001 | F | F ✓ |
| agreeing | **0.997** | 0.001 | 0.003 | T | T ✓ |
| unrelated | 0.007 | 0.002 | **0.984** | I | I ✓ |

Against claim mode's **median contradiction of 0.017** on this corpus. The gate
costs a minute and runs before any experiment, because the previous venue
attempt spent 107 minutes before its framing error surfaced.

**A bug this caught, in my own code:** `pairwise_breakdown` first computed
sufficiency on the *capped* passage set, so every CONFLICTS instance showed
exactly `max_passages=5` and `support_count` re-saturated at a constant 0.5 —
reintroducing the exact defect Decision 2 removes. `r` now uses the full
retrieved set; the cap is a compute budget on the O(k²) comparison only.

## FEVER regression under the new targets

```bash
python experiments/run_r_ablation.py     # replays fever_run_200_current.csv; model not re-run
```

| condition | accuracy | macro-F1 |
|---|---:|---:|
| (a) `r = 0.9` constant | 0.7550 | 0.7571 |
| (b) `r = sufficiency` | **0.6650** | 0.6730 |
| raw `argmax(e,n,c)` | 0.7550 | 0.7574 |

**(b) − (a) = −0.0900**, improved from −0.2950 under the old targets but **still
negative and significant**: McNemar b=0, c=18, p=0.0000. Computed `r` does not
win a single row that it loses. `r` now spans 26 distinct values, median 0.750.
Reported, not tuned.

## Task 4: the policy test in pairwise mode — BOTH stop conditions fired

```bash
python experiments/score_conflicts_pairwise.py      # 100 instances, 2000 pairs
python experiments/run_conflicts_pairwise_eval.py
python experiments/diagnose_pairwise_reversal.py
```

### Stop condition 2 — contradiction reverses again

| conflict type | n | median contradiction | median corroboration |
|---|---:|---:|---:|
| Conflicting opinions | 40 | **0.100** | 0.878 |
| No conflict | 40 | **0.331** | 0.757 |
| Complementary information | 10 | 0.667 | 0.709 |
| Outdated information | 10 | 0.888 | 0.911 |

Mann-Whitney U=525, z=−2.65, **p = 8.14e-03** — significantly **higher on
No conflict**. The prediction was the opposite.

### Stop condition 3 — the policy fires on the wrong class

| conflict type | answer | expand | surface_contradiction | n |
|---|---:|---:|---:|---:|
| Conflicting opinions | 18 | 15 | **7** | 40 |
| No conflict | 9 | 16 | **15** | 40 |
| Complementary information | 1 | 4 | 5 | 10 |
| Outdated information | 1 | 2 | 7 | 10 |

SURFACE_CONTRADICTION fires on **17.5%** of Conflicting opinions vs **37.5%** of
No conflict. Exact-match accuracy **0.200** (95% CI 0.112–0.288, n=80);
surface_contradiction recall 0.175, answer recall 0.225.

**On this evidence the policy's central claim is unsupported.**

### This is not the previous measurement error

Pairwise mode **passed its sanity gate** on real NLI, so the instrument works.
The mechanism, from `diagnose_pairwise_reversal.py`:

- **Real conflicts are hedged, and NLI reads them as agreement.** The
  lowest-contradiction Conflicting-opinions instance pairs *"some companies
  require a degree, but it's becoming less…"* against *"anyone can become
  proficient"* — max entailment **0.992**, max contradiction **0.006**.
- **Different aspects produce spurious contradiction.** The
  highest-contradiction No-conflict instance pairs *"Starmer became PM on 5 July
  2024"* against *"there is no specific date for when the office first
  appeared"* — **c = 0.999**. Both true, different questions.
- Per-pair contradiction is rare (median 0.021; 5.7% of 2000 pairs above 0.5),
  so a max over 20 pairs is an order statistic latching onto the rare spurious
  pair.
- **The reversal is not an artefact of `max`:** the mean over pairs reverses too
  (−0.053). The per-pair signal itself is anti-correlated with annotated
  conflict. Reported as a diagnostic only — the statistic was **not** changed
  after seeing which way it separates.

**Unresolved confound:** the top-5 cap keeps the passages most similar to the
*question*, and a passage restating the question outranks one disputing its
premise, so the cap is not conflict-neutral. Not testable from this cache.

**Cost:** projected 2.49 h, actual **12.8 h** (23.0 s/pair against a 4.48 s/pair
probe). The probe was unrepresentative; the stated 3-hour ceiling was breached.

## RETRACTED

These numbers appear in the First Review report and must not be reused as written.

### 1. `0.745` as a NeutroRAG result — RETRACTED

*This section describes the **Gate A** measurement, i.e. the construction as
submitted. The current construction scores lower still — see the RQ1 subsection
under "Current, defensible numbers".*

The figure itself reproduces exactly, but it is **not attributable to the
neutrosophic layer**. Running `argmax` directly on the NLI head's own
distribution, using the *identical* model calls, scores **0.7550** — higher.

| system | accuracy | 95% CI (Wald) | macro-F1 |
|---|---:|---:|---:|
| NeutroRAG `argmax(T,I,F)` | 0.7450 | [0.685, 0.805] | 0.7476 |
| raw NLI `argmax(e,n,c)` | **0.7550** | [0.695, 0.815] | **0.7574** |
| difference | −0.0100 | | −0.0098 |

The two systems differ on **4 of 200** rows. Exact McNemar: b=1, c=3, **p = 0.625** —
the difference is not statistically significant in either direction. For the
submitted construction the honest claim was therefore *"the neutrosophic
construction produces no measurable benefit over the bare classifier on this
data,"* not *"it is worse."*

> **That qualification no longer holds for the current code.** Under the
> construction on `main` the same comparison gives 0.6650 vs 0.7550, a gap of
> −0.0900 with exact McNemar b=1, c=19, **p = 0.0000** — significant, and in the
> direction of worse. "No measurable benefit" was the correct reading of the
> Gate A numbers and remains the correct reading of the *submitted* system; it
> understates what the current numbers say.

Corroboration: the model card for `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
reports **0.777** on the full FEVER-NLI dev set. Our 0.755 on a 200-row subset of
that same split is consistent with simply re-measuring the classifier.

*Source: `results/fever_200_metrics_gateA.md`, from `fever_run_200_gateA.csv`.*

### 2. The fuzzy-collapse ablation delta `+0.205` — RETRACTED

Arithmetic, not an experiment (memo F4). Two independent problems:

- **Ceiling.** With gold counts 65 / 67 / 68, *any* I/F-merged model — even one with
  a perfect T detector — is capped at (65+68)/200 = **0.665** by the label prior.
  The delta measures how many gold-I items exist, not representational power.
- **Recoverable by inspection.** The reported 0.540 = `conf[T][T] + F_gold` =
  40 + 68 = 108 → 0.540, exactly. A reviewer can derive it from the published
  confusion matrix in about a minute.

### 3. "T + I + F can exceed 1, demonstrating independence" — RETRACTED as stated

**Status: fixed as of Gate B (revised), but the retraction stands for the numbers
already reported.** Under the `I = (1-r) + r·n` used for every number in the First
Review report, `T + I + F = 1 + λ·disagree` identically (max deviation 2.22e-16
over the 200 rows), so the triple carried one degree of freedom beyond the
probability simplex, not three, and "mass exceeds 1" merely restated
"disagree > 0". Under the current `I = max(1-r, n)` the identity is broken
(deviation up to 0.25) and the mass does carry independent information — see the
Gate B section above. Any *reported* number predating that change must still be
described using the old semantics.

---

## Current, defensible numbers

### FEVER run (n = 200) — the current construction

`copenlu/fever_gold_evidence` validation split, `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, CPU.

```bash
python experiments/run_fever_eval.py --limit 200 --csv fever_run_200_current.csv   # ~150s
python experiments/make_results_table.py --csv fever_run_200_current.csv \
    --out-prefix fever_200_metrics_current
```

Full per-class table and both confusion matrices:
**`results/fever_200_metrics_current.md`**. Per-example dump:
**`fever_run_200_current.csv`** (200 rows: claim, r, e, n, c, disagree, T, I, F,
gold, predicted, evidence_count).

This is the construction on `main`: `r` from retrieval sufficiency,
`I = max(1-r, n)`, pairwise `disagree`, Decision-2 saturation targets.

| quantity | value | note |
|---|---|---|
| attribution accuracy `argmax(T,I,F)` | **0.6650 ± 0.0334** | 95% CI [0.600, 0.730] |
| baseline accuracy `argmax(e,n,c)` | **0.7550 ± 0.0304** | identical NLI calls; unchanged since Gate A |
| difference | **−0.0900** | macro-F1 −0.0844 (0.6730 vs 0.7574) |
| rows where predictions differ | 20 / 200 | |
| McNemar (exact) | **b=1, c=19, p=0.0000** | **significant** at α=0.05 |
| `N_empty` (memo F3) | **0 / 200** | the leak is live code but fired zero times here |
| `r` (memo F1, resolved by Gate B) | 26 distinct values, median 0.750, range [0.500, 1.000] | `r` is now a signal, not a constant |
| `disagree` (memo F6/F9) | median 0.0115, mean 0.0299, max 0.7165; 13/200 rows > 0.1 | conflict term near-inert on FEVER |
| evidence passages per row | 185 rows have exactly 1 | so there is no pair to compare on most rows |
| mass identity deviation (memo F5) | **2.275e-01** | identity **broken**, as the revised Gate B predicted |
| mass `T+I+F` | min 0.775, median 0.988, max 1.277; >1 on 49/200 (24.5%) | |

#### This answers RQ1, and the answer got worse

**RQ1 — does the neutrosophic decomposition beat the classifier alone? No, and
now significantly so.**

| | Gate A (as submitted) | current construction |
|---|---:|---:|
| `argmax(T,I,F)` | 0.7450 | **0.6650** |
| `argmax(e,n,c)` baseline | 0.7550 | **0.7550** |
| difference | −0.0100 | **−0.0900** |
| McNemar | b=1, c=3, p=0.625 (ns) | **b=1, c=19, p=0.0000** |

Three repairs landed between those columns — `I = max(1-r, n)`, pairwise
`disagree`, and the calibrated targets — and the gap **widened by a factor of
nine and became statistically significant**. At Gate A the honest statement was
"the layer adds nothing measurable". It is now stronger than that: on FEVER the
layer measurably *destroys* information the NLI head already had. The baseline is
identical in both columns, as it must be — it is a function of `e`, `n`, `c`
only, and those are bit-identical across the two dumps.

The mechanism is visible in the per-class table. Recall on the two decidable
classes falls (T 0.646 → 0.554, F 0.853 → 0.662) while recall on `I` rises
(0.761 → 0.776), and precision on `I` falls to 0.500. `I = max(1-r, n)` sets a
floor of `1-r` under indeterminacy, and on FEVER `r` has a median of 0.750, so a
floor of ~0.25 is imposed on every row regardless of evidence. Where the NLI head
was confidently right, that floor is enough to let `I` win the argmax. The
repair that made the mass identity break genuinely is the same repair that costs
accuracy here, and the two cannot be separated by tuning — they are the same
term.

This does not retract the Gate B change. `I = max(1-r, n)` was adopted because
the old `I` made "T+I+F > 1" an algebraic identity, which is a correctness
problem independent of accuracy. The honest summary is that fixing it revealed
the decomposition was carrying less information than the headline suggested, not
more.

**What this does not say.** FEVER supplies gold, annotator-selected evidence at
constant relevance, with a single passage on 185/200 rows (memo F9). Three of
`r`'s four components cannot vary there and the conflict term has nothing to
compare, so this is close to a worst case for the construction — it is the
corpus least able to reward retrieval-side signal. That is an argument for
measuring on a corpus with real retrieval depth, not an argument that the number
here is wrong. The CONFLICTS test was that measurement, and it also came out
negative.

### The frozen Gate A dump

**`fever_run_200_gateA.csv`** is the per-example dump for the construction
exactly as submitted at First Review: **constant `r` = 0.9**, **`I = (1-r) + r·n`**,
and **pooled `disagree` = 2·min(e, c)**. It is retained **unmodified** —
byte-identical to the file committed as `fever_run_200.csv`, SHA-256
`70a3a783…84c6b7` — because the retraction of the 0.745 headline rests on it.
Nothing in this repository should regenerate or overwrite it.

```bash
python experiments/make_results_table.py --csv fever_run_200_gateA.csv \
    --out-prefix fever_200_metrics_gateA
```

| quantity | Gate A value | note |
|---|---|---|
| attribution accuracy `argmax(T,I,F)` | 0.7450 ± 0.0308 | the submitted headline; RETRACTED above |
| baseline accuracy `argmax(e,n,c)` | 0.7550 ± 0.0304 | identical NLI calls |
| rows where predictions differ | 4 / 200 | |
| McNemar (exact) | b=1, c=3, p=0.625 | not significant |
| distinct values of `r` (memo F1) | `{0.9}` | `r` was a constant, not a signal |
| mass identity deviation (memo F5) | 2.22e-16 | identity **held** — which is why it was retracted |

Its `claim`, `e`, `n`, `c`, `gold_*` and `evidence_count` columns are still
valid and are what every replay experiment (`run_r_ablation.py`,
`run_sensitivity_sweep.py`) consumes; its `r`, `T`, `I` and `F` columns describe
the superseded construction and must never be quoted as current.

### Contamination check (memo F10)

```bash
python experiments/check_contamination.py
```

| against FEVER-NLI split | claim-text overlap | claim-ID overlap |
|---|---:|---:|
| **train** | 0 / 196 (0.0%) | 0 (0.0%) |
| **dev** | **196 / 196 (100%)** | **196 (100%)** |

**No training contamination.** But the 200 evaluated claims *are* the FEVER-NLI dev
set, and the model card states the model "was evaluated using … the dev set for
Fever-NLI". Numbers measured here must be disclosed as measured on the model's own
reported development split, not presented as held-out.

Unresolved, and what would settle it:
1. Whether dev was used for checkpoint selection. The published `TrainingArguments`
   show a fixed `num_train_epochs=3` with no early stopping, which argues against it,
   but hyperparameters may still have been chosen by looking at dev. Only the
   author's training logs would settle this.
2. Whether `pietrolesci/nli_fever` is byte-identical to the split the model author
   used. It is a community re-host; the model card links no specific split file.

### Pipeline demo — the report's Section 5(d) screenshot no longer matches

```bash
python experiments/run_pipeline_demo.py --nli real
```

**The screenshot in the submitted report is a historical artefact of the submitted
code, and this run no longer reproduces it.** Both the triple and the action have
changed:

| | submitted report (screenshot) | current code (post-Step-6) |
|---|---|---|
| triple | `TIF(T=0.223, I=0.603, F=0.347)` | `TIF(T=0.209, I=0.603, F=0.663)` |
| steps | identical at steps 0, 1, 2 | a single step 0 |
| action | **ABSTAIN** | **SURFACE_CONTRADICTION** |
| `expansions_used` | 2 | 0 |

**The flip is the memo F8 conflict fix propagating into the full pipeline.** At
submission, conflict was computed as `2·min(e, c)` on the relevance-weighted *pooled*
means. The demo retrieves six passages, of which four are filler — a formulary note,
an approval date, an age-prevalence fact and an off-topic line about the stock market.
Pooling averaged the one contradicting passage against those four, so `F` fell to
0.347, below `tau_F` = 0.40, and the system abstained on the single example built to
showcase SURFACE_CONTRADICTION. Conflict is now the strongest opposition between *any*
entailing and *any* contradicting passage,
`disagree = 2·max_{i,j} min(e_i, c_j)`, which is a set-level property and does not
dilute. `F` rises to 0.663, clears `tau_F`, and the action fires at step 0.

That is the mechanism working end to end rather than abstaining, and it is the one
place in the repository where an audit finding can be seen to change a pipeline
outcome. Two caveats belong with it:

- **This is one hand-built trace, not a result.** The same F8 fix, measured against
  external ground truth on the CONFLICTS dataset, made the policy test come out
  *negative* (see *The CONFLICTS policy test* above). A mechanism firing correctly on
  an example built to exercise it is not evidence that it fires correctly in general,
  and this section must not be read as if it were.
- **`expansions_used` dropped to 0 for a reason unrelated to the fix.** The policy now
  acts at step 0, so it never enters the expansion loop, which means this trace no
  longer exercises that loop at all. The loop's behaviour is covered by
  `tests/test_memo_regressions.py::test_expansion_with_a_real_scorer_changes_the_triple`
  (the triple moves when the retriever returns new passages) and
  `::test_expansion_is_inert_when_the_retriever_returns_nothing_new` (it does not when
  the toy corpus is exhausted — which is what produced the three identical steps in
  the original screenshot). Both pass.

The mock-NLI invocation now reaches the **same action** by a different route:
`TIF(T=0.407, I=0.323, F=0.728)` → SURFACE_CONTRADICTION, 0 expansions. At submission
the two backends disagreed on this example; they no longer do. Still quote the
`--nli real` run and say which backend you ran, because the triples differ.

### Library behaviour (offline, mock NLI)

| quantity | value | command |
|---|---|---|
| README quickstart triple | `TIF(T=0.091, I=0.000, F=1.000)` | `python experiments/reproduce_all.py` |
| README quickstart, with `expand_fn` | ANSWER after 1 expansion | same |
| README quickstart, without `expand_fn` | ABSTAIN, 0 expansions | same |
| `MockNLIBackend.predict` sums to 1 | max \|sum − 1\| = 0.00e+00 | `pytest tests/test_memo_regressions.py -k distribution` |

### demo.py — illustrative only, not a research result

`demo.py` runs the **mock** NLI backend over 12 hand-written examples. Its
attribution figure is **0.833** (n=12) and is a smoke test of the plumbing, not
evidence of anything; it should not appear in the report as a result.

The fuzzy-collapse delta that this section previously quoted as a number is
**withdrawn** — see *Retraction 2* above. Merging the gold `I` and `F` classes
caps the collapsed score at the majority-class prior, so any positive delta is
arithmetic rather than evidence for the I/F distinction. `demo.py` no longer
prints it, and `experiments/reproduce_all.py` lists it under `[RETRACTED]` with
no value, so no command in this repository reproduces it. That is deliberate.

---

## Test suite

```bash
pytest -q          # 98 passed, 6 skipped, 1 xfailed
NEUTRORAG_TEST_HF=1 pytest tests/test_memo_regressions.py -k hf_backend   # 6 passed (~44s)
```

The **1 xfail is deliberate**: `test_unusable_evidence_is_distinguishable_from_nothing_retrieved`
documents memo F3 — the scorer cannot distinguish "nothing retrieved" from "retrieved
but unusable", so both collapse to `TIF(0,1,0)`. It is marked `strict=True`, so if the
defect is ever fixed the test will fail until the marker is removed deliberately.
Fixing F3 is downstream of Gate B.

The 6 skips are the opt-in HFNLIBackend distribution tests, which require the real model.
