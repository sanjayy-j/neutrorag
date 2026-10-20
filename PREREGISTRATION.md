# Pre-registration — Phase 3 of NeutroRAG

**Registered 19 October 2026, before the experiments it describes were run.**

This file exists so the research questions below cannot be mistaken for questions
chosen after their answers were known. It is committed as its own commit, with
nothing else in it, *before* the first Phase 3 experiment. The commit order in
`git log` is the evidence; if this file were written afterwards, the history
would show it.

Nothing here may be edited to match a result. If a prediction fails, the failure
is recorded in `RESULTS.md` and this file stays as written.

---

## 1. Why the project is being rescoped

A literature check found **EvidentialRAG** (ERAG), which makes the head-on claim
this project was built around:

> S M Asif Hossain, Ruksat Khan Shayoni, M. F. Mridha, "EvidentialRAG:
> Quantifying and Mitigating Information Conflict in Multi-Source
> Retrieval-Augmented Generation via Evidential Deep Learning,"
> arXiv:2607.10491 [cs.LG], submitted 11 July 2026.
> doi:10.48550/arXiv.2607.10491

**Verified against the arXiv record on 19 October 2026**, not taken on trust.
The abstract states that a "conflict-preserving Dempster-Shafer fusion rule
transfers unresolved disagreement into epistemic uncertainty rather than
normalizing it away", and that the generator is "routed to direct answering,
conflict-aware answering, or abstention according to the fused uncertainty
score". It is evaluated on CRAG, ConflictQA and MuSiQue.

So the claim *"separate missing evidence from contradictory evidence"* is no
longer available. ERAG does exactly that, publishes it, and routes on it.

### What survives

Checked against FLARE, CRAG, Self-RAG, Adaptive-RAG, ConflictScore, CARE and
ERAG. **Every one of them collapses to a scalar before deciding what to do:**

| system | decomposed representation? | decision variable |
|---|---|---|
| ERAG | **yes** (Dirichlet + DS fusion) | one fused uncertainty score |
| CRAG | no | one confidence degree from a fine-tuned T5 |
| FLARE | no | generator token probability — a property of the *generation*, not the evidence |
| Self-RAG | reflection tokens | critic-emitted tokens from a 7B LM |
| Adaptive-RAG | no | query-complexity classifier |
| ConflictScore / CARE | conflict only | one conflict score |

ERAG is the closest prior work and the gap is narrow: it separates the two
quantities in its representation and then **merges them into a single
`U_global`** for routing. The surviving distinction is therefore not *having* a
decomposed state but *acting* on the components separately.

### The narrowed claim, which is what Phase 3 tests

> Retrieval expansion conditioned on a **decomposed** evidence state, where
> **indeterminacy specifically** (not low confidence generally) triggers
> expansion while **contradiction triggers a different action**. Training-free,
> with no LLM in the gate.

This claim is falsifiable by RQ3 and the project is prepared to abandon it.

---

## 2. What was already known when this was registered

Pre-registration is worthless if it hides prior knowledge, so here is the
complete disclosure. The harness for Phases 2–5 was built in a previous session
and is committed at `4a268c1`, which precedes this file.

**Already measured, and informing the predictions below:**

- **RQ1's empirical half.** The dilution sweep has been run: pairwise `disagree`
  is flat at 1.000 across 0–4 filler passages (raw, unclamped 1.739 at every
  point) while mean-pooled decays 0.847 → 0.400. Phase 1 adds the *proof*, not
  the measurement.
- **The cost of the HotpotQA harness**, timed over 40 complete instances per
  condition: 9.98 s answerable, 18.08 s unanswerable.
- **Structural facts about HotpotQA validation** (`experiments/describe_hotpot.py`):
  gold is always exactly 2 paragraphs; 99.19% of instances carry 10 paragraphs;
  37 instances (0.50%) cannot form the unanswerable condition; an extractive
  reader caps Exact Match at 93.8% overall and 69.2% on comparison questions.
- **Two defects in `eval.py`**, found by running `risk_coverage` and
  `expected_calibration_error` for the first time (pinned in
  `tests/test_eval_calibration.py`). They are reported, not silently fixed.
- **That `r` is partly inert on HotpotQA.** `diversity` is constant because every
  retrieved paragraph has a distinct title, and `connectivity` is unavailable
  because no graph is built. Only `coverage` and `support_count` can vary — two
  of four components, against one of four on FEVER.
- **The FEVER result**, which is negative: under the current construction
  `argmax(T,I,F)` scores 0.6650 against the bare NLI classifier's 0.7550, a gap
  of −0.0900 with exact McNemar b=1, c=19, p=0.0000. The mechanism is the
  `I ≥ 1-r` floor overwriting correct NLI judgements on low-`r` rows.

**Not measured, and genuinely open at registration:**

- every quantity in RQ2 (no AUROC has been computed for `r` or any comparator);
- every quantity in RQ3 (the size of the recoverable-miss subset is unknown, as
  are all three conditions' recovery rates, and no expansion-rate calibration
  has been attempted);
- every quantity in RQ4 beyond parameter counts, which are public;
- all Track B answer metrics (EM, token-F1, faithfulness, citation accuracy,
  AURC, ECE, Brier) for all three systems.

---

## 3. Research questions, predictions and stop conditions

### RQ1 — Does pairwise conflict aggregation resist evidence dilution where mean-pooling does not?

**Prediction.** Pairwise aggregation is **invariant** to the addition of neutral
passages; mean-pooled aggregation **decays monotonically**.

**Status.** Already measured (see §2). Phase 1 supplies the two-direction proof.

**How it could fail.** If the proof does not go through in both directions, or if
the measured sweep is found to depend on clamping rather than on the operator.

**Scope limit, registered in advance.** This distinguishes the operator from
**mean-pooled NLI aggregation** — the standard in NLI-based RAG verification and
SURE-RAG's own weak baseline. It does **not** distinguish it from
Dempster-Shafer methods, where a vacuous mass is the neutral element of the
combination rule *by construction* and so dilution-invariance is automatic. Any
write-up claiming more than this is overclaiming.

### RQ2 — Does retrieval sufficiency `r` predict answerability better than raw retrieval score or coverage alone?

**Prediction.** `r` achieves a **higher AUROC** than both the raw maximum
retrieval score and `coverage` alone.

**Stop condition.** If `r` does not beat *both* baselines, `r` is not earning its
complexity and that is reported plainly. No reweighting of its four components
is permitted to rescue it — the weights are frozen.

**Known risk.** `diversity` is constant and `connectivity` unavailable on this
corpus, so `r` is effectively a two-component signal here. A null result is
therefore weaker evidence against the *design* than against its *value on this
corpus*, and will be reported with that caveat rather than as a general refutation.

### RQ3 — Does indeterminacy-triggered expansion recover answers that fixed-`k` retrieval misses, and does it beat expansion triggered by a scalar confidence at equal expansion budget?

**This is the question that decides the project.**

**Prediction.** Yes to both. The second half is the novelty claim.

**Design, fixed in advance.** Three conditions, identical retriever, identical
reader:

- **(a) FIXED-k** — no expansion.
- **(b) SCALAR-GATE** — expand when the NLI head's own maximum class probability
  falls below a threshold. This is the ablation that removes the decomposition
  while holding everything else constant.
- **(c) I-GATE** — expand when `I` dominates (the current policy).

**(b)'s threshold is set so its expansion *rate* matches (c)'s on the same
data.** Both rates are reported and their match confirmed. Choosing that
threshold to equalise rates is calibration; changing it afterwards to improve
either arm is tuning and is forbidden. The comparison is void if the budgets
differ.

**Reported:** recovery rate on the recoverable-miss subset; unnecessary-expansion
rate where top-`k` was already sufficient; end-to-end EM and token-F1; exact
McNemar for (c) vs (a) and (c) vs (b).

**Stop conditions, both registered before seeing data:**

1. **Power.** If the recoverable-miss subset is **under ~40 questions**, say so
   before proceeding; the comparison has no power and no verdict will be claimed
   from it.
2. **The gate.** **If (c) does not beat (b) at equal budget, STOP.** Conditioning
   on `I` specifically then adds nothing over conditioning on low confidence
   generally, the claim reduces to CRAG/FLARE territory, and Phases 4 and 5 are
   not run. Thresholds are not to be adjusted to rescue it.

### RQ4 — Does the gate match LLM-based gating quality at roughly 40× fewer parameters?

**Prediction.** The **cost advantage is certain** (it is arithmetic on published
parameter counts). **Quality parity is uncertain** and may well fail.

**Comparator, named in advance.** The NeutroRAG gate is a DeBERTa-v3-base NLI
head (~184M parameters). The ~40× figure is against a 7B-parameter LLM critic of
the kind Self-RAG uses (7B / 184M ≈ 38×). Exact counts will be verified from the
publications rather than asserted.

**Registered honesty constraint.** The cost comparison is **our measured cost
against their reported cost**. Their systems are not being run here. Any claim
phrased as though both were measured under the same conditions is false, and the
write-up must say which side is measured and which is reported.

---

## 4. Earlier negative results are retained, not removed

The rescope does **not** retract or delete the project's negative findings. They
stay in `RESULTS.md`, as motivation for the narrowing and as limitations:

- **FEVER failure-type attribution.** The headline 0.745 is retracted; the bare
  NLI classifier scores 0.7550 on identical calls. Under the current
  construction the layer scores 0.6650, i.e. −0.0900 and significantly *worse*
  (p=0.0000).
- **The r-ablation.** Computed `r` loses to a constant `r` on FEVER.
- **The CONFLICTS conflict and policy tests.** Both negative; the action fires
  more often on the wrong class.
- **Three retractions**: the 0.745 headline, the fuzzy-collapse ablation delta,
  and "T+I+F > 1 demonstrates independence".

Anyone reading this repository must be able to see both the old questions and the
new ones. Phase 3 is a narrowing of the claim after a literature check, not a
reinterpretation of results that did not support the original.

---

## 5. Analysis decisions fixed now, so they cannot be chosen later

- **Sample.** N = 400 HotpotQA distractor validation questions, stratified by
  question type, seeded, drawn only from instances where the unanswerable
  condition is constructible. Chosen because 400 fits the three-hour budget at
  the *upper* bound of the measured per-instance cost, not merely at the point
  estimate.
- **Frozen parameters.** `tau_T`, `tau_I`, `tau_F`, `tau_T_conflict`, `lambda`,
  the sufficiency weights and the saturation targets are not modified in any
  phase. The only threshold chosen during Phase 3 is (b)'s scalar cut-off, and
  only to equalise expansion rates.
- **Statistics.** Wald intervals for proportions; exact (binomial) McNemar for
  paired comparisons; Mann-Whitney U for distributions. Two-sided, α = 0.05.
- **`eval.py` defects.** AURC and ECE are run **as they stand**, and the defects
  are reported alongside the numbers rather than fixed mid-experiment. Where a
  defect affects a reported figure, the affected figure is marked.
- **Reader.** One extractive reader (`deepset/roberta-base-squad2`) in every
  condition, so no comparison is confounded by a change of generator. Its
  ceiling (93.8% EM overall, 69.2% on comparison) is reported with every EM
  figure.
- **Question-type breakdown.** Bridge vs comparison, with intervals. The
  comparison arm is expected to be underpowered at N = 400 (~80 instances, of
  which ~25 are yes/no and unreachable by a span reader); that is stated with
  the result rather than discovered afterwards.

---

## 6. What would falsify the narrowed claim

Stated plainly, so the answer cannot be reframed afterwards:

1. **(c) failing to beat (b) at equal expansion budget.** The decomposition then
   adds nothing over a scalar, and the claim is dead. This is the primary test.
2. **`r` failing to beat raw retrieval score and coverage on answerability.** The
   sufficiency signal then does not earn its complexity.
3. **Indeterminacy-triggered expansion failing to recover misses at all.** The
   mechanism then does not work even before the novelty question arises.

A result in which NeutroRAG loses on end-to-end EM while winning on (1) is
**not** a falsification of the narrowed claim, because the claim is about the
gating signal, not about beating a span reader on answer accuracy. That
distinction is registered here in advance precisely so it cannot be invented
later to excuse a bad EM number.

---

## Addendum 1 — a wiring bug in the Phase 3 harness, found before the verdict

**Added 19 October 2026, after a 40-question pilot and before the powered run.**
Appended rather than edited: nothing above this line has been changed.

### What happened

The Phase 3 harness built its scorer as `NLIRetrievalScorer(nli=nli)`, passing no
saturation targets and therefore silently inheriting the **library defaults,
`support_target = 1` and `diversity_target = 1`** — which are *FEVER's* targets,
derived from FEVER's p75 evidence depth of 1. HotpotQA's depth is 10.

Measured consequence, at the base depth `k = 3` and after full expansion
`k = 10`:

| saturation targets | r at k=3 | r at k=10 | `1-r` at k=3 |
|---|---:|---:|---:|
| (1, 1) — FEVER's, as run in the pilot | 0.907 | 0.942 | **0.093** |
| (10, 10) — HotpotQA's own depth p75 | 0.441 | 0.942 | **0.559** |

Because `I = max(1-r, n)`, a `1-r` pinned near 0.09 can never be the active
branch: on the pilot it was the active branch on **3 of 40 rows**, and `n` on the
other 37. **The retrieval-sufficiency signal was contributing nothing, and
condition (c) had degenerated into "expand when the NLI head's neutral class
dominates".** That is not the I-gate; it is a relabelling of one NLI class.

Reporting the pilot's verdict would therefore have been a **false negative
against a degenerate implementation** — concluding that the decomposition adds
nothing, on a configuration in which the decomposition was not running.

### Why correcting it is not tuning

Stated explicitly, because the correction follows a pilot in which (c) lost on
Exact Match (0.300 vs 0.325) and the appearance matters:

1. **Decision 2**, a standing project rule predating this phase, requires
   saturation targets to be per-corpus and derived from that corpus's **depth
   distribution only, never from accuracy**.
2. The depth derivation for HotpotQA was run and committed in
   `experiments/describe_hotpot.py` at commit **`4a268c1`**, which precedes this
   pre-registration (`2551e65`) and precedes any accuracy measurement. It
   printed `paragraph-count p75: 10` and `distinct-source p75: 10`.
3. So **10 was already derived, from depth alone, and already committed, before
   any result existed.** Passing (1, 1) was a wiring omission, not a choice.
4. No accuracy figure was consulted to select the corrected value, and no other
   frozen parameter is being changed.

The fix restores a pre-derived, pre-committed constant. It is a bug fix.

### What is retained

The voided pilot is **not deleted**. Its numbers are reported in `RESULTS.md`
labelled as a mis-wired configuration, so a reader can see both the bug and its
effect. The pilot had no power in any case — every McNemar p-value was 1.0000 at
n = 40.

### Fresh predictions, registered before the corrected run

With `r` actually varying (0.441 at k=3, rising to 0.942 at k=10), the I-gate is
no longer equivalent to tracking `n`, so the comparison becomes meaningful. In
advance:

- **(c) will beat (b) on recovering gold into context.** `1-r` is large exactly
  when few passages have been retrieved, which is when gold is most likely to be
  missing. Confidence: moderate.
- **Whether (c) beats (b) on end-to-end Exact Match is genuinely open, and I
  expect it may not.** The pilot showed every gated condition scoring at or
  below FIXED-k on EM (0.300–0.350 against 0.325) while recovering more gold into
  context. If that dissociation survives, the cause is the reader, not the gate:
  an extractive span reader given more passages has more distractors to choose
  between, so recovering the gold paragraph does not entail recovering the
  answer. Confidence: this dissociation persists, moderate-to-high.
- **`one_minus_n` remains the baseline that matters**, not the pre-registered
  `max(e, n, c)`. See the note registered in §5 of `RESULTS.md`: `max(e, n, c)`
  is *high* for a confidently-neutral passage set and so is structurally blind to
  the state the I-gate fires on, which makes it a weak baseline. If (c) cannot
  beat `one_minus_n`, the honest conclusion is that the gate reduces to
  monitoring neutrality — a scalar — and the claim must be restated accordingly.

**The stop condition is unchanged.** If (c) does not beat (b) at equal budget on
the corrected run, Phase 3 halts and Phases 4-5 are not run. This addendum buys
one re-run of a mis-wired harness, not a second chance at a result.

---

## Addendum 2 — the two target settings bracket the intended gate

**Added 19 October 2026, before the corrected run. Appended, not edited.**

Measured on 40 questions at the base depth k=3, scoring only (no accuracy, no
EM):

| saturation targets | median r | I-gate fires | dominant channel counts |
|---|---:|---:|---|
| (1, 1) — FEVER's | 0.929 | 52.5% | F:15, T:4, I:21 |
| (10, 10) — HotpotQA's depth p75 | 0.463 | 75.0% | F:10, **T:0**, I:30 |

Under the corrected, depth-derived targets **`T` is structurally eliminated at
base depth.** This is algebra, not an empirical accident: `T = r·e ≤ r = 0.463`
while `I = max(1-r, n) ≥ 1-r = 0.537`, so `T < I` for every possible NLI output.
The base stage sits inside the memo F12 regime (`r < 0.5 ⇒ I ≥ 1-r > r ≥ T`), and
the I-gate there reduces to *"fire unless conflict dominates"*.

So the two settings bracket the gate the construction intends, and neither is it:

- at **(1, 1)** all three channels compete but `1-r ≈ 0.07`, so retrieval
  sufficiency contributes nothing and the gate is the NLI neutral class
  relabelled;
- at **(10, 10)** sufficiency contributes strongly but `T` cannot win, so the
  gate cannot distinguish "well supported" from "insufficiently retrieved".

**This is a structural property of `I = max(1-r, n)`, registered here before the
verdict so it cannot be offered afterwards as an excuse.** The construction
requires `r > 0.5` with real variance for all three channels to compete while
`r` still matters; a target derived from corpus depth does not guarantee that,
and on HotpotQA it does not deliver it.

**Decision: the corrected run uses (10, 10).** Decision 2 mandates
depth-derived, per-corpus targets and that value was derived and committed
before any result; choosing (1, 1) instead *because* it produces a livelier gate
would be selecting a parameter on the basis of its effect, which is the thing
the rule forbids. The 40-question (1, 1) pilot is reported alongside as the
voided comparison.

A firing rate of 75% is high but not degenerate, so the equal-budget
calibration remains well posed: the scalar gate is calibrated to the same 75%.
Had the rate been ~100% the comparison would have been void and this phase
would have halted here instead.

---

## Addendum 3 — outcome, and the line between what was registered and what was not

**Added 19 October 2026, after all experiments were complete. Appended, not
edited: nothing above this line has been changed at any point.**

All figures below are regenerated by the scripts named in
[`results/claims.md`](results/claims.md), which recomputes them from the
committed artefacts.

### (a) The pre-registered stop condition fired

RQ3's registered condition was that condition **(c) I-GATE** must beat condition
**(b) SCALAR-GATE** on end-to-end accuracy at equal expansion budget. It did
not.

| | pooled EM | token-F1 | EM on recoverable-miss |
|---|---:|---:|---:|
| (a) FIXED-k | 0.2675 | 0.3636 | 0.1486 |
| **(c) I-GATE** | **0.3025** | 0.4032 | 0.2432 |
| (b) `max(e,n,c)` — the registered scalar | **0.3225** | 0.4247 | 0.2658 |
| (b) `1-n` | 0.3075 | 0.4086 | 0.2658 |
| (b) `e` | 0.3100 | 0.4066 | 0.2703 |

- Primary comparison: **I-gate 0.3025 against 0.3225**, difference −0.0200,
  exact McNemar b=14, c=22, **p=0.2430**. Not significant, and in the wrong
  direction. The registered condition was that (c) must *beat* (b).
- Expansion rate matched **exactly** as the design requires: 0.7775, 311/400,
  for every gated condition.
- The power condition passed: the recoverable-miss subset was **222/400**
  against a registered floor of ~40.
- **RQ3's first half did succeed**: expansion beats no expansion on the
  recoverable-miss subset, EM 0.1486 → 0.2432, exact McNemar b=30, c=9,
  **p=0.0011**. That is the one significant positive result in Phase 3, and it
  reproduces FLARE/CRAG/Adaptive-RAG in a training-free setting rather than
  extending them.

**Phases 4 and 5 were not run**, as the stop condition requires. No threshold,
weight, lambda or saturation target was altered after the verdict.

### (b) The Addendum 2 prediction was FALSIFIED

Addendum 2 registered: *"(c) will beat (b) on recovering gold into context."*

It did not. **The I-gate recovered the least gold of all four gated
conditions**, and the registered scalar outperformed it on three independent
axes at once:

| | gold recovered into context | pooled EM | comparison EM | fires less where expansion hurts |
|---|---:|---:|---:|---:|
| **(c) I-GATE** | **0.7027** (worst) | 0.3025 | 0.1139 | **+0.0249** (fires *more*) |
| (b) `max(e,n,c)` | 0.7342 | **0.3225** | **0.1392** | **−0.1486** |

Stated plainly, as required: **`max(e,n,c)` — the scalar this document had
flagged in advance as the *weak* baseline, on the grounds that it is
structurally blind to a confidently-neutral evidence set — outperformed the
decomposed I-gate on pooled EM, on comparison EM, and on firing selectivity.**
It was the only gate of the four that correctly fired *less* on the question
type where expansion degrades accuracy.

The decomposition is also not merely untested against it. Decision overlap is
**225/311 (72.3%)** against a pigeonhole floor of 222/311 (71.4%), so the two
triggers select close to maximally different questions given the matched rate;
on the 172 discordant questions the I-gate still scores lower, 0.2674 against
0.2907.

---

## POST-HOC, EXPLORATORY — NOT PRE-REGISTERED

### (c) Everything from here was chosen after seeing the null

The analyses in this section were selected *because* Phase 3 returned a null
and the cause needed finding. They carry **no inferential protection**: the
hypotheses were generated from the same data that tests them, the choice of
which splits and which components to examine was made after seeing the result,
and no p-value here is corrected for that selection. They are reported as
**hypotheses for a successor study**, not as confirmed findings. They happen to
be the most interesting output of the project, which is exactly why the
distinction has to be stated rather than left implicit.

**The by-type split.** Expansion helps bridge questions (0.293 → 0.349–0.368)
and **degrades comparison questions below fixed-k** (0.165 → 0.101–0.139),
uniformly across all four gated conditions. On comparison questions alone,
fixed-k beats every gated condition; each test is individually null
(p = 0.18–0.69) but `c > b` in all four, totalling b=9 against c=24. n=79, so
direction only.

**The r-component diagnostics.** `connectivity` is dropped entirely (no graph is
built), and of the three remaining components **two have exactly zero
variance** — `support_count` and `diversity` are both pinned at 0.3000, one
distinct value each across 400 questions. Therefore

&nbsp;&nbsp;&nbsp;&nbsp;`r = 0.3333 · coverage + 0.2000`

with max residual **1.11e-16** over 400 questions, confining `r` to
**[0.2000, 0.5333]**. Composition divides coverage's variance by nine
(0.023640 → 0.002627) while preserving its rank order exactly, so `r` has
**identical** Spearman rho to `coverage`: 0.2232 pooled (p<0.0001) and 0.4338 on
comparison (p=0.0001). The raw retriever score has rho 0.0196 (p=0.6956).

Two consequences, one negative and one positive:

- `r`'s four-component construction contributes nothing on this corpus beyond a
  single lexical-overlap feature, rescaled into a range that cannot reach the
  decision boundary.
- **`coverage` significantly outpredicts the retriever's own relevance score**
  for whether retrieval succeeded. That is a real positive finding and the only
  one favouring the sufficiency idea.

**The depth and circularity analysis.** At full depth both count-based
components saturate, so `r = coverage/3 + 0.6667` there against
`coverage/3 + 0.2000` at k=3 — making **+0.4667 of every expansion's gain in `r`
mechanical**. Backing the components out of the observed `r` confirms it:
implied diversity is 1.0 with max deviation **1.478e-06**, which is exactly the
6-decimal rounding of `r` carried through a factor of three.

The consequence is a feedback loop. `final_k` alone explains
**eta² = 0.9144** of the I-gate's variance in `dr` (0.8739, 0.7562 and 0.6384
for the three scalars — the gate that reads `r` is the most depth-determined of
the four). The fraction of questions with `r > 0.5` goes from **59/400 (14.8%)**
at base depth to **321–324/400 (80–81%)** after expanding. Since
`T = r·e ≤ r` and `I ≥ 1-r`, the gate `I > T` is therefore pushed toward true
before expanding and toward false after it, **by arithmetic on the retrieval
depth the gate itself chose**, not by anything about the evidence.

**One correlation that must not be read as evidence.** `dr` does correlate with
whether expansion fetched the gold (rho 0.436 for the I-gate, p<0.0001), but
this is confounded beyond repair in this data: k=10 retrieves *every* paragraph
of a HotpotQA distractor instance, so `gold_in_context` is **1.0000 at k=10 by
construction**, and `dr` is near-constant within each depth. The chain is
`dr → depth → helped`. It is not evidence that `r` measures sufficiency.

### (d) Claims withdrawn or corrected during the diagnosis

Three. Each is listed with what was said, why it was wrong, and the commit that
fixed it.

**1. "`r` is not measuring retrieval adequacy in any way that tracks retrieval
success."**
*Wrong because* it conflated level with rank. `r` does rank-correlate with
Recall@3 — rho 0.2232 pooled, p<0.0001 — and significantly more strongly within
comparison questions (0.4338). What `r` lacks is **range**, not signal: its
absolute level barely differs between question types (0.4562 vs 0.4483) while
its ordering within a type is informative. The overstated sentence would have
supported a stronger negative conclusion than the data allow.
*Withdrawn at source in* **`c339b98`**.

**2. The 40-question pilot's verdict, that the I-gate lost (EM 0.300 vs 0.325).**
*Wrong because* the harness had silently inherited FEVER's saturation targets
(1, 1) instead of HotpotQA's depth-derived (10, 10). With `1-r ≈ 0.093` pinned
near zero, it was the active branch of `I = max(1-r, n)` on only 3 of 40 rows:
the retrieval-sufficiency signal was contributing nothing and condition (c) had
degenerated into "expand when the NLI neutral class dominates". Reporting that
verdict would have been a **false negative against an implementation in which
the decomposition was not running**.
*Declared in Addendum 1 before the re-run, fixed in* **`e19bc4f`**. The voided
pilot is retained as `results/hotpot_gate_pilot40_VOID_wrong_targets.csv`.

**3. The first implied-diversity back-substitution, which appeared to refute the
saturation assumption (283/388 implied values above 1.0).**
*Wrong because* it computed coverage on the question alone, while the gate had
scored sufficiency on `question + reader_answer`. On the correct hypothesis
string the implied value is 1.0 to within 1.5e-6. The apparent refutation was an
artefact of the diagnostic, not a property of the data.
*Corrected in* **`e1c03b6`**.

A fourth item is a limitation rather than a withdrawal, and is recorded because
it constrains any re-analysis: `results/hotpot_gate.csv` stores only the depth
each condition **selected**, not an answer at every candidate depth. All three
depths exist for 113/400 questions and k=6 for only 124/400, so an arbitrary
gate rule **cannot** be replayed offline without re-running the model pass.
Pass 2 computed every stage in memory and the script discarded all but the
final one.

### (e) What a successor construction would need

Four requirements, each derived from a specific measurement above rather than
from preference:

1. **A saturation target strictly below the candidate pool size.** With
   `target = 10` and a pool of 10, `support_count` and `diversity` reach exactly
   1.0 at full depth and are pinned at `k/target` everywhere else — zero
   variance at any fixed `k`, for any target. The target must sit below the pool
   so the component can distinguish states within it.
2. **Depth-invariant or depth-normalised sufficiency terms.** Any component of
   the form `min(1, count/target)` is a monotone function of how many passages
   were fetched. When the gate's own action changes that count, the gate reads
   back its own decision — measured here at eta² = 0.9144. A sufficiency term
   must be normalised by the number of passages considered, or measure something
   other than their number.
3. **`r` kept above 0.5 with real variance.** `T = r·e ≤ r` and `I ≥ 1-r`, so
   at `r < 0.5` indeterminacy beats truth for *every possible* NLI output and
   "well supported, do not expand" becomes unrepresentable. It was unrepresentable
   on **82% of this sample**. Either `r` must be kept in a range where all three
   channels can win, or `I` must stop taking a max against `1-r`.
4. **An outcome label that still varies at maximum depth.** On HotpotQA
   distractor, k=10 retrieves every paragraph, so both `gold_in_context` and
   `Recall@k` are **degenerate at full depth** — the former is identically 1.0.
   Any evaluation of a recalibrated `r` needs a label that still discriminates
   when the whole candidate pool has been retrieved, which rules out both.

A construction satisfying all four has **not** been tested. The negative result
recorded here is a verdict on this construction at this operating point, not on
decomposed gating in general — and the asymmetry is worth stating plainly,
because requirement 3 was violated on 82% of the sample before any evidence was
consulted.

---

## Addendum 4 — pre-registering the comparison-question finding for confirmation

**Registered 19 October 2026, BEFORE the sample was drawn and before any
measurement on it. Appended, not edited.** Nothing above this line has changed
at any point. The commit carrying this addendum precedes the first commit of
the experiment it describes; that ordering is the evidence that this is a
registration and not a write-up.

### Motivation

Phase 3's **post-hoc** by-type analysis found every gated condition scoring
*below* fixed k=3 retrieval on comparison questions. Verified from
`results/hotpot_gate.csv`:

| condition | comparison EM | Δ vs fixed-k | b | c | discordant | discordant % |
|---|---:|---:|---:|---:|---:|---:|
| (a) fixed k=3 | **0.1646** | — | — | — | — | — |
| (b) `max(e,n,c)` | 0.1392 | −0.0253 | 2 | 4 | 6 | 7.59% |
| (c) **I-GATE** | 0.1139 | **−0.0506** | 2 | 6 | 8 | 10.13% |
| (b) `e` | 0.1139 | −0.0506 | 3 | 7 | 10 | 12.66% |
| (b) `1-n` | 0.1013 | **−0.0633** | 2 | 7 | 9 | 11.39% |

*(b = gated condition right where fixed-k wrong; c = the reverse.)* Totals
across the four: **b=9 against c=24**. All four fell in the same direction and
**all four tests were null at n=79** (p = 0.18–0.69).

That analysis was exploratory: the split was chosen after seeing the Phase 3
null, and `claims.md` marks it `NO-POST-HOC`. **This addendum promotes it to a
registered hypothesis and tests it on a fresh, disjoint sample.**

### Hypothesis H1 — directional, registered in advance

> **On HotpotQA comparison questions, fixed k=3 retrieval achieves HIGHER
> exact match than indeterminacy-gated adaptive expansion.**

The direction is registered **because it was already observed** in Phase 3.
This is therefore a **one-sided confirmation test of a stated direction**, not a
two-sided exploration, and the one-sidedness is declared here rather than
chosen after seeing the new data.

### Mechanism, stated in advance and not fitted afterwards

Two measured facts from Phase 3 predict H1 without reference to the new data:

1. **Comparison questions already retrieve well.** Recall@3 = **0.8734** against
   bridge's **0.6589**. Expansion therefore adds distractor passages to a
   context that already contains the gold, and an extractive reader given more
   distractors has more wrong spans to choose between.
2. **No gate can decline to expand on them.** `r = 0.3333·coverage + 0.2000` is
   capped at **0.5333** and sits below 0.5 on 82% of questions, so
   `T = r·e ≤ r < 1-r ≤ I` and *"well supported, do not expand"* is
   **unrepresentable**. The I-gate consequently fired on comparison questions at
   0.7975, *more* often than on bridge (0.7726).

If H1 holds on fresh data, the mechanism above is the reason offered; the test
does not itself discriminate between that mechanism and others.

### Design

**Sample.** 600 comparison-type questions from HotpotQA distractor validation,
**disjoint from all 400 qids used in Phase 3**. Disjointness is asserted in code
and the run **fails loudly** if it does not hold. Availability confirmed before
registration, by parsing only: 1487 comparison questions exist, 1408 are
disjoint from Phase 3, and **1384** are disjoint *and* pass the existing
`eligible_for_unanswerable` sampling filter, so 600 is drawable with margin.

**Seed: `20261119`** — fixed here, distinct from Phase 3's `20261019`.

**Conditions: the same five as Phase 3, unchanged** — `a_fixed_k`,
`b_scalar_entail`, `b_scalar_max_enc`, `b_scalar_one_minus_n`, `c_i_gate`. Same
depth ladder {3, 6, 10}, same saturation targets (support=diversity=10), same
retriever (`all-MiniLM-L6-v2`), same reader (`deepset/roberta-base-squad2`),
same equal-budget calibration procedure (each scalar threshold set to the
quantile of its own score distribution on *this* sample that reproduces the
I-gate's firing rate — in-sample, exactly as in Phase 3). **Nothing changes
except the sample.** Changing two things at once would make the result
uninterpretable against Phase 3.

**Primary test.** Exact McNemar, `a_fixed_k` vs `c_i_gate`, **one-sided in the
registered direction**. `c_i_gate` is primary **because it is the project's own
construction**, not because it showed the largest Phase 3 effect — it did not.
`1-n` showed the largest at −0.0633; the I-gate is mid-range at −0.0506, tied
with `e`.

A clarification registered now so it cannot be confused later: this primary
comparison is **not** an equal-budget comparison. `a_fixed_k` never expands, so
its expansion budget is zero. H1 asks whether expanding *at all* hurts on
comparison questions, which is a different question from Phase 3's primary
(I-gate vs scalar at matched budget). Both are reported.

**Secondary tests.** The same one-sided McNemar for the other three gated
conditions against `a_fixed_k`, **Holm-corrected across those three** at family
α = 0.05.

**Uniformity claim, pre-specified.** That all four gated conditions fall below
fixed-k in EM. This is reported as a **descriptive** count plus the pooled b/c
totals, **not** as a p-value: the four conditions share a reader, a retriever
and most of their decisions, so they are heavily dependent and a combined test
would overstate the evidence.

**Pre-specified robustness check.** The primary test repeated on
**span-answerable questions only**. Yes/no answers cap extractive EM at zero and
contribute only concordant ties, which dilutes the discordant count without
carrying information. ~30.9% of the eligible comparison pool is yes/no, so
roughly **415 of 600** are expected to be span-answerable.

**Diagnostics logged for replication on fresh data** (not hypothesis tests): the
four firing rates, the `r` distribution at base depth, and whether
`r = 0.3333·coverage + 0.2000` still holds with its residual.

**Harness change, logging only.** Phase 3 recorded only the depth each condition
selected, so no gate rule could be replayed offline. This run records answer,
EM, F1, `gold_in_context`, `r`, `T`, `I`, `F` at **every** candidate depth. This
adds logging and changes no measurement.

### Power

Phase 3 gave **8 discordant pairs of 79** for I-gate vs fixed-k (**10.13%**),
split b=2, c=6. Projected to n=600 — arithmetic verified before registration:

| assumption | discordant | b | c | z | one-sided p |
|---|---:|---:|---:|---:|---:|
| observed share, b/(b+c) = 0.25 | ~60.8 | ~15.2 | ~45.6 | **3.90** | <0.0001 |
| halved for winner's curse, 0.375 | ~60.8 | ~22.8 | ~38.0 | **1.95** | **0.0257** |

The effect was selected *after* being observed, so it is probably inflated;
the honest planning figure is the second row. Significance at one-sided 0.05
requires the gated condition to win fewer than **39.4%** of discordant pairs.
Phase 3 observed **25%**. **n=600 is adequate under both assumptions**, and
under-powered only if the true share exceeds ~0.39, i.e. if the Phase 3 effect
was more than half winner's curse.

### Reporting rule — binding

- The outcome is reported in `RESULTS.md` and in the report **whether H1 is
  supported, null, or reversed**.
- **No interim analysis at any point.** No accuracy, EM, F1 or test statistic is
  computed or printed until the run is complete. The run writes per-question
  rows; the analysis is a separate pass invoked once.
- If wall-clock forces truncation, the analysis runs **once** on all completed
  questions. Truncation is **by time only** and never by inspecting results, and
  the **realised n is reported**.
- No threshold, weight, lambda or saturation target is adjusted at any point.

### What would falsify H1

Either fixed k=3 **not exceeding** the I-gate on comparison-question exact
match, or the direction **reversing**. Both outcomes are reported as
falsification of a registered directional hypothesis, which is a stronger
result than the original exploratory finding and would retract it.

### Amendment to Addendum 4 — 19 October 2026, recorded before the run

**Dropped from this run:** the per-depth logging registered above, i.e. the
recording of answer, EM, F1, `gold_in_context`, `r`, `T`, `I` and `F` at *every*
candidate depth. **Only the depth each condition selects is logged, as in
Phase 3.** *Why:* scoring every depth unconditionally removes Phase 3's
depth-skipping and raises the wall-clock estimate from ~3.2 h to ~5 h, and H1
needs only final-depth exact match per condition. *Scope:* logging only — it
alters no measurement, no condition, no sample, no threshold and no test, and
the primary, secondary, robustness and uniformity analyses are unchanged.
*Recorded before the run rather than after,* because an unrecorded deviation
from a registered protocol is the failure mode this registration exists to
prevent. Consequence accepted: as with Phase 3, an arbitrary gate rule cannot
be replayed offline from this run's output.

---

## Addendum 5 — testing the decomposition under conditions where it can express the judgement

**Registered 19 October 2026, BEFORE the sample was drawn and before any
measurement on it. Appended, not edited.** The commit carrying this addendum
contains it and nothing else, and precedes the first commit of the experiment it
describes.

### Motivation

The Phase 3 refutation tested a **structurally crippled** gate. The diagnosis
established that `r` was confined to **[0.2000, 0.5333]**, so
`T = r·e ≤ r < 1-r ≤ I` on every question: *"well supported, do not expand"* was
**unrepresentable**, and the gate could not decline. Addendum 3(e) named four
requirements for a successor. This registers one construction that meets them
and tests the decomposition **for the first time under conditions where it can
express the judgement an expansion gate needs**.

### The repair — `r_v2`, derived from the diagnosis, not tuned

&nbsp;&nbsp;&nbsp;&nbsp;**`r_v2 = (coverage + support_frac) / 2`**

&nbsp;&nbsp;&nbsp;&nbsp;**`support_frac = |{passages with relevance ≥ floor}| / k`**

Three changes, each tied to a registered requirement:

**(a) `support_frac` is a fraction of what was actually retrieved**, not a count
against a fixed target. It is therefore **depth-invariant in form**, closing the
circularity — Addendum 3 requirement 2.

**(b) `diversity` is dropped, not retargeted.** On HotpotQA every paragraph is a
distinct article, so distinct-sources/k is identically 1.0 at every depth and
cannot carry information on this corpus; averaging it in only dilutes.
Retargeting would not help, because the quantity is constant for any target.
`connectivity` stays dropped — no graph is built.

**(c) The relevance floor is reset so that it binds.** The existing floor of 0.3
**never bound on any of the 1200 base-depth passages**: `DenseRetriever` remaps
cosine from [−1, 1] to [0, 1], so `relevance ≥ 0.3` means `cos ≥ −0.4`.

### The floor, fixed now from committed data

Derived by `python experiments/derive_relevance_floor.py`, which reads **no
accuracy, no label and no gate decision** — a corpus statistic in the same
category as `describe_hotpot.py` for Decision 2. Reconstructed the Phase 3
sample from its committed seed (verified: 400/400 qids match) and re-retrieved.

Base depth k=3, **1200 passages**:

| min | Q1 | **median** | Q3 | max | mean | sd |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5645 | 0.7244 | **0.7706** | 0.8238 | 0.9433 | 0.7724 | 0.0697 |

&nbsp;&nbsp;&nbsp;&nbsp;**`RELEVANCE_FLOOR_V2 = 0.770580`** — FIXED, not to be adjusted afterwards for any reason.

The old floor of 0.3 was cleared by **100.00%** of those passages; this one by
exactly **50.00%**, by construction. The base-depth median is used rather than
the all-paragraph median (0.6905) because Addendum 5 requires `support_frac` to
*start* near 0.5 with variance in both directions, and base depth is where it is
first evaluated.

### Depth ladder

**Base k=3, single expansion to k=5.** Maximum depth 5 against a 10-paragraph
pool, so the pool is **never exhausted** and `gold_in_context` still varies at
maximum depth (Recall@5 = 0.8137) — Addendum 3 requirement 4. The saturation
targets are not used by `r_v2` at all.

### Manipulation checks — VOIDING conditions, not results

Confirmed **before** the primary test is interpreted. If either fails, the
repair did not take and the primary test is **VOID, not negative**.

**(i) `r_v2 > 0.5` on at least 40% of questions at base depth.** This is the
condition that makes `T` reachable.

**(ii) η² of Δ`r_v2` from `final_k` below 0.30.**

**A defect in check (ii) as first drafted, corrected here before the run.**
Phase 3's η² = 0.9144 was computed over *expanders only*, stratified across two
expansion depths {6, 10}. With a **single-rung** ladder there is only one
expansion depth among expanders, so that form has one stratum, zero
between-group variance, and would **pass vacuously**. Reporting a vacuous pass
as evidence the repair took would be exactly the failure this registration
exists to prevent. So check (ii) is defined precisely as:

> **η² of Δ`r_v2` from `final_k` computed over ALL questions**, with
> `final_k ∈ {3, 5}` — non-expanders contributing Δ`r_v2` = 0 exactly. This asks
> "does knowing the final depth tell you how much `r_v2` moved?" and is
> non-degenerate with a two-rung ladder. Threshold: **< 0.30**.

Because that is not numerically comparable to Phase 3's 0.9144, a **scale-free
statistic that is** comparable is reported alongside, computed identically in
both: **|mean Δr| / within-depth sd**. Phase 3's I-gate: **11.40**.

### Hypothesis H2 — directional, registered in advance

> With `T` reachable, **the I-gate beats the best scalar-confidence gate on
> pooled exact match at equal expansion budget.**

### Design

- **400 questions, stratified by question type**, disjoint from **both** the
  Phase 3 400 **and** the H1 600. Disjointness asserted against both unions;
  the run aborts on any overlap. Availability confirmed before registration by
  parsing only: **6368** instances are disjoint from both and pass the existing
  eligibility filter (5584 bridge, 784 comparison).
- **Seed: `20261219`** — fixed here, distinct from `20261019` (Phase 3) and
  `20261119` (H1).
- Same five conditions, same equal-budget calibration, same retriever, same
  reader. **The only changes are `r_v2` and the depth ladder**, both specified
  above.
- **PRIMARY:** pooled EM, `c_i_gate` vs `b_scalar_max_enc`, exact McNemar,
  **one-sided** in the predicted direction, two-sided reported alongside.
  `max_enc` is primary because it is the condition that **beat** the I-gate in
  Phase 3 — the comparison that produced the stop condition.
- **SECONDARY:** the other two scalars, Holm-corrected across the two.
- **MECHANISTIC SECONDARY — the sharpest prediction.** The I-gate's firing-rate
  differential by question type should now be **negative**: firing *less* on
  comparison questions, which retrieve at Recall@3 = 0.8734. Phase 3 gave
  **+0.0249** with `T` unreachable. If the repair works, this flips sign.

### Projections from committed Phase 3 data — labelled as projections

Computed on the **Phase 3** corpus, which is *not* the sample H2 will be tested
on, and involving no accuracy. Recorded so the expected behaviour is on the
page before the run:

| quantity | old `r` | projected `r_v2` |
|---|---:|---:|
| mean at base depth | 0.4546 | **0.6103** |
| variance at base depth | 0.001981 | **0.037169** (19× more) |
| fraction > 0.5 at base depth | 14.8% | **67.8%** |
| mean Δ on expanding | **+0.4667** | **−0.0427** |
| \|mean Δr\| / within-depth sd | **11.40** | **0.527** (21× better) |
| Δ sign | positive on ~100% | negative 247/400, **positive 90/400** |

So check (i) projects to pass comfortably (67.8% against a 40% threshold), and
the depth dependence projects to fall by a factor of ~21 while changing sign and
becoming bidirectional.

**An honest note on ordering.** The floor was fixed by the registered rule — the
median of the base-depth relevances — and only then were these projections
computed. The floor is **not** adjusted in light of them, and would not have
been had they looked unfavourable. They are reported for transparency, not as
justification.

**A risk registered in advance.** `support_frac` *falls* with depth on this
corpus (0.5000 at k=3 to 0.3480 at k=5), because deeper passages are less
relevant, while `coverage` rises (0.7207 to 0.7874) and partly offsets it. Net
Δ`r_v2` is small and bidirectional. But this is a **new** depth dependence in
the opposite direction, and if it is stronger on the fresh sample than projected
here, check (ii) is the guard that should catch it. It is registered as a
foreseen risk rather than discovered afterwards.

### Reporting rule — binding

- Reported whether H2 is **supported, null, void or reversed**.
- **No interim analysis.** The `--no-analysis` path from the H1 harness is
  reused, so no accuracy is computable during the run.
- Truncation by wall-clock **only**, never by inspecting results; analysed once;
  realised n reported.
- No threshold, weight, lambda or floor adjusted at any point.

### What would falsify H2

The I-gate failing to exceed `max_enc` on pooled exact match, or the direction
reversing. A **void** outcome — either manipulation check failing — is reported
as void and is **not** evidence either way about the decomposition.

---

## Addendum 6 — does `r` predict answer correctness once retrieval binds?

**Registered 20 October 2026, 21:25 IST, BEFORE the sample was drawn and before
any measurement on it. Appended, not edited.**

### Motivation

Report 7.7.4 concludes that the decomposition fails because **`r` predicts
retrieval adequacy while answer correctness is decoupled from it**: expansion
put gold in context on **70–82%** of recoverable misses while EM on those
questions reached only **0.24–0.27**. Retrieval was not the binding constraint
in this pipeline.

That is falsifiable. **Make retrieval the binding constraint and `r` should
begin to predict answer correctness.**

### The registered selection rule is AMENDED, before any accuracy exists

Stage 1a swept the pooled-index size, building one dense index over the
paragraphs of N questions and retrieving from it instead of the 10 supplied per
question (`experiments/sweep_pool_size.py`, superset seed 20261020):

| N | pool size | Recall@3 | Recall@10 | ratio | original rule met |
|---:|---:|---:|---:|---:|---|
| 250 | 2,488 | 0.6600 | 0.8400 | 0.786 | no |
| 500 | 4,946 | 0.6690 | 0.8230 | 0.813 | no |
| 1000 | 9,787 | 0.6375 | 0.7940 | 0.803 | no |
| **2000** | **19,294** | **0.6100** | **0.7592** | **0.803** | no |

**The registered proxy `Recall@3 ≤ 0.6 · Recall@10` is unsatisfiable at any pool
size.** `Recall@3 / Recall@10` measures the **retriever's ranking quality**, not
whether retrieval binds: the ratio moved only **0.786 → 0.803** across an **8×**
change in N, because pooling degrades all depths proportionally. **The proxy was
wrong, not the pool.**

It is replaced by a direct criterion. **N is set to the LARGEST swept value,
2000 (19,294 passages)**, giving **Recall@10 = 0.7592** and
**Recall@3 = 0.6100**. Retrieval then binds most tightly of any configuration
measured: **24% of questions have gold unreachable at maximum depth**, against
**0%** in the distractor setting, where Recall@10 was 1.0 by construction. The
endpoint is taken **deliberately**, so that no N could have been chosen to suit
an outcome.

**This amendment follows a retrieval-only sweep that read no accuracy, no label
beyond the published gold supporting facts, and no gate decision. It is recorded
before the run.**

**Why base depth stays at k=3.** `k=1` satisfies the *original* rule at every N
(ratio 0.476–0.518). It is rejected anyway: at one passage `support_count` and
`diversity` both pin to 0.1, legacy `r` collapses to `coverage/3 + 0.0667`, and
`coverage` becomes a one-paragraph sample. The primary compares against AUROC
values computed on **three** passages, so changing base depth would destroy the
very comparison that is the point of the experiment.

### PRIMARY — two-sided, registered

> Under hard retrieval, **AUROC of LEGACY `r` for ANSWER CORRECTNESS** (`em` at
> base depth) is **above 0.5**.

Reported with a **95% CI** and whether the CI **excludes 0.5**. Reference values
under easy retrieval, both **legacy `r` on three passages**:

| sample | AUROC of legacy `r` for `em` | 95% CI |
|---|---:|---|
| Phase 3, n=400 | 0.5483 | [0.4840, 0.6127] |
| H1, n=600 | 0.4689 | [0.4073, 0.5305] |

**LEGACY `r`, not `r_v2`.** The references are legacy-`r` figures and the
comparison must be like-for-like. `r_v2`'s AUROC is computed alongside as a free
secondary and is labelled explicitly as **NOT comparable** to those references.

### SECONDARY

- **AUROC of legacy `r` for retrieval success** (`gold_in_context` at base
  depth), to confirm it still works there. Reference: **0.5958**.
- **DeLong against `max_relevance`** for **both** labels. The embedder is loaded
  anyway, so this closes the gap Addendum 4's analysis left open, where
  `max_relevance` was unavailable for the H1 sample.
- **The five-condition gate comparison**: `a_fixed_k` plus the four gated
  conditions, exact McNemar, **I-gate vs `max(e,n,c)`**, as in Phase 3.

### Sample

- **400 questions, stratified**, **disjoint from the Phase 3 400, the H1 600 and
  the H2 400**. Asserted against all three CSVs, aborting on any overlap.
  Availability confirmed before registration by parsing only: **1634** of the
  swept 2000-question pool are disjoint from all three (1430 bridge, 204
  comparison).
- **Seed: `20261021`** — distinct from 20261019 (Phase 3), 20261119 (H1),
  20261219 (H2) and 20261020 (the sweep superset).
- Retrieval is from the **pooled 19,294-passage index**, never the 10 supplied
  paragraphs. **Asserted in code.** The queried 400 are a disjoint subset of the
  2000 whose paragraphs form that index, so the measured recalls above apply
  unchanged.
- **Ladder k=3 → 10. Legacy `r`. Everything else unchanged** from Phase 3.

### Reporting rule — binding

- Reported whether the primary is **positive, null or reversed**.
- **No interim analysis**: the `--no-analysis` path is used, so no accuracy is
  computable during the run.
- **Truncation by wall-clock ONLY**, never by inspecting results; analysed once
  on whatever completed; **realised n reported**. **If the run has not finished
  by 23:30 IST it is truncated.**
- No threshold, weight, lambda, target or floor adjusted at any point.

### What would falsify the prediction

AUROC of legacy `r` for `em` failing to exceed 0.5, or its 95% CI including 0.5.
That would mean making retrieval the binding constraint does **not** make `r`
predictive of answer correctness, and 7.7.4's diagnosis would need replacing
rather than extending.
