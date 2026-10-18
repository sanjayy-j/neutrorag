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
