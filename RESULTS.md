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
| **Decision 2** (saturation targets) | re-derived from depth alone and committed before any accuracy run (`4df956b`) |
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
| random draws, `r ~ Uniform(0,1)` | **2.497e-01** | 4.441e-16 |
| random draws, `r` computed by `sufficiency` | **2.467e-01** | 4.441e-16 |
| real run: pipeline corpus + graph | **1.364e-01** | 2.220e-16 |
| real run: 200 FEVER claims | **2.482e-01** | 2.220e-16 |

This matches the analytic bound exactly. Deviation is `-r·n` when `1-r > n` and
`-(1-n)(1-r)` otherwise; both are maximised at `n = 1-r`, where each equals
`r(1-r)` — i.e. **0.25 at r = 0.5**.

### The mass now carries independent information

| corpus | min | median | max | mass > 1 | mass < 1 |
|---|---:|---:|---:|---:|---:|
| pipeline (4 claims) | 0.958 | 1.059 | 1.368 | 75% | 25% |
| FEVER (200 claims) | 0.765 | 0.986 | 1.255 | 11.5% | 88.5% |

Under the old `I`, **all 200** FEVER rows had mass > 1, because "mass > 1" was
identically "disagree > 0". Under the new `I`, 23/200 do, and the two criteria
**disagree on 88.5% of rows**. The mass can now also fall *below* 1, which it
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
passage per claim: 185/200 rows carry exactly one, so with `support_target = 3`
the `support_count` component is pinned at 1/3, and with no graph and no
provenance `connectivity` and `diversity` are unavailable. `r` is therefore
bounded above by `(coverage + 0.333)/2 ≤ 0.667` on those rows **structurally**.
Median `r` = 0.417; 143/200 rows land at `r ≤ 0.5`, where `I = max(1-r, n) ≥ 0.5`
by construction. Condition (b) consequently predicts I nearly everywhere
(I-recall 0.970; F-recall collapses 0.853 → 0.162). No parameter was retuned.

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

I is the argmax for *every* claim at any `r ≤ 0.42`. FEVER's median `r` of 0.417
puts the whole corpus inside that regime.

Other sweep observations, none acted on: accuracy is highest at `r = 1.0`
(0.765) — `r` contributes nothing positive on FEVER, and the small edge over raw
`argmax(e,n,c)` (0.755) comes from `λ·disagree` in F. `lambda` is nearly flat for
attribution (0.455–0.470 over 0–2) but monotonically raises the
SURFACE_CONTRADICTION rate. `connectivity` and `diversity` weights are inert on
FEVER; at weight 1.0 for either, every *available* component has zero weight and
`r` falls to 0.

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
invariance, not clipping. `F` itself still drifts 0.812 → 0.692 because
`F = r·c + λ·disagree` and both `r` and the pooled `c` fall as filler is added;
only the `disagree` term was ever expected flat.

**F8 is fixed.** The flagship drug-trial case now returns **SURFACE_CONTRADICTION
with F = 0.812** (real NLI) against `tau_F = 0.40`, versus the memo's F = 0.347
and ABSTAIN. It still fires with 4 filler passages added (F = 0.692).

Mass identity still broken (2.497e-01). FEVER numbers **bit-identical** — with
one caveat: `fever_run_200.csv` stores the *pooled* triple per claim, so on
replay every passage carries the same values and pairwise ≡ pooled by
construction for **all** row sizes, not only the 185 single-passage ones.

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
below 0.5. FEVER pinned `r` low (median 0.417, 143/200 below 0.5); CONFLICTS
pins it high. **Neither venue exercises `r` across its range**, which is a
property of the components' targets, not of either dataset. Reported, not
retuned.

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

| prediction, committed in `4df956b` before any run | outcome |
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
python experiments/run_r_ablation.py     # replays fever_run_200.csv; model not re-run
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

The figure itself reproduces exactly, but it is **not attributable to the
neutrosophic layer**. Running `argmax` directly on the NLI head's own
distribution, using the *identical* model calls, scores **0.7550** — higher.

| system | accuracy | 95% CI (Wald) | macro-F1 |
|---|---:|---:|---:|
| NeutroRAG `argmax(T,I,F)` | 0.7450 | [0.685, 0.805] | 0.7476 |
| raw NLI `argmax(e,n,c)` | **0.7550** | [0.695, 0.815] | **0.7574** |
| difference | −0.0100 | | −0.0098 |

The two systems differ on **4 of 200** rows. Exact McNemar: b=1, c=3, **p = 0.625** —
the difference is not statistically significant in either direction. The honest
claim is therefore *"the neutrosophic construction produces no measurable benefit
over the bare classifier on this data,"* not *"it is worse."*

Corroboration: the model card for `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
reports **0.777** on the full FEVER-NLI dev set. Our 0.755 on a 200-row subset of
that same split is consistent with simply re-measuring the classifier.

*Source: `results/fever_200_metrics.md`, from `fever_run_200.csv`.*

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

### FEVER run (n = 200)

`copenlu/fever_gold_evidence` validation split, `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, CPU.

```bash
python experiments/run_fever_eval.py --limit 200 --csv fever_run_200.csv   # ~150s
python experiments/make_results_table.py
```

Full per-class table and both confusion matrices: **`results/fever_200_metrics.md`**.
Per-example dump: **`fever_run_200.csv`** (200 rows: claim, r, e, n, c, T, I, F,
gold, predicted, evidence_count).

| quantity | value | note |
|---|---|---|
| attribution accuracy `argmax(T,I,F)` | 0.7450 ± 0.0308 | not attributable to NeutroRAG (above) |
| baseline accuracy `argmax(e,n,c)` | 0.7550 ± 0.0304 | identical NLI calls |
| rows where predictions differ | 4 / 200 | |
| McNemar (exact) | b=1, c=3, p=0.625 | not significant |
| `N_empty` (memo F3) | **0 / 200** | the leak is live code but fired zero times here |
| distinct values of `r` (memo F1) | `{0.9}` | `r` is a constant, not a signal |
| `disagree` (memo F9) | median 0.011, mean 0.027, max 0.717; 9/200 rows > 0.1 | conflict term near-inert on FEVER |
| evidence passages per row | 185 rows have exactly 1 | so `2·min(e,c)` measures single-passage ambiguity, not cross-source conflict |
| mass identity deviation (memo F5) | 2.22e-16 | holds to machine precision |

> **`fever_run_200.csv` is now partly stale.** Its `claim`, `e`, `n`, `c`, `gold_*`
> and `evidence_count` columns remain valid (the NLI calls are unchanged), but its
> `r`, `T`, `I` and `F` columns were computed with the old constant `r = 0.9`.
> Regenerating it would produce a new headline number, which is downstream of the
> blocked stop-gate, so it has deliberately **not** been re-run.

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

### Pipeline demo — reproduces the report's Section 5(d) screenshot

```bash
python experiments/run_pipeline_demo.py --nli real
```

Reproduces the screenshot **exactly**: `TIF(T=0.223, I=0.603, F=0.347)`, identical at
steps 0, 1 and 2, final action **ABSTAIN**, `expansions_used=2`.

Two things this demonstrates, both of which belong in the report:

- **The flagship case false-negatives (memo F8).** The evidence set was hand-built to
  show SURFACE_CONTRADICTION — one passage reporting a 30% risk reduction, one
  reporting no significant reduction — and F = 0.347 falls below `tau_F` = 0.40, so
  the system abstains instead. The one example built to showcase the core action does
  not trigger it.
- **The expansion loop is inert here because the retriever is exhausted, not because
  the loop is broken.** The demo's toy corpus (3 graph facts, 3 documents) returns the
  same passages at every hop radius, so the triple cannot move. `tests/test_memo_regressions.py::test_expansion_with_a_real_scorer_changes_the_triple`
  drives the same policy with a retriever that *does* return new passages, and the
  triple changes — that test passes.

The default (mock NLI) invocation gives a *different* action —
`TIF(T=0.434, I=0.323, F=0.486)` → SURFACE_CONTRADICTION — so the two backends
disagree on the flagship example. Quote the `--nli real` run, and say which one you ran.

### Library behaviour (offline, mock NLI)

| quantity | value | command |
|---|---|---|
| README quickstart triple | `TIF(T=0.082, I=0.100, F=0.909)` | `python experiments/reproduce_all.py` |
| README quickstart, with `expand_fn` | ANSWER after 1 expansion | same |
| README quickstart, without `expand_fn` | ABSTAIN, 0 expansions | same |
| `MockNLIBackend.predict` sums to 1 | max \|sum − 1\| = 0.00e+00 | `pytest tests/test_memo_regressions.py -k distribution` |

### demo.py — illustrative only, not a research result

`demo.py` runs the **mock** NLI backend over 12 hand-written examples. Its numbers
(attribution 0.750, collapse delta +0.083) are a smoke test of the plumbing. They are
not evidence of anything and should not appear in the report as results.

---

## Test suite

```bash
pytest -q          # 44 passed, 6 skipped, 1 xfailed
NEUTRORAG_TEST_HF=1 pytest tests/test_memo_regressions.py -k hf_backend   # 6 passed (~44s)
```

The **1 xfail is deliberate**: `test_unusable_evidence_is_distinguishable_from_nothing_retrieved`
documents memo F3 — the scorer cannot distinguish "nothing retrieved" from "retrieved
but unusable", so both collapse to `TIF(0,1,0)`. It is marked `strict=True`, so if the
defect is ever fixed the test will fail until the marker is removed deliberately.
Fixing F3 is downstream of Gate B.

The 6 skips are the opt-in HFNLIBackend distribution tests, which require the real model.
