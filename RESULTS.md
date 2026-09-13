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
| **Gate B** (definition of `r`) | signed off (route (i)); `r` implemented and wired. **Its stated prediction is falsified** — see below |
| **Gate C** (expansion loop not a no-op) | test written and **passing** — see below |

> **Blocked:** the r-ablation, the F12 sensitivity sweep and the repeated Gate A
> were **not** run. The Gate B prediction test is a stop-gate and it did not pass.

---

## Gate B: `r` implemented, and its prediction falsified

### What was implemented

`neutrorag/sufficiency.py` computes `r` from four retrieval-side components
(coverage, support_count, connectivity, diversity), equal-weighted, with no
dependence on `e`, `n` or `c` — enforced structurally and asserted by
`tests/test_sufficiency.py`. `scoring.py` now calls it in place of
`max(relevance)`; `fever.py`'s hardcoded `0.9` is now the named, documented
constant `GOLD_EVIDENCE_RELEVANCE = 1.0`.

**`r` now carries information** where it previously could not: 0.167 / 0.667 /
0.833 across the existing test scenarios, and **0.292 – 0.969** on the pipeline
corpus with all four components live, versus a former constant 0.9. Memo **F1 is
addressed**.

### The prediction test — FALSIFIED

The Gate B proposal committed in advance to: *"The mass identity breaks … once
`r` carries retrieval information orthogonal to the NLI simplex, the sum is no
longer a function of `disagree` alone."*

```bash
python experiments/check_mass_identity.py
```

| check | max \|deviation\| |
|---|---:|
| random draws, `r ~ Uniform(0,1)` | 4.441e-16 |
| random draws, `r` computed by `sufficiency` | 4.441e-16 |
| real run: pipeline corpus + graph, all four components live (`r` = 0.292–0.969) | 2.220e-16 |
| real run: 200 FEVER claims, replaying the Gate A NLI outputs | 2.220e-16 |

**The identity still holds, to machine precision. `T + I + F = 1 + λ·disagree`
is unchanged.**

### Why — it is algebraic, not empirical

The cancellation never depended on `r` being uninformative:

```
T + I + F = r·e + (1 − r + r·n) + (r·c + λ·disagree)
          = r·(e + n + c) + (1 − r) + λ·disagree
          = r + 1 − r + λ·disagree            [since e + n + c = 1]
          = 1 + λ·disagree
```

`r` multiplies the *whole* NLI simplex across the three channels and then cancels
against `I`'s `(1 − r)` term — **for any `r`, from any source**. Verified directly:
holding `e,n,c` fixed and sweeping `r` over 0.0, 0.137, 0.5, 0.9, 1.0 gives
`T+I+F = 1.250000000000` at every value.

So redefining `r` cannot break the identity **while the three formulas are
unchanged**. Note what the memo's F5 route (i) actually said: *"break the identity
by **deriving I** from a retrieval-sufficiency signal"* — that is a change to `I`'s
**formula**, which this session was explicitly scoped not to touch. Route (i) is
not refuted; it was scoped in a way that made it unreachable. Breaking the identity
requires an `I` whose `r`-dependence does not exactly cancel `r·(e+n+c)` — a new
framing decision needing sign-off.

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

`T + I + F = 1 + λ·disagree`, **identically** (memo F5, Appendix A). Verified on the
real run: max deviation over 200 rows = **2.22e-16**. The triple carries one degree
of freedom beyond the probability simplex, not three; "mass exceeds 1" is a
restatement of "disagree > 0". Rewriting this claim is a **Gate B** decision and has
not been made.

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
