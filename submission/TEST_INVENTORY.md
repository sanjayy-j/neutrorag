# Registered-test inventory — Blocker D

**Source of truth: [`PREREGISTRATION.md`](../PREREGISTRATION.md)** (1000+ lines,
six dated addenda) and the committed result artefacts. Figures verified by
`python experiments/verify_report_claims.py` (31 PASS, 0 MISMATCH).

**No historical classification has been changed.** This document counts what the
pre-registration already says; it does not reinterpret any registered item.

## Definitions, stated before the counts

- **Registered test** — a prediction with a decision criterion recorded in
  `PREREGISTRATION.md` *before* the data it tests existed. Manipulation checks
  are **not** counted as tests: they are voiding conditions on whether a test is
  interpretable at all.
- **Positive** — the registered criterion was met *and* the supporting
  statistical test reached α = 0.05, or, for RQ1, the registered invariance held
  exactly. A directional result that did not reach α is **not** positive.
- One experiment is counted **once**, even where it supports several narrative
  claims. RQ3 is counted as **two** tests because the pre-registration gives it
  two separately-criterioned halves ("does expansion recover misses" and "does
  it beat a scalar at equal budget"), and they resolved in opposite directions.

## The inventory

| # | ID | Registered prediction | Primary/secondary | Executed | Criterion met? | Evidence |
|---|---|---|---|---|---|---|
| 1 | **RQ1** | Pairwise conflict aggregation is invariant to neutral-passage dilution; mean-pooling decays | primary | yes | **YES — positive** | pairwise flat at 1.000 across 0–4 filler; pooled 0.847→0.400; raw unclamped flat at 1.739 |
| 2 | **RQ2** | `r` achieves higher AUROC for answerability than raw retrieval score and coverage alone | primary | yes (as the proposal-registered quantity) | **PARTIAL** | beats `max_relevance` for retrieval success, AUROC 0.5958 vs 0.5004, DeLong p = 0.0149; for answer correctness 0.5483, CI spans 0.5 — not positive |
| 3 | **RQ3a** | Indeterminacy-triggered expansion recovers answers fixed-`k` misses | primary | yes | **YES — positive** | EM 0.1486 → 0.2432 on 222 recoverable misses, exact McNemar b=30, c=9, **p = 0.0011** |
| 4 | **RQ3b** | …and beats a scalar-confidence trigger at equal expansion budget | primary — **the novelty claim** | yes | **NO — stop condition fired** | I-gate 0.3025 vs `max(e,n,c)` 0.3225, b=14, c=22, p = 0.2430; rates matched 0.7775 |
| 5 | **RQ4** | Gate matches LLM-gating quality at ~40× fewer parameters | primary | **NOT RUN** | **n/a** | Phases 4–5 were not run because RQ3b's stop condition fired. No cost figure is claimed anywhere |
| 6 | **Addendum 2** | The I-gate beats the scalar on recovering gold into context | registered prediction | yes | **NO — falsified** | I-gate recovered the least of four gated conditions, 0.7027 vs 0.7342 / 0.7477 / 0.8153 |
| 7 | **H1** (Addendum 4) | On comparison questions, fixed k=3 beats the I-gate (one-sided) | primary | yes | **NO — not supported** | 0.1567 vs 0.1533, b=20, c=22, one-sided p = 0.4388; the 4/4 uniformity became 1/4 |
| 8 | **H2** (Addendum 5) | With `T` reachable, the I-gate beats the best scalar at equal budget | primary | yes | **NO — falsified, direction reversed** | 0.2900 vs 0.3075, one-sided p = 0.9461; both manipulation checks PASSED, so the test was interpretable and not void |
| 9 | **Addendum 6** | Under hard retrieval, AUROC of legacy `r` for answer correctness exceeds 0.5 | primary, two-sided | yes | **YES — positive** | AUROC **0.6032**, 95% CI **[0.5331, 0.6733]**, excludes 0.5; n = 400 |
| 10 | **Addendum 6c** | Paired: `r` is more predictive under hard than easy retrieval on the same 400 questions | planned-conditional secondary | yes (condition met) | **DIRECTIONAL ONLY** | 0.6032 vs 0.5542; raw +0.0490; DeLong on the 367 label-agreeing questions +0.0382, z = 1.725, p = 0.0844 |

**Not counted as tests** — two voiding manipulation checks registered in
Addendum 5, both **PASSED**: `r_v2 > 0.5` on 77.75% (threshold ≥ 40%), and η² of
Δ`r_v2` from `final_k` = 0.1172 (threshold < 0.30, against 0.9144 before).

## The reconciliation

| quantity | count |
|---|---:|
| **Registered tests** | **10** |
| of which executed | 9 |
| of which **not run** (RQ4) | 1 |
| **Positive** (criterion met and α = 0.05 reached) | **3** — RQ1, RQ3a, Addendum 6 |
| Partial | 1 — RQ2 |
| Directional only | 1 — Addendum 6c |
| Negative / falsified / stop condition | 4 — RQ3b, Addendum 2, H1, H2 |

**"Ten registered tests, three positive" reconciles exactly**, under the
definitions above. Two things must travel with that sentence or it misleads:

1. **One of the ten was never run.** RQ4 is registered but unexecuted, so the
   honest phrasing is *"ten registered tests, nine executed, three positive."*
2. **RQ3 is counted as two.** Collapsing it to one gives nine registered and
   three positive; splitting it gives ten and three. Either is defensible, but
   the report must say which it used. This inventory splits it, because the two
   halves have separate criteria and resolved oppositely — and because the
   novelty claim rests entirely on RQ3b, which failed.

## Claims not in this repository

The audit brief refers to **RQ5** and **RQ6**, and to "simulated-cascade"
findings. **Neither appears anywhere in this repository** — the pre-registration
defines RQ1–RQ4 only, and a repo-wide search for "simulated cascade" returns
nothing. Any such item in the report has no basis in the committed evidence and
must be either substantiated from a source outside this repository or removed.

> **Correction, superseding the paragraph above on RQ5/RQ6.** The inference
> drawn there was wrong, and it is left in place rather than deleted so the audit
> trail stays honest. The two documents use **different RQ numbering over the
> same work**: this repository numbers its research questions as
> `PREREGISTRATION.md` defines them, RQ1–RQ4, while the report uses its own
> RQ1–RQ6 and says so. **The report's RQ5 and RQ6 are the two halves of this
> repository's RQ3.** So the report's RQ5/RQ6 are substantiated after all, by
> RQ3's evidence, and **no figure differs between the two documents** — only the
> labels do. Cross-check by question text, not by number. The
> "simulated-cascade" half of the original finding is a separate matter and
> **still stands**: no such artefact exists in this repository.

