# CONFLICTS policy test -- PAIRWISE mode (Decision 1)

Cache `data/conflicts_pairwise.jsonl` | model `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | max_passages 5 | sampling `{"seed": 20261011, "quota": {"Conflicting opinions and research outcomes": 40, "No conflict": 40, "Complementary information": 10, "Conflict due to outdated information": 10}}`
`python experiments/run_conflicts_pairwise_eval.py` | commit `72c719c`

**T is CORROBORATION in this mode** -- mutual agreement between sources, not support for a claim.

N = 100 instances. Pair-table lookup misses: 0 (expect 0).

| conflict type | n |
|---|---:|
| Conflicting opinions and research outcomes | 40 |
| No conflict | 40 |
| Complementary information | 10 |
| Conflict due to outdated information | 10 |

## r under the new targets, pairwise mode

- min 0.674, median 0.930, max 1.000; instances at r <= 0.5: **0/100**
- for comparison: FEVER claim mode median 0.750 (was 0.417); CONFLICTS claim mode under old targets was median 1.000, pstdev 0.0297

## 4a. Inter-source contradiction by annotated conflict type

| conflict type | n | median contradiction | mean | median corroboration | median I |
|---|---:|---:|---:|---:|---:|
| Conflicting opinions and research outcomes | 40 | **0.100** | 0.230 | 0.878 | 0.450 |
| No conflict | 40 | **0.331** | 0.427 | 0.757 | 0.510 |
| Complementary information | 10 | **0.667** | 0.535 | 0.709 | 0.611 |
| Conflict due to outdated information | 10 | **0.888** | 0.632 | 0.911 | 0.356 |

**Conflicting opinions (median 0.100, n=40) vs No conflict (median 0.331, n=40)**: Mann-Whitney U=525, z=-2.65, two-sided p=8.140e-03. Difference of medians -0.231.

**Verdict: contradiction is HIGHER on No conflict (the REVERSE of the prediction); the difference is significant at alpha=0.05.**
Claim mode on the same corpus gave the reverse: 0.222 vs 0.556, p=2.14e-03.

## 4b. The policy test -- actions vs external ground truth

Excluded: ['Conflict due to misinformation'] (5 instances, too few for any claim). Complementary information and Outdated information carry no expected action in the five-way space and are reported for context only.

### Action x conflict type

| conflict type | answer | expand | surface_contradiction | hedge | abstain | n |
|---|---|---|---|---|---|---|
| Conflicting opinions and research outcomes | 18 | 15 | 7 | 0 | 0 | 40 |
| No conflict | 9 | 16 | 15 | 0 | 0 | 40 |
| Complementary information | 1 | 4 | 5 | 0 | 0 | 10 |
| Conflict due to outdated information | 1 | 2 | 7 | 0 | 0 | 10 |

### Per-action, n=80

Exact-match accuracy **0.200** (95% CI [0.112, 0.288])

| expected action | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| surface_contradiction | 0.318 | 0.175 | 0.226 | 40 |
| answer | 0.333 | 0.225 | 0.269 | 40 |

**SURFACE_CONTRADICTION fires on 7/40 (17.5%) of Conflicting opinions vs 15/40 (37.5%) of No conflict.**

**Verdict: it does NOT fire preferentially on Conflicting opinions.** The policy central claim is UNSUPPORTED on this evidence.

