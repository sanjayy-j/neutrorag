# NeutroRAG — demo session transcript

**Captured 2026-10-23 00:37:51 local time** by `python submission/demo/capture_demo_session.py`.

> ### THIS IS NOT THE REQUIRED DEMO VIDEO
> It is a genuine transcript: every line of output below was produced by running the command shown above it, on this machine, at capture time. Nothing is reconstructed, edited or simulated. Progress bars, HuggingFace download warnings and symlink warnings are filtered out; no result line is altered.
>
> A video could not be recorded here. The environment has no screen recorder (`ffmpeg`, `asciinema`, `script`, `termtosvg` all absent) and no audio. More decisively, these commands run in a non-interactive shell with **no visible window**, so a screen capture would have recorded the desktop while execution happened invisibly — a fabricated demo rather than a recording of one. See `submission/FINAL_AUDIT_REPORT.md` for the exact remaining step.

Follows the four beats of [`DEMO_SCRIPT.md`](../../DEMO_SCRIPT.md).

---

## 1. The construction and the pre-registered prediction

Shows the registered question and its stop condition, written before the data existed.

`$ git log --format=%h %ad %s --date=short -1 2551e65`

*captured 00:37:51, exit code 0, 0.0s*

```
2551e65 2026-10-19 Pre-register Phase 3: four research questions, predictions and stop conditions
```

## 1b. The pre-registration is 1000+ lines of predictions, not prose

Size and the four research questions it fixes.

`$ python -c import pathlib;t=pathlib.Path('PREREGISTRATION.md').read_text(encoding='utf-8');print(f'PREREGISTRATION.md: {len(t.splitlines())} lines');print('\n'.join(l for l in t.splitlines() if l.startswith('### RQ')))`

*captured 00:37:52, exit code 0, 0.1s*

```
PREREGISTRATION.md: 1087 lines
### RQ1 — Does pairwise conflict aggregation resist evidence dilution where mean-pooling does not?
### RQ2 — Does retrieval sufficiency `r` predict answerability better than raw retrieval score or coverage alone?
### RQ3 — Does indeterminacy-triggered expansion recover answers that fixed-`k` retrieval misses, and does it beat expansion triggered by a scalar confidence at equal expansion budget?
### RQ4 — Does the gate match LLM-based gating quality at roughly 40× fewer parameters?
```

## 2. The prediction failing

The five-condition comparison at matched expansion budget. The decomposed gate is c_i_gate.

`$ python experiments/analyse_hotpot_gate.py`

*captured 00:37:52, exit code 0, 0.1s*

```
==============================================================================
PHASE 3 ANALYSIS
==============================================================================
n = 400   recoverable-miss = 222   already-sufficient = 178

==============================================================================
EQUAL-BUDGET AUDIT  (the comparison is void if these differ)
==============================================================================
condition                 any-exp rate   mean exps  total exps  mean final k
a_fixed_k                       0.0000      0.0000           0          3.00
b_scalar_entail                 0.7775      1.5275         611          8.33
b_scalar_max_enc                0.7775      1.4575         583          8.05
b_scalar_one_minus_n            0.7775      1.5075         603          8.25
c_i_gate                        0.7775      1.3550         542          7.64

==============================================================================
DECISION OVERLAP  (equal rates do not mean equal decisions)
==============================================================================
  (c) I-gate fired on 311/400 questions
  pigeonhole: any gate with the same rate must overlap (c) on at
  least 2*311 - 400 = 222 questions, i.e. >= 71.4% of its
  firing set. Upper bound is 311 (identical decisions, 100%).

  gate                       |A&B|    ovl%  c-only  b-only   agree  eff.n
  b_scalar_entail              251   80.7%      60      60   70.0%    120
  b_scalar_max_enc             225   72.3%      86      86   57.0%    172
  b_scalar_one_minus_n         277   89.1%      34      34   83.0%     68

  c-only / b-only are the discordant decision sets. They are equal in
  size by construction, because the firing rates were equalised.
  eff.n is the number of questions that can possibly differ: on a
  concordant question both conditions run the same pipeline and give
  the same answer, contributing a tie and no information.

==============================================================================
OUTCOMES
==============================================================================
condition                      EM           95% CI       F1  EM(recov)    recov    unnec
a_fixed_k                  0.2675   [0.224, 0.311]   0.3636     0.1486   0.0000   0.0000
b_scalar_entail            0.3100   [0.265, 0.355]   0.4066     0.2703   0.8153   0.7247
b_scalar_max_enc           0.3225   [0.277, 0.368]   0.4247     0.2658   0.7342   0.7584
b_scalar_one_minus_n       0.3075   [0.262, 0.353]   0.4086     0.2658   0.7477   0.7921
c_i_gate                   0.3025   [0.257, 0.348]   0.4032     0.2432   0.7027   0.7865

==============================================================================
McNEMAR vs c_i_gate
==============================================================================
  vs a_fixed_k                whole: b= 34 c= 20 p=0.0759   recoverable: b= 30 c=  9 p=0.0011
  vs b_scalar_entail          whole: b=  7 c= 10 p=0.6291   recoverable: b=  4 c= 10 p=0.1796
  vs b_scalar_max_enc         whole: b= 14 c= 22 p=0.2430   recoverable: b= 12 c= 17 p=0.4583
  vs b_scalar_one_minus_n     whole: b=  6 c=  8 p=0.7905   recoverable: b=  3 c=  8 p=0.2266

==============================================================================
BY QUESTION TYPE
==============================================================================
condition                              bridge            comparison
a_fixed_k                   0.293 [0.24,0.34]     0.165 [0.08,0.25]
b_scalar_entail             0.358 [0.31,0.41]     0.114 [0.04,0.18]
b_scalar_max_enc            0.368 [0.31,0.42]     0.139 [0.06,0.22]
b_scalar_one_minus_n        0.358 [0.31,0.41]     0.101 [0.03,0.17]
c_i_gate                    0.349 [0.30,0.40]     0.114 [0.04,0.18]
    n(bridge) = 321
    n(comparison) = 79

==============================================================================
OUTCOMES ON THE DISCORDANT SET ONLY
==============================================================================

  vs b_scalar_entail   (n_discordant = 120)
    (c) I-gate EM 0.3083  95% CI [0.226, 0.391]
    (b) scalar  EM 0.3167
    difference    -0.0083   McNemar b=6 c=7 p=1.0000

  vs b_scalar_max_enc   (n_discordant = 172)
    (c) I-gate EM 0.2674  95% CI [0.201, 0.334]
    (b) scalar  EM 0.2907
    difference    -0.0233   McNemar b=13 c=17 p=0.5847

  vs b_scalar_one_minus_n   (n_discordant = 68)
    (c) I-gate EM 0.3235  95% CI [0.212, 0.435]
    (b) scalar  EM 0.3382
    difference    -0.0147   McNemar b=5 c=6 p=1.0000

==============================================================================
PRE-REGISTERED VERDICT
==============================================================================
  (c) I-gate EM 0.3025  vs  (b) scalar max(e,n,c) EM 0.3225   difference -0.0200
  McNemar b=14 c=22 p=0.2430
  Did (c) beat (b)? NO
  -> STOP CONDITION FIRED (pre-registered). Do not proceed to
     Phases 4-5 and do not adjust thresholds.
```

## 3. The mechanism -- r is an affine function of coverage alone

The zero-variance components and the affine law, read from the generated manifest.

`$ python -c import pathlib;print('\n'.join(l[:160] for l in pathlib.Path('results/claims.md').read_text(encoding='utf-8').splitlines() if l.startswith('| D1-') or l.startswith('| D2 ')))`

*captured 00:37:52, exit code 0, 0.1s*

```
| D1-coverage | Distribution of coverage at base depth k=3. | min 0.2222, max 1.0000, range 0.7778, mean 0.7207, var 0.023640, 66 distinct values | descriptive 
| D1-support_count | Distribution of support_count at base depth k=3. | min 0.3000, max 0.3000, range 0.0000, mean 0.3000, var 0.000000, 1 distinct values | des
| D1-diversity | Distribution of diversity at base depth k=3. | min 0.3000, max 0.3000, range 0.0000, mean 0.3000, var 0.000000, 1 distinct values | descriptive
| D1-r | Distribution of r at base depth k=3. | min 0.2741, max 0.5333, range 0.2593, mean 0.4402, var 0.002627, 66 distinct values | descriptive | 400 | `exper
| D2 | With two of three components pinned, r is an affine function of coverage alone. | r = 0.3333*coverage + 0.2000; max |predicted - actual| = 1.11e-16; r co
```

## 3b. The circularity, quantified

eta^2 of the change in r explained by retrieval depth alone.

`$ python experiments/diagnose_r_depth.py`

*captured 00:37:52, exit code 0, 8.3s*

```
==============================================================================
IS r COUNTING DEPTH?  (committed CSVs only; no model loaded)
==============================================================================
n = 400   commit 2460554

------------------------------------------------------------------------------
(4) The 0.7775 firing rate: same COUNT, or the same QUESTIONS?
------------------------------------------------------------------------------
  b_scalar_entail          fired on 311/400  rate 0.7775
  b_scalar_max_enc         fired on 311/400  rate 0.7775
  b_scalar_one_minus_n     fired on 311/400  rate 0.7775
  c_i_gate                 fired on 311/400  rate 0.7775

  identical COUNT across all four? True
  identical QUESTION SET across all four? False
  fired by ALL four : 168   by AT LEAST one : 399

  The equal count is BY CONSTRUCTION, not a coincidence: the
  experiment calibrates each scalar threshold to the quantile of its
  own score distribution that reproduces the I-gate's rate, which
  forces the counts equal. That is the pre-registered equal-budget
  design. The SETS are what carries information, and they differ.

------------------------------------------------------------------------------
(1) r at base vs final depth, and the empirical dr
------------------------------------------------------------------------------
  base (k=3)                 min 0.3111 max 0.5333 mean 0.4546 var 0.001981   r>0.5: 59/400

  condition                      min     max    mean       var     r>0.5   mean dr
  b_scalar_entail             0.3212  1.0000  0.8417  0.046600   323/400    0.4978
  b_scalar_max_enc            0.3111  1.0000  0.8198  0.048450   321/400    0.4697
  b_scalar_one_minus_n        0.3429  1.0000  0.8362  0.046919   321/400    0.4908
  c_i_gate                    0.3905  1.0000  0.7939  0.044618   324/400    0.4363

  dr distribution over questions that expanded:
  condition                      n      min   median      max       var   dr<0
  b_scalar_entail              311   0.1798   0.4987   0.6611  0.004365      0
  b_scalar_max_enc             311   0.1591   0.4926   0.6235  0.010324      0
  b_scalar_one_minus_n         311   0.1798   0.4970   0.6611  0.006367      0
  c_i_gate                     311   0.1798   0.4889   0.6611  0.017094      0

------------------------------------------------------------------------------
(3) Backing out the components at final depth k=10
------------------------------------------------------------------------------
  questions reaching k=10 under some condition : 388
  of which hold exactly 10 paragraphs          : 388
  (for those, k=10 retrieves every paragraph, so coverage at k=10 is
   determined by the parse and needs no retriever)

  implied diversity at k=10: min 1.0000 max 1.0000 mean 1.0000 var 0.00000000
  saturated at 1.0 : 303/388 (78.1%)
  max |implied - 1.0| : 1.478e-06
  deviating by more than 1e-3 : 0/388
    The CSV stores r rounded to 6 decimals, so backing a component
    out of it carries up to 3 * 5e-7 = 1.5e-6 of error. Deviations at
    that scale are rounding, not evidence against saturation.

  coverage at k=10, on the hypothesis : min 0.5556 max 1.0000 mean 0.8868

  => mechanical part of dr = 2 * (1.0 - 0.3)/3 = +0.4667

------------------------------------------------------------------------------
(2) Is dr larger when expansion actually HELPED?
------------------------------------------------------------------------------
  helped  = gold_in_context 0 -> 1     (expansion fetched the gold)
  failed  = gold_in_context stayed 0   (expansion fetched nothing useful)
  already = gold_in_context already 1 at base depth

  CONFOUND, stated first: dr is mechanically a function of final_k
  (+0.2333 at k=6, +0.4667 at k=10, before any coverage gain), and
  k=10 retrieves EVERY paragraph so gold_in_context is guaranteed
  there. A raw dr-vs-helped correlation is therefore partly a
  dr-vs-depth correlation. The stratified table below is the test.

  condition                            helped             failed     diff     rho        p
  b_scalar_entail              0.5049 (n=181)       0.2741 (n=1)  +0.2309   0.120   0.1079
  b_scalar_max_enc             0.4949 (n=163)      0.2280 (n=13)  +0.2670   0.423   0.0000
  b_scalar_one_minus_n         0.5010 (n=166)       0.1994 (n=4)  +0.3017   0.259   0.0007
  c_i_gate                     0.4596 (n=156)      0.2097 (n=15)  +0.2499   0.436   0.0000

------------------------------------------------------------------------------
(2b) dr STRATIFIED BY final_k -- is dr a function of depth alone?
------------------------------------------------------------------------------
  condition                 final_k     n   mean dr     var dr  gold_in_ctx
  b_scalar_entail                 6    11    0.2221   0.000916       0.9091
  b_scalar_entail                10   300    0.5079   0.001603       1.0000
  b_scalar_max_enc                6    39    0.2189   0.000957       0.6667
  b_scalar_max_enc               10   272    0.5057   0.001351       1.0000
  b_scalar_one_minus_n            6    19    0.2188   0.000961       0.7895
  b_scalar_one_minus_n           10   292    0.5085   0.001591       1.0000
  c_i_gate                        6    80    0.2239   0.001126       0.8125
  c_i_gate                       10   231    0.5099   0.001581       1.0000

  variance of dr explained by final_k alone (eta-squared):
    b_scalar_entail          total var 0.004365  within 0.001578  eta^2 = 0.6384
    b_scalar_max_enc         total var 0.010324  within 0.001302  eta^2 = 0.8739
    b_scalar_one_minus_n     total var 0.006367  within 0.001552  eta^2 = 0.7562
    c_i_gate                 total var 0.017094  within 0.001464  eta^2 = 0.9144

  If mean dr is near-constant WITHIN each final_k and the gap
  between depths is ~0.2333, then dr is determined by how many
  passages were fetched. gold_in_ctx at k=10 is 1.0 by construction,
  because k=10 retrieves every paragraph -- so the earlier
  dr-vs-helped correlation cannot be separated from depth here.
```

## 4. The design rule and the one positive registered result

The Addendum 6 primary: under hard retrieval r does predict answer correctness.

`$ python -c import pathlib;print('\n'.join(l[:200] for l in pathlib.Path('results/claims.md').read_text(encoding='utf-8').splitlines() if l.startswith('| A6 ') or l.startswith('| A6-paired')))`

*captured 00:38:00, exit code 0, 0.0s*

```
| A6 | PRE-REGISTERED PRIMARY (Addendum 6): under HARD retrieval, AUROC of legacy r for ANSWER CORRECTNESS is above 0.5 -- POSITIVE, the CI excludes 0.5. | AUROC 0.6032, SE 0.0358, 95% CI [0.5331, 0.6
| A6-paired | PLANNED-CONDITIONAL SECONDARY (Addendum 6): paired on the same 400 questions, r is predictive under hard retrieval and NOT under easy retrieval; the improvement itself is directional onl
```

## 5. Reproducibility

The test suite and the manifest that regenerates every figure.

`$ python -m pytest -q`

*captured 00:38:00, exit code 0, 1.2s*

```
..................x..x.................................................. [ 43%]
...x........ssssss...................................................... [ 86%]
......................                                                   [100%]
157 passed, 6 skipped, 3 xfailed in 0.74s
```

---

Capture finished 00:38:01; 7 beats.