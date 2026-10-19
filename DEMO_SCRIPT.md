# Demo video runbook — 3 to 5 minutes

**The arc is the diagnosis, not a system tour.** The project's central
hypothesis was pre-registered and refuted; the valuable part is *why*, and that
is a four-beat story. Do not demo features. A reviewer who watches this should
come away able to state the design rule.

Four beats, 3:40 total:

| beat | content | budget |
|---|---|---|
| 1 | the construction and the pre-registered prediction | 0:45 |
| 2 | the prediction failing | 1:00 |
| 3 | `r = 0.3333·coverage + 0.2000` — the gate reading back its own action | 1:30 |
| 4 | the design rule and the open question | 0:45 |

---

## Pre-flight, before you hit record

```bash
cd ~/neutrorag
git status                      # expect: clean
python -m pytest -q             # expect: 141 passed, 6 skipped, 3 xfailed
```

**Measured runtimes on the recording machine** — check these before you plan
the beats, because one of them is slow:

| command | time | loads |
|---|---:|---|
| `analyse_hotpot_gate.py` | **0.1 s** | committed CSV only |
| `analyse_hotpot_by_type.py` | **0.1 s** | committed CSV only |
| `diagnose_r_depth.py` | **6.7 s** | CSV + the HotpotQA parse |
| `make_claims_manifest.py` | **6.1 s** | CSVs + the parse |
| `diagnose_r_components.py` | **62 s** | **re-embeds 400x10 paragraphs** |

Every beat below uses only the fast three. **Do not run
`diagnose_r_components.py` on camera** — it re-embeds the corpus and would put a
minute of dead air inside beat 3. Its one figure is read from the generated
manifest instead, which is equally a committed-script output.

Nothing downloads, so there is no network dependency on the day.

Have three things open: a terminal, [`results/claims.md`](results/claims.md),
and [`PREREGISTRATION.md`](PREREGISTRATION.md).

---

## Beat 1 — the construction and the prediction · 0:00–0:45

*On screen:* `PREREGISTRATION.md`, scrolled to the RQ3 block.

> "NeutroRAG scores retrieved evidence into three channels — truth,
> indeterminacy, falsity — instead of one confidence number. The claim I
> pre-registered is narrow and specific: that conditioning retrieval expansion
> on **indeterminacy specifically** beats conditioning it on **low confidence
> generally**. Same retriever, same reader, same expansion budget — the only
> difference is the decision rule."

Point at the registered stop condition and read it aloud:

> "And I wrote down in advance what would count as failure: if the decomposed
> gate doesn't beat a scalar gate at equal budget, the claim is dead and I stop.
> That's committed before the experiment ran — the commit order is the
> evidence."

**Number to point at:** the commit hash `2551e65`, and that the file is 388
lines of predictions written before the data.

---

## Beat 2 — the prediction failed · 0:45–1:45

*Command:*
```bash
python experiments/analyse_hotpot_gate.py
```

*On screen:* the `OUTCOMES` table and the `McNEMAR vs c_i_gate` block.

> "Here are all five conditions on 400 HotpotQA questions. The decomposed gate
> is the row marked `c_i_gate`. It scores **0.3025**. The scalar gate scores
> **0.3225** — and that scalar is just the NLI head's own maximum class
> probability, which I'd registered in advance as the *weak* baseline. McNemar
> gives p equals 0.24: not significant, and in the wrong direction. The
> decomposition lost to the thing it was supposed to improve on."

**The single number to point at: `0.3025` against `0.3225`.**

Then pre-empt the obvious objection, because a reviewer will raise it:

> "First thing I checked was whether the two gates simply picked the same
> questions. They don't. Expansion rates are matched exactly at 0.7775 by
> construction, but the decision overlap is **72.3%** against a pigeonhole floor
> of **71.4%** — so they're close to maximally different given the matched rate.
> On the 172 questions where they disagree, the decomposed gate is *still*
> behind. This isn't an untested tie."

If you have seconds spare, add the honest positive:

> "One thing did work: expansion beats no expansion on the questions where a
> gold paragraph sits outside top-k. p equals **0.0011**. But that's FLARE and
> CRAG reproduced training-free, not a new result."

---

## Beat 3 — the mechanism · 1:45–3:15

**This is the beat that matters. Give it the most time.**

*Command* (instant — reads the generated manifest, not the 62-second script):
```bash
grep -E '^\| D1-|^\| D2 ' results/claims.md
```

*On screen:* the four `D1` component rows, then the `D2` row.

> "So I went looking for why. The sufficiency term `r` is supposed to combine
> four retrieval signals. Look at the variance figures. Two of them are
> **exactly zero** — `support_count` and `diversity` are pinned at 0.3 on every
> one of 400 questions, one distinct value each. The fourth, graph
> connectivity, isn't computed at all because there's no graph. So `r` is one
> component wearing a four-component costume."

*(These rows come from `experiments/diagnose_r_components.py`, which takes about
a minute because it re-embeds the corpus. The manifest carries its output, so
you show the figure without the wait. Say so if asked.)*

**Number to point at: `r = 0.3333 * coverage + 0.2000`, max residual
`1.11e-16`.**

> "`r` is an affine function of lexical coverage alone, exact to machine
> precision. Which also means it's trapped in **0.2 to 0.533** — it can never
> reach the 0.5 boundary the policy tests against unless coverage exceeds 0.9."

*Command:*
```bash
python experiments/diagnose_r_depth.py
```

*On screen:* the `eta-squared` block.

> "And here's the real problem. At full depth the two pinned components
> saturate, so expanding adds **0.4667 to `r` mechanically** — whether or not
> expansion retrieved anything useful. Retrieval depth alone explains
> **eta-squared equals 0.9144** of how much `r` moves."

**Number to point at: `eta^2 = 0.9144`.**

> "Ninety-one percent. And it's highest for the one gate that actually reads
> `r`. Since truth is bounded by `r` and indeterminacy is bounded below by one
> minus `r`, the gate flips toward 'expand' before expanding and toward 'stop'
> after — by arithmetic on the retrieval depth **the gate itself chose**. It's
> reading back its own previous action. The fraction of questions above the
> boundary goes from 14.8% to 81% just by expanding."

---

## Beat 4 — the design rule and what's open · 3:15–4:00

*On screen:* [`PREREGISTRATION.md`](PREREGISTRATION.md), Addendum 3(e).

> "So the finding isn't 'the decomposition doesn't work'. It's sharper than
> that, and it generalises:"

> **"A sufficiency signal must not be a monotone function of the retrieval depth
> that the gate reading it controls."**

> "That's a design rule any adaptive-retrieval system can be checked against,
> and it came out of a negative result. There are three more requirements in the
> pre-registration — the saturation target has to sit below the candidate pool,
> `r` has to stay above 0.5 with real variance, and the outcome label has to
> still vary at maximum depth, which rules out the two obvious ones on this
> dataset."

Close on the honest boundary of the claim:

> "What I have **not** shown is that decomposed gating fails in general. I've
> shown that *this* construction fails at *this* operating point — and one of
> those four requirements was violated on 82% of the sample before any evidence
> was looked at. That's the open question, and it's stated as an open question
> in the repository rather than smoothed over."

*Optional closing shot if you have seconds spare:*
```bash
python experiments/make_claims_manifest.py
```
> "Forty-six claims, nine of them pre-registered, every figure regenerated from
> the committed data by that script rather than typed in. The thirty-seven
> post-hoc ones are labelled post-hoc."

---

## What NOT to show

- **Any feature tour.** No walkthrough of the five actions, no `demo.py`, no
  pipeline trace. The arc is the diagnosis; a tour costs a beat and adds nothing
  a reviewer is marking.
- **Any accuracy figure as an achievement.** The headline is refuted and three
  First Review figures are retracted. Presenting a number as a win is the one
  thing that would undo the credit the honesty earns.
- **`experiments/run_hotpot_gate.py`.** It takes 150 minutes. Beat 2 reads its
  committed output in under a second.
- **Exact Match against 1.0.** The reader is extractive, so EM is capped at
  0.938 overall and 0.692 on comparison questions. If you quote an EM, quote
  the ceiling with it.

## If something goes wrong on the day

No beat command touches the network, and only `diagnose_r_depth.py` reads
anything beyond a CSV. If a script errors, the numbers are all in
[`results/claims.md`](results/claims.md) — show that file instead and say it is
generated, not transcribed. The beats survive intact.

## Timing summary

| beat | budget | the one number |
|---|---|---|
| construction and prediction | 0:45 | commit `2551e65` |
| the prediction failing | 1:00 | **0.3025 vs 0.3225** |
| the mechanism | 1:30 | **`1.11e-16`**, then **`eta^2 = 0.9144`** |
| design rule and open question | 0:45 | 82% of the sample |
| **total** | **4:00** | |

Inside a 5-minute limit with a minute spare. If you overrun, cut from beat 1 —
the pre-registration can be asserted in one sentence. **Do not cut beat 3**; it
is the contribution.
