# Demo video runbook — 3 to 5 minutes

A recording plan for the mandatory demo artefact. Commands in order, what is on
screen, and what to say over it.

**The framing, before anything else.** This project's headline evaluation result
is **negative**. The demo shows that the *mechanism* works — that the five-way
policy chooses different actions for different evidence states. It must not
claim an improvement the evaluation does not support. Saying so on camera is a
strength, not a weakness: the first review explicitly credited the project's
self-reported defects, and a demo that over-claims is the one thing that would
undo that credit.

---

## Pre-flight (before you hit record)

```bash
cd ~/neutrorag
git status                      # expect: clean
python -m pytest -q             # expect: 98 passed, 6 skipped, 1 xfailed
python experiments/demo_video_run.py   # warm the model cache -- first load is slow
```

The last line matters: run it **once before recording** so the NLI model is in
the HuggingFace cache. Warm, the script takes **~10 seconds**. Cold, the first
run also downloads/loads weights and will stall your recording.

Have two things open: a terminal, and `RESULTS.md` in an editor.

**If the model fails to load on the day** (no network, cache cleared), fall back
to `python experiments/demo_video_run.py --nli mock`. It is instant and offline.
Say that you are using the offline heuristic backend — the actions are the same,
the numbers differ.

---

## Timeline

### 0:00 – 0:30 · What the project claims

*On screen:* `README.md`, top section.

> "NeutroRAG is a trust layer for retrieval-augmented generation. The idea is
> that 'I found no evidence' and 'I found evidence that disagrees with itself'
> are different failures that need different responses, and a single confidence
> score can't tell them apart. So instead of one number, each claim gets three —
> Truth, Indeterminacy, Falsity — and a five-way policy acts on them."

### 0:30 – 1:40 · The policy choosing different actions

*Command:*
```bash
python experiments/demo_video_run.py
```
*On screen:* three cases scrolling past. Runs in ~10 s warm, so you will be
talking over a finished screen — scroll back to each case as you describe it.

**Case 1 — ANSWER.** T=0.991, I=0.007, F=0.003.
> "Two sources, both entailing. High truth, nothing else. It answers."

**Case 2 — SURFACE CONTRADICTION.** T=0.527, I=0.042, **F=0.906**.
> "One source says the drug reduced risk by 30 percent, another says no
> significant reduction. Note it does *not* average these into 'probably true'.
> Truth and Falsity are both high at once — which a probability can't represent,
> because it would have to put those on one axis. The action is to surface the
> disagreement rather than pick a side."

Worth saying plainly:
> "This specific case is here because the first review found it **broken**.
> Conflict was computed on a pooled mean, so adding unrelated passages washed it
> out, and the system abstained on the one example built to demonstrate it. It
> now compares passages pairwise, and it fires."

**Case 3 — EXPAND then ABSTAIN.** Three steps, then abstain.
> "Thin evidence. High indeterminacy doesn't mean refuse — it means go and look
> again. The policy spends its retrieval budget first, and only abstains once
> the retriever has nothing new. That staging is the part that distinguishes
> this from ordinary abstention work."

The line above the steps reads `retrieval sufficiency r = 0.750`. Use it to make
the F1 point verbally, without a fourth case:
> "That r is retrieval sufficiency, and it's computed — from claim coverage,
> passage count, source diversity and graph connectivity. In the first review it
> was hardcoded at 0.9 for all 200 claims, so it wasn't a signal at all. On the
> FEVER run it now takes 26 distinct values with a median of 0.75."

If asked to *show* r varying, don't improvise it on camera — point at the
r-ablation table in `RESULTS.md`, which measures exactly that and reports it
coming out **negative** (0.665 computed vs 0.755 constant). An earlier fourth
case swept r across three retrieval states and was cut: it used the CONFLICTS
saturation targets while cases 1–3 use the library defaults, so the same
one-passage evidence set printed r = 0.154 there and r = 0.750 here. Both are
correct and the difference is per-corpus calibration by design, but it is not
explainable in the ten seconds a demo allows, and a reviewer who spots it will
spend your remaining time on it.

### 1:40 – 2:55 · The honest result

*On screen:* `RESULTS.md`, the status table and the RETRACTED section.

> "Now the part that matters. The headline number from the first review was an
> attribution accuracy of 0.745 on FEVER. On audit, that figure is **not
> attributable to this system**: running plain argmax on the NLI model's own
> output, with the identical model calls, scores 0.755 — slightly higher. So the
> neutrosophic layer contributed nothing measurable there, and the claim is
> retracted."

> "Two more retractions. The fuzzy-collapse ablation was arithmetic, not an
> experiment — any merged model is capped at 0.665 by the label prior. And
> 'T plus I plus F exceeds one' turned out to be an identity, not evidence of
> independence."

> "We then changed the indeterminacy formula so the identity genuinely breaks,
> and tested the policy against external ground truth on a knowledge-conflict
> dataset. That test also came out negative — the action fires more often on the
> wrong class. That is in RESULTS.md too."

### 2:55 – 3:40 · What that is worth

> "So the mechanism works and the evaluation doesn't support the claim yet.
> The diagnosis is specific: real-world disagreement is hedged — 'some studies
> suggest' against 'others found no effect' — and an NLI contradiction head
> scores that as agreement. It detects propositional negation, not evidential
> disagreement. That's a measurement problem with a clear next step, not a dead
> end."

> "Every number I've shown comes from a script committed in `experiments/`, and
> the repository records which commit produced which figure. The defects I
> described were found by audit and are written up rather than quietly fixed."

*Optional closing shot, if you have seconds spare:*
```bash
python -m pytest -q
```
> "98 tests, including one deliberately failing test that documents a defect we
> have not fixed yet."

---

## What NOT to show

- **`demo.py`'s old output.** It used to print a fuzzy-collapse delta framed as
  "the paper's key ablation". That framing is withdrawn. The script now prints a
  retraction notice instead, but there is no reason to put it on camera — its
  numbers are mock-backend plumbing on 12 toy examples.
- **Any accuracy figure as an achievement.** The only accuracy numbers this
  project can defend are the ones in RESULTS.md, and the headline one is
  retracted.
- **`run_pipeline_demo.py`** unless you have time to spare. It shows the
  retrieval→scoring→generation wiring, which is nice but is one trace, not a
  result. It now carries a banner saying so.

## Timing summary

| segment | budget |
|---|---|
| what the project claims | 0:30 |
| the three policy cases | 1:10 |
| the honest result | 1:15 |
| what that is worth | 0:45 |
| **total** | **3:40** |

Comfortably inside a 3–5 minute limit, with 1:20 of headroom. The honest-result
segment is the one that earns the marks, so if you overrun, take the time out of
the policy cases rather than out of it — Case 1 can be described in a sentence
without scrolling back to it.
