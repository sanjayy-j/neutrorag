# Audit ledger — NeutroRAG final submission

Each row: the issue, the evidence consulted, the correction, and verification
status. No credentials or secrets appear here.

**Repository state at audit start:** branch `main`, in sync with
`origin/main`, **working tree clean** (`git status --porcelain` empty, 0 files),
HEAD `2460554`. No uncommitted or untracked user work existed, so none could be
lost.

## The finding that reshapes this audit

> **The report is not in this repository.** No `.pdf`, `.docx` or `.tex` has
> ever been committed on any branch (`git log --all --name-only` over
> `main` and `review1-gate-a-diagnostics`), there is no `final-review/` or
> `submission/` directory in history, and `CITATIONS.md:3-4` states explicitly
> that the report is `NeutroRAG_First_Review_Report_final.docx`, *"edited
> outside this repository"*.

Consequence: every blocker that is **a defect in the report document** —
title-page placeholder, Table 10.1, Table 6.1, heading numbering, PDF
regeneration, abstract/conclusion sync — **cannot be executed here**. Every
blocker that is **a factual question about the evidence** can be, and has been.

## Blockers

| ID | Issue | Evidence consulted | Finding / correction | Status |
|---|---|---|---|---|
| **A** | Title-page demo placeholder | repo-wide search; recorder and codec availability; `PIL.ImageGrab` probe | **Not executable.** No `ffmpeg`/`asciinema`/`script`/`termtosvg`; no `mss`/`cv2`/`imageio`/`moviepy`/`pyautogui`; no audio. Decisively: commands run in a non-interactive shell with **no visible window**, so a screen capture would record the desktop while execution happened invisibly — a fabricated demo. A genuine timestamped transcript was produced instead and is labelled as **not** the video. | **UNRESOLVED — needs human** |
| **B** | Reproducibility checklist says "Pending" | the checklist lives in the absent report | **Cannot edit.** The underlying facts are now verified and tabulated in [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) so the checklist can be filled accurately by hand. Demo row must stay **not met**. | **BLOCKED on A + report absent** |
| **C** | Conflicting manifest totals: 46 / 66 / 29 | `git log -- results/claims.md`, each revision's header | **Resolved — all three are real, none wrong.** The manifest grew as registrations landed: 46/9 (`f3b577a`…`8f00121`) → 53/16 (`4354310`, H1) → 57/20 (`639f7ec`, RQ2) → 61/24 (`16728ab`, H2) → **66/29 (`2460554`, current)**. So 46 and 66 are *the same metric at different commits* and 29 is *a different metric* (pre-registered subset). **Authoritative: 66 claims, 29 pre-registered, 37 post-hoc.** | **RESOLVED** |
| **D** | "Ten registered tests, three positive" vs an inventory citing RQ1–RQ6 | `PREREGISTRATION.md` headings and all six addenda; result artefacts | **Resolved and it reconciles** — see [`TEST_INVENTORY.md`](TEST_INVENTORY.md). 10 registered, **9 executed**, 3 positive (RQ1, RQ3a, Addendum 6). Two caveats must travel with the sentence: RQ4 was never run, and RQ3 is counted as two halves. **RQ5 and RQ6 do not exist** in this repository. | **RESOLVED (2 disclosures)** |
| **E** | Reference arithmetic "17 checked, 15 correct, 3 errors" | `CITATIONS.md` lines 14, 20–22 | **The report's 17 is wrong; the repository already reconciles.** `CITATIONS.md` records **18** checked (`[1]`–`[17]` and `[19]`), 15 correct, 3 carrying errors — and 15 + 3 = 18 exactly. The report must say **18**. The two open qualifications (`[11]`'s truncated author list, `[18]` pre-dating DOIs) are preserved. | **RESOLVED — report must change 17→18** |
| **F** | CONFLICTS denominator: 100 instances, "exclude 5 Misinformation", n=80 | `neutrorag/conflicts.py`, `results/conflicts_pairwise.csv` (recomputed) | **n = 80 is CORRECT; the stated reason is wrong.** Release 458 → scored sample **100** (quota 40/40/10/10) → **80** eligible. The reduction is Complementary (10) + Outdated (10) carrying **no expected action** in the five-way space — *not* the 5 Misinformation instances, which were **never sampled** (0 present in the CSV). "100 minus 5" would wrongly imply 95. Recomputed agreement **16/80 = 0.2000**, 95% CI [0.112, 0.288]. | **RESOLVED — explanation must be corrected** |
| **G** | Carry-over numbering (objectives 15–21, baselines 22–28, …) | the lists live in the absent report | **Cannot edit.** No numbering defect exists in any repository document; `RESULTS.md` and `PREREGISTRATION.md` use section-local structure. Scientific identifiers (RQ1–RQ4, H1, H2) must not be renumbered. | **BLOCKED — report absent** |

## Numerical verification

All figures the brief attributes to the report were recomputed from raw
artefacts by `experiments/verify_report_claims.py`:
**31 PASS, 0 MISMATCH, 3 UNVERIFIABLE.** Full output in
[`logs/verify_report_claims.txt`](logs/verify_report_claims.txt).

Notably the brief's **p = 4.0 × 10⁻⁵** for the main FEVER comparison is
**correct** — the exact McNemar value is 4.005 × 10⁻⁵ (b=1, c=19), which the
project's own scripts print rounded as `0.0000`. The report's figure is the more
informative one.

The three unverifiable items are the ones with no repository basis: RQ5/RQ6,
"simulated-cascade" findings, and Tables 6.1/10.1.

## Changes made to the repository

| file | change | why |
|---|---|---|
| `experiments/verify_report_claims.py` | **new** | recomputes every report-asserted figure from raw artefacts and exits non-zero on mismatch, so it can gate a submission |
| `submission/` | **new directory** | audit artefacts, kept out of `results/` so no experimental artefact is touched |
| `results/claims.md` | regenerated | only the embedded commit-hash line changes; documented expected churn |

**Nothing else was modified.** No experiment output, log, frozen artefact,
pre-registration record or historical classification was altered. The frozen
Gate A dump still hashes to `70a3a783…84c6b7`.

## Explicitly not done

- **No commit or push.** The brief withholds authorization for both; all changes
  are staged in the working tree for review.
- **No screen capture.** Would have exposed a desktop I cannot inspect in
  advance and would not have shown the execution.
- **No pip install of a video codec.** A ~30 MB download plus a system change,
  and it would not have solved the no-visible-window problem.
- **No edits to `PREREGISTRATION.md`.** Historical records stay as written.
