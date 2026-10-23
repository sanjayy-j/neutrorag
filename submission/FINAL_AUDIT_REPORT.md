# NeutroRAG — final submission audit

**Audit date:** 23 October 2026 · **Repository:** `main` @ `2460554` ·
**Auditor role:** research auditor / verification engineer

# VERDICT: NOT READY

One mandatory requirement is unresolved: **there is no demo video, and no
accessible demo link.** The brief states this verdict may not be softened for an
unmet demo requirement, so it is not.

Everything that *is* executable in this environment has been completed and
verified. The scientific content is in good order: **31 of 31 recomputable
figures match the raw artefacts, with 0 mismatches.** The blockers that remain
are a missing video and a report document that is not in this repository.

---

## 1. The finding that reshapes this audit

**The report does not exist in this repository.**

- No `.pdf`, `.docx` or `.tex` has **ever** been committed, on `main` or
  `review1-gate-a-diagnostics` (`git log --all --name-only`).
- There is no `final-review/` directory in history.
- `CITATIONS.md:3-4` states it directly: the report is
  `NeutroRAG_First_Review_Report_final.docx`, *"edited outside this
  repository."*

So the brief's document-level tasks — the title-page placeholder, Table 10.1 on
page 49, Table 6.1 on page 20, objectives numbered 15–21, PDF regeneration and
visual inspection — **cannot be performed here at all.** I have not simulated
them.

Three items the brief attributes to the report have **no basis in this
repository whatsoever**, and I flag them as things to check in the document
itself:

| brief claims | repository says |
|---|---|
| an inventory spanning **RQ1–RQ6** | ~~`PREREGISTRATION.md` defines **RQ1–RQ4 only**. No RQ5, no RQ6, anywhere~~ — **superseded, see the correction below**: the labels differ between the documents and the report's RQ5/RQ6 are the two halves of this repository's RQ3 |
| **"simulated-cascade" findings** | no such artefact, script, result or mention exists |
| Tables 6.1 / 10.1, pages 20 / 49 | the report is not here to contain them |

~~If the report contains RQ5, RQ6 or a simulated cascade, those claims are not
supported by any committed evidence and must be substantiated from elsewhere or
removed.~~

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


## 2. Deliverables and exact paths

| artefact | path | status |
|---|---|---|
| Audit ledger | `submission/AUDIT_LEDGER.md` | complete |
| Registered-test inventory | `submission/TEST_INVENTORY.md` | complete |
| Verification log | `submission/VERIFICATION_LOG.md` | complete |
| This report | `submission/FINAL_AUDIT_REPORT.md` | complete |
| Claim verifier (new script) | `experiments/verify_report_claims.py` | 31 PASS / 0 MISMATCH |
| Demo **transcript** | `submission/demo/DEMO_TRANSCRIPT.md` | genuine, 7 beats, all exit 0 — **not a video** |
| Demo capture script | `submission/demo/capture_demo_session.py` | reproducible |
| Raw logs | `submission/logs/*.txt` | 4 files |
| Claims manifest (pre-existing) | `results/claims.md` | **66 claims, 29 pre-registered** |
| **Demo video** | — | **DOES NOT EXIST** |
| **Final PDF** | — | **not in this repository** |

## 3. Blockers A–G

| ID | Status | Resolution |
|---|---|---|
| **A** demo link | **UNRESOLVED — needs human** | No recorder (`ffmpeg`, `asciinema`, `script`, `termtosvg` absent), no capture library beyond `PIL.ImageGrab`, no audio. Decisively, commands run in a non-interactive shell with **no visible window**: a screen capture would have recorded the desktop while execution happened invisibly, which is a fabricated demo, and would have risked exposing a desktop I cannot inspect first. **No URL was invented.** |
| **B** checklist "Pending" | **BLOCKED** | The checklist is in the absent report. Its underlying facts are now verified in `VERIFICATION_LOG.md` so it can be filled accurately by hand. The demo row must remain **not met**. |
| **C** manifest counts 46/66/29 | **RESOLVED** | All three are genuine; none is an error. The manifest grew with each registration: 46/9 → 53/16 → 57/20 → 61/24 → **66/29**. 46 and 66 are the same metric at different commits; 29 is the pre-registered subset. **Use 66 claims, 29 pre-registered, 37 post-hoc.** |
| **D** test counts | **RESOLVED, 2 disclosures** | **10 registered, 9 executed, 3 positive** (RQ1, RQ3a, Addendum 6). Reconciles exactly — but RQ4 was **never run**, and RQ3 is counted as two separately-criterioned halves. Both must be stated. See `TEST_INVENTORY.md`. |
| **E** reference arithmetic | **RESOLVED** | The report's **17 is wrong**. `CITATIONS.md` records **18** checked, 15 correct, 3 with errors — and 15 + 3 = 18. **Change 17 → 18.** The `[11]` and `[18]` qualifications are preserved. |
| **F** CONFLICTS denominator | **RESOLVED** | **n = 80 is correct; the stated reason is wrong.** 458 release → 100 sampled (quota 40/40/10/10) → 80 eligible, because Complementary (10) and Outdated (10) carry **no expected action**. The 5 Misinformation instances were **never sampled** — 0 appear in the CSV — so "100 minus 5" would wrongly imply 95. Recomputed: **16/80 = 0.2000**, CI [0.112, 0.288]. |
| **G** broken numbering | **BLOCKED** | Lists are in the absent report. No numbering defect exists in any repository document. Scientific identifiers (RQ1–RQ4, H1, H2) must not be renumbered. |

## 4. Claims reproduced

`python experiments/verify_report_claims.py` → **exit 0, 31 PASS, 0 MISMATCH, 3 UNVERIFIABLE.**

| § | claim | asserted | recomputed | status |
|---|---|---|---|---|
| A | FEVER `argmax(T,I,F)` / raw | 0.6650 / 0.7550 | 0.6650 / 0.7550 | PASS |
| A | exact McNemar p | 4.0 × 10⁻⁵ | **4.005 × 10⁻⁵** (b=1, c=19) | PASS |
| B | recoverable-miss EM | 0.1486 → 0.2432 | identical, n=222 | PASS |
| B | McNemar p | 0.0011 | 0.0011 (b=30, c=9) | PASS |
| C | AUROC `r` retrieval success | 0.5958 | 0.5958; `max_relevance` 0.5004; DeLong p 0.0149 | PASS |
| D | hard-retrieval AUROC | 0.6032 [0.5331, 0.6733] | identical, n=400, CI excludes 0.5 | PASS |
| E | H2 checks / gate | 0.7775; 0.2900 vs 0.3075 | identical; non-significant (p=0.2295) | PASS |
| F | fourth-sample paired DeLong | p = 0.0844 | 0.0844 (z=1.725, n=367/400) | PASS |
| F | four samples disjoint | — | **pairwise disjoint**, 0 overlaps, 400/600/400/400 | PASS |

**The brief's `p = 4.0 × 10⁻⁵` is correct** and is more informative than the
project's own scripts, which print it rounded as `0.0000`. I recommend the
report keep the brief's form.

## 5. Commands executed, with real exit codes

| command | exit | outcome |
|---|---:|---|
| `python -m pytest -q` | **0** | 157 passed, 6 skipped, 3 xfailed |
| `python experiments/reproduce_all.py` | **0** | all 6 invoked analysers completed cleanly |
| `python experiments/make_claims_manifest.py` | **0** | 66 claims, 29 pre-registered |
| `python experiments/verify_report_claims.py` | **0** | 31 PASS / 0 MISMATCH |
| `sha256sum fever_run_200_gateA.csv` | 0 | `70a3a783…84c6b7` — matches the cited value |
| `python submission/demo/capture_demo_session.py` | **0** | 7 beats, all exit 0 |

## 6. Counts, with definitions

**Manifest:** 66 claims, **29 pre-registered**, 37 post-hoc. "Pre-registered"
means the claim answers a prediction recorded before the data it tests existed
(Phase 3 primary, Addendum 4, RQ2-as-proposed, Addendum 5, Addendum 6).

**Registered tests:** 10 registered · 9 executed · **3 positive** · 1 partial ·
1 directional · 4 negative. "Positive" = registered criterion met *and*
α = 0.05 reached. RQ4 registered but never run; RQ3 counted as two halves.

## 7. References

**18** post-2023 references checked (13 Oct 2026, against ACL Anthology, arXiv,
Springer, Oxford ORA, NeurIPS, ICLR, UNM, alphaXiv); **15** correct as written;
**3** carried errors (`[1]` incomplete author list, `[10]` wrong venue/year/title,
`[11]` truncated authors + missing locators). 15 + 3 = 18. **Two qualifications
remain open and must stay:** `[11]`'s full author list (Springer truncates it;
needs the PDF) and `[18]`, a 1998 monograph pre-dating DOIs. Metadata was **not
re-fetched in this pass** — treated as previously verified, not re-verified.

## 8. Demo

| property | value |
|---|---|
| Video | **does not exist** |
| Link | **does not exist** — none invented |
| Best available artefact | `submission/demo/DEMO_TRANSCRIPT.md`, 305 lines, 16 KB |
| Nature | genuine timestamped transcript; 7 beats, **all exit 0**; no output reconstructed |
| Labelling | the file's own header states it is **NOT** the required video |
| Duration / resolution / audio | **n/a — no video to inspect** |

### The exact remaining step, for a human

1. On a machine with a desktop session, open a terminal at the repository with a
   large font.
2. Run the four beats of `DEMO_SCRIPT.md`. All are fast: `analyse_hotpot_gate.py`
   ≈ 0.1 s, `diagnose_r_depth.py` ≈ 7 s. **Do not** run
   `diagnose_r_components.py` on camera — it re-embeds the corpus and takes
   ≈ 62 s; its figure is read from `results/claims.md` instead.
3. Record with OBS, Windows Game Bar (`Win+Alt+R`), or Zoom with narration.
4. Target 3–6 minutes; point at `0.3025 vs 0.3225`, then `1.11e-16`, then
   `eta^2 = 0.9144`, then the Addendum 6 primary `0.6032 [0.5331, 0.6733]`.
5. Upload, set link-visible-to-anyone, and **open it in a private window** to
   confirm no permission prompt.
6. Replace the title-page placeholder and set the checklist's demo row to met.

## 9. Remaining limitations and unavailable evidence

- **No video, no link** — mandatory, unmet.
- **Report absent** — all document-level fixes (B, G, title page, PDF) blocked.
- **RQ4 never run** — no cost or parameter comparison is claimed anywhere.
- **Faithfulness, citation accuracy, AURC, ECE, Brier never run** — Phases 4–5
  halted by RQ3b's stop condition. `eval.py`'s AURC and ECE carry two confirmed
  defects (pinned in `tests/test_eval_calibration.py`); because Phase 5 never
  ran, **no reported number depends on either**.
- **Reference metadata not re-fetched** this pass.
- **Scope**: one retriever, one reader, HotpotQA + FEVER + CONFLICTS only.
- **Extractive-reader ceiling**: EM capped at 0.938 overall, 0.692 on comparison
  questions. No EM here may be read against 1.0.

These are legitimate scientific disclosures and were **preserved, not removed.**

## 10. Scientific integrity checks

- No negative result was softened. Phase 3 and H2 remain refutations; H1 remains
  a failed replication; the three retractions stand.
- No non-significant result is described as significant. H2's p = 0.2295, the
  Addendum 6 paired p = 0.0844 and the H3 gate p = 0.2266 are all reported as
  non-significant.
- Exploratory work stays separated: 37 of 66 claims are marked `NO-POST-HOC`.
- No figure was tuned, no hypothesis rewritten, no placeholder fabricated.

## 11. Git status

**Before:** `main` @ `2460554`, in sync with `origin/main`, **working tree clean
(0 files)** — so no pre-existing user work could be lost.

**After:** `results/claims.md` modified (one embedded commit-hash line, expected
churn); `experiments/verify_report_claims.py` and `submission/` new and
untracked.

**No commit, no push, no release, no remote change.** The brief withholds
authorization; everything is left in the working tree for review.

Frozen artefact intact: `fever_run_200_gateA.csv` still hashes to
`70a3a783334c346baca6bd58116b9d2e99f1bd12674edfdce0e359ef3984c6b7`.

## 12. Action required from you

1. **Record and host the demo** (§8), then update the report's title page and
   checklist. *Blocking.*
2. **Apply four corrections to the report**, all now evidenced:
   manifest **66 / 29**; references **17 → 18**; CONFLICTS **n = 80 with the
   corrected reason** (Complementary + Outdated, not Misinformation); test counts
   **10 registered / 9 executed / 3 positive** with both caveats.
3. **Check the report for any simulated-cascade claim** — that one has no
   repository support. **RQ5/RQ6 no longer need checking:** they are the two
   halves of this repository's RQ3 under the report's own numbering, and the
   earlier finding against them is superseded above.
4. **Authorize a commit** of `submission/` and `experiments/verify_report_claims.py`
   if you want this audit in version control.

## 13. Acceptance criteria

| criterion | met |
|---|---|
| Final report clearly identified | **NO** — not in this repository |
| Demo video is a genuine recording | **NO** — does not exist |
| Demo inspected and plays | **NO** |
| Demo link verified, or requirement disclosed | **disclosed, not met** |
| Title-page placeholder removed | **NO** — blocked |
| Reproducibility checklist matches reality | **blocked** (facts verified) |
| Manifest counts sourced and consistent | **YES** |
| Test/positive counts sourced and consistent | **YES** |
| Reference arithmetic reconciles | **YES** |
| CONFLICTS denominator explained and correct | **YES** |
| Broken numbering fixed | **blocked** — none in repo documents |
| Metrics and p-values traced to evidence | **YES** — 31/31 |
| Failed hypotheses and limitations preserved | **YES** |
| Incomplete evaluations disclosed | **YES** |
| PDF regenerated and inspected | **NO** — blocked |
| No accidental placeholders or contradictions | **YES** |
| Verification commands documented with exit codes | **YES** |
| No unrelated user changes lost | **YES** — tree was clean at start |
| No unauthorized commits or pushes | **YES** — none made |
