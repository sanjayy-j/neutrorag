# Verification log

Every command below was executed in this repository. Exit codes are the actual
ones returned, not assumed. Raw output is in [`logs/`](logs/).

Repository at `2460554`, branch `main`, in sync with `origin/main`.

## Commands executed

| # | command | exit | outcome | log |
|---|---|---:|---|---|
| 1 | `python -m pytest -q` | **0** | **157 passed, 6 skipped, 3 xfailed** | `logs/pytest.txt` |
| 2 | `python experiments/reproduce_all.py` | **0** | all 6 invoked analysis scripts completed cleanly | `logs/reproduce_all.txt` |
| 3 | `python experiments/make_claims_manifest.py` | **0** | wrote `results/claims.md` — **66 claims, 29 pre-registered** | `logs/manifest.txt` |
| 4 | `python experiments/verify_report_claims.py` | **0** | **31 PASS, 0 MISMATCH, 3 UNVERIFIABLE** | `logs/verify_report_claims.txt` |
| 5 | `python experiments/sweep_pool_size.py` | 0 | the pooled-index sweep underlying Addendum 6 (run during that experiment) | — |
| 6 | `sha256sum fever_run_200_gateA.csv` | 0 | `70a3a783…84c6b7` — **matches the value cited in `RESULTS.md`** | `logs/gateA_sha256.txt` |
| 7 | `python submission/demo/capture_demo_session.py` | **0** | 7 beats, **all exit 0**, 305-line transcript | `demo/DEMO_TRANSCRIPT.md` |
| 8 | `python demo.py` | 0 | mock-backend scaffold; banner and per-row `[mock]` markers present | — |

## The 6 analysis scripts `reproduce_all.py` invokes

All completed cleanly (command 2):

```
[        ok] Phase 3 gate analysis
[        ok] Phase 3 by question type
[        ok] r vs retrieval depth
[        ok] regenerate the claims manifest
[        ok] r component diagnostics (~60s: re-embeds the corpus)
[        ok] Phase 2 retrieval metrics (~60s: re-embeds the corpus)
```

## Figures verified against raw artefacts

`experiments/verify_report_claims.py` recomputes each from the committed CSVs
and compares it to the value the audit brief attributes to the report.

| group | items | result |
|---|---:|---|
| A — main FEVER comparison | 4 | PASS (incl. exact McNemar **4.005e-5**) |
| B — recoverable-miss subset | 4 | PASS |
| C — retrieval-success AUROC | 3 | PASS |
| D — hard-retrieval primary | 5 | PASS |
| E — construction / gate | 4 | PASS |
| F — fourth disjoint sample | 2 | PASS |
| CONFLICTS denominator | 5 | PASS |
| manifest + references | 4 | PASS |
| **total** | **31** | **31 PASS, 0 MISMATCH** |

Sample independence was verified directly: Phase 3 (400), H1 (600), H2 (400) and
H3 (400) are **pairwise disjoint** on question id, 0 overlaps across all six
pairs.

## Not verified, and why

| item | reason |
|---|---|
| Report tables, page numbers, numbering, title page, abstract | **the report is not in this repository** — never committed on any branch; `CITATIONS.md:3` records it as a `.docx` edited elsewhere |
| RQ5, RQ6 | **do not exist**; `PREREGISTRATION.md` defines RQ1–RQ4 only |
| "simulated-cascade" findings | **no such artefact, script or result** in the repository |
| Reference metadata against live publisher records | not re-fetched in this pass; `CITATIONS.md` records the 13 Oct 2026 verification against ACL Anthology, arXiv, Springer, Oxford ORA, NeurIPS, ICLR, UNM and alphaXiv. Treated as **previously verified, not re-verified here** |
| RQ4 cost comparison | **never run** — Phases 4–5 were halted by RQ3b's stop condition. No parameter-count or wall-clock figure is claimed anywhere |
| Faithfulness, citation accuracy, AURC, ECE, Brier | **never run** for the same reason. `eval.py`'s AURC and ECE carry two confirmed defects pinned in `tests/test_eval_calibration.py`; since Phase 5 never ran, no reported number depends on either |
| Demo video playback, duration, resolution, audio | **no video exists** — see Blocker A |
| Demo link accessibility | **no link exists**; none was invented |

## Known expected churn

`results/claims.md` embeds the commit its generator ran at, so regenerating it
after any later commit rewrites exactly one line. The file says so in its own
header. Verified: the only diff after command 3 was that line.
