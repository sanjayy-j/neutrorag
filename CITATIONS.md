# Citation corrections

Traceable record of citation errors found in `NeutroRAG_First_Review_Report_final.docx`.
The .docx is edited outside this repository; this file is the audit trail for what
was changed and against which authority.

Standing rule (supervisory memo, Section 6): *verify every citation against the
publisher record before it is typeset, not after.*

---

## Status: verification COMPLETE (13 October 2026)

Memo finding **F13 is closed.** All **18** post-2023 references — `[1]`–`[17]` and
`[19]` — were checked against the publisher record on 13 October 2026. `[18]` is a
1998 monograph and is treated separately below.

| | count | references |
|---|---:|---|
| checked | 18 | `[1]`–`[17]`, `[19]` |
| correct as written | 15 | all except the three below |
| **carried errors** | **3** | `[1]`, `[10]`, `[11]` |

**Every reference now carries a DOI or arXiv ID**, with the single documented
exception of `[18]`.

Two items remain genuinely open and are recorded as such rather than closed —
see *Still open* at the end. Neither blocks typesetting; both are disclosed.

### Authorities consulted

| source | references |
|---|---|
| ACL Anthology | 2, 3, 4, 5, 6, 8, 14, 16, 17, 19 |
| arXiv | 1, 9 |
| Springer | 11 |
| Oxford ORA (for the *Nature* record) | 7 |
| NeurIPS proceedings | 15 |
| ICLR proceedings | 13 |
| UNM digital repository | 10 |
| alphaXiv | 12 |

---

## [1] Edge et al. — INCOMPLETE AUTHOR LIST

**Error.** The paper has **ten** authors; the citation listed **eight**. Two were
dropped from the middle of the list:

> **Dasha Metropolitansky** and **Robert Osazuwa Ness**, between *S. Truitt* and
> *J. Larson*.

**Fix.** Insert both names in that position, preserving author order. Verified
against the arXiv record.

This is the kind of error a reader cannot detect and a tool will not flag: the
citation resolves correctly, the venue and year are right, and the first and last
authors are right. Only a field-by-field comparison against the publisher record
catches it — which is the reason the standing rule above exists.

## [10] NCMPy — WRONG VENUE, WRONG YEAR, TRUNCATED TITLE

**As printed in the report**

> Neutrosophic Sets and Systems … 2023

**Correct, verified against the publisher record** (`digitalrepository.unm.edu/math_fsp/677`):

> [10] I. Kandasamy, D. Arumugam, A. Rathore, A. Arun, M. Jain, W. B. Vasantha, and
> F. Smarandache, "NCMPy: A modelling software for neutrosophic cognitive maps based
> on Python package," *Neutrosophic Systems with Applications*, vol. 13, 2024,
> doi: 10.61356/j.nswa.2024.114.

Three separate errors, not one:

| field | report | correct |
|---|---|---|
| venue | *Neutrosophic Sets and Systems* (NSS) | *Neutrosophic Systems with Applications* (**NSWA**) |
| year | 2023 | **2024** |
| title | truncated | "… based on Python package" |

NSS and NSWA are different journals. The memo (F13) flagged this as a venue error;
the venue is in fact only one of the three fields that is wrong.

**Status: corrected.**

**On the year, and a correction to something reported earlier in this project.**
An earlier session resolved the DOI through Crossref, which returns an `issued`
date-part of **2023**, and that was reported here as a discrepancy against the
memo's 2024. The publisher record is the authority and it says 2024 — so the
memo is right and **Crossref's issued-date is the outlier**, most likely
publisher-side metadata drift (note the DOI suffix itself encodes `2024`).
Use 2024. The earlier note casting doubt on it is withdrawn.

## [11] Meenakshi et al. — TRUNCATED AUTHOR LIST, MISSING LOCATORS

**Error, two parts.**

1. **Authors.** The citation listed **two** authors of **at least six**.
2. **Locators.** Volume, issue, article number and DOI were all absent.

**Fix.** Add the locators, verified against the Springer record:

| field | value |
|---|---|
| volume | **11** |
| issue | **9** |
| article | **413** |
| doi | **10.1007/s40747-025-02029-x** |

The author list is **not** fixed by this entry — see below.

## [9] SURE-RAG — verified, no change

arXiv 2605.03534, including the 0.9075 / 0.6516 / 0.7284 figures. Confirmed by
the memo against the publisher record; nothing to correct.

---

## Still open

Two items are genuinely unresolved. They are recorded here rather than quietly
closed, because the standing rule is about the authority consulted, not about the
citation merely looking complete.

### [11] — complete author list

The citation names two authors of at least six. **Springer truncates the author
list on the article landing page, and no indexed record exposes the full list**,
so it cannot be completed from metadata. It must be **read off the article PDF**
and transcribed by hand.

Until that is done, cite as *A. Meenakshi et al.* — the `et al.` is accurate and
honest, where naming two authors as if they were the whole list is not.

### [18] — Smarandache (1998), no DOI

A 1998 monograph, which **predates DOI assignment**. It therefore cannot satisfy
"every reference carries a DOI or arXiv ID", and this is a property of the
artefact's age rather than a defect in the citation. Cite with publisher, place
and year, and do not manufacture an identifier for it.

This is the one documented exception to the DOI/arXiv-ID rule above.
