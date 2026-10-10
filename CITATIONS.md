# Citation corrections

Traceable record of citation errors found in `NeutroRAG_First_Review_Report_final.docx`.
The .docx is edited outside this repository; this file is the audit trail for what
was changed and against which authority.

Standing rule (supervisory memo, Section 6): *verify every citation against the
publisher record before it is typeset, not after.*

---

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

**On the year, and a correction to something reported earlier in this project.**
An earlier session resolved the DOI through Crossref, which returns an `issued`
date-part of **2023**, and that was reported here as a discrepancy against the
memo's 2024. The publisher record is the authority and it says 2024 — so the
memo is right and **Crossref's issued-date is the outlier**, most likely
publisher-side metadata drift (note the DOI suffix itself encodes `2024`).
Use 2024. The earlier note casting doubt on it is withdrawn.

## [9] SURE-RAG — verified, no change

arXiv 2605.03534, including the 0.9075 / 0.6516 / 0.7284 figures. Confirmed by
the memo against the publisher record; nothing to correct.

---

## Remaining

The other eleven references have **not** been independently verified here — the
reference list lives in the .docx and only [9] and [10] were supplied to this
repository. F13's instruction to "verify all thirteen references independently
against the publisher record" is therefore **open**, and no claim is made in this
file about the other eleven.
