# NeutroRAG

**A neutrosophic trust layer for retrieval-augmented generation.**
NeutroRAG scores every candidate answer with three *independent* signals —
**Truth (T)**, **Indeterminacy (I)**, and **Falsity (F)** — and acts on each
differently: it *expands retrieval* when evidence is missing, *surfaces the
conflict* when sources contradict each other, and *abstains* only when
indeterminacy is irreducible. A single confidence score can't tell "I found no
evidence" apart from "I found conflicting evidence" — and so it hallucinates on
both. NeutroRAG keeps them apart.

> Positioning: this is **trustworthy / uncertainty-aware retrieval**. The
> neutrosophic (T, I, F) formalism is the *mechanism*, not the headline.

---

## Why (T, I, F) and not a probability or a fuzzy score

In a single-valued neutrosophic set, T, I and F are independent memberships in
`[0, 1]` that **need not sum to 1**. That is the entire theoretical
justification for this project: probability and fuzzy membership collapse
*unsupported* and *contradicted* onto one axis, whereas neutrosophy is the
minimal formalism that keeps them separate — so `T + I + F` can exceed 1 exactly
when some sources support a claim and others refute it.

```
r      = retrieval sufficiency in [0,1]          # did we retrieve anything on-topic?
e,n,c  = relevance-weighted mean NLI             # entailment / neutral / contradiction
disagree = 2 * min(e, c)                         # high only if support AND conflict co-occur

T = r * e                        # support needs on-topic evidence that entails
F = r * c + lambda * disagree    # contradiction from conflict + cross-source disagreement
I = 1 - r + r * n                # structural (no evidence) + semantic (neutral) doubt
```

## The decision policy (the core contribution)

| Dominant signal            | Action                    | Why                                             |
|----------------------------|---------------------------|-------------------------------------------------|
| High T, low I, low F       | **Answer**                | corroborated                                    |
| High F (with some T)       | **Surface contradiction** | conflict is *not* fixed by retrieving more      |
| High I, budget remaining   | **Expand retrieval**      | missing evidence is often *recoverable*         |
| High I, budget exhausted   | **Abstain**               | indeterminacy is irreducible here               |
| weak / mixed               | **Hedge**                 | answer with an explicit caveat                  |

The staged loop — *indeterminacy triggers action, not just refusal* — is what
separates NeutroRAG from prior selective-prediction / abstention work.

---

## Quickstart

```bash
pip install -e .        # or: pip install numpy
python demo.py          # FEVER premise check + policy + ablation (offline mock NLI)
pytest -q               # 17 unit tests
```

```python
from neutrorag import Evidence, NeutroPolicy, NLIRetrievalScorer

scorer = NLIRetrievalScorer()          # mock NLI by default (offline)
policy = NeutroPolicy()

evidence = [Evidence("Sharks are not mammals; they are fish.", relevance=0.9)]
decision = policy.run("Sharks are mammals.", evidence, scorer.score)
print(decision.action, decision.tif)   # Action.SURFACE_CONTRADICTION TIF(T=0.09, I=0.10, F=1.00)
```

## Project layout

```
neutrorag/
  core.py      # TIF triple: independence, confidence, collapse() for the ablation
  scoring.py   # NLIRetrievalScorer + pluggable NLI backends (Mock / HuggingFace)
  policy.py    # NeutroPolicy: the five-way staged decision controller
  eval.py      # risk-coverage, ECE, failure-type attribution, fuzzy-collapse ablation
  fever.py     # FEVER -> (T,I,F) proof-of-concept (SUPPORTS/REFUTES/NEI == T/F/I)
tests/         # unit tests for core, policy, scoring
demo.py        # runnable end-to-end demonstration
```

## Wiring in the real components

The scaffold is deliberately dependency-light so the *contribution* (scoring +
policy) is testable in isolation. Swap in production pieces as you go:

- **Real NLI** — `from neutrorag import HFNLIBackend; scorer = NLIRetrievalScorer(nli=HFNLIBackend())`
  (downloads `DeBERTa-v3-...-mnli-fever-anli` on first use).
- **Real FEVER** — replace `fever.load_sample()` with `fever.load_fever_jsonl(path)`.
- **GraphRAG retriever** — feed your retrieved passages in as `Evidence(text, relevance)`;
  the `expand_fn` callback is where you plug re-querying / subgraph widening
  (LlamaIndex or Microsoft GraphRAG + Neo4j).

## Evaluation

- **Failure-type attribution** (the distinctive metric): does `argmax(T, I, F)`
  match the gold SUPPORTS / NEI / REFUTES channel? A collapsed (I+F merged)
  model *structurally cannot* separate missing-evidence from contradiction.
- **Selective prediction**: AURC + selective accuracy at fixed coverage.
- **Calibration**: expected calibration error (ECE).
- **Fuzzy-collapse ablation** (`eval.fuzzy_collapse_ablation`): full neutrosophic
  vs. I+F merged — the experiment that justifies the whole design.

## Roadmap

- [x] **M1–M2** — core library: TIF, scorer, five-way policy, FEVER PoC, tests ← *you are here*
- [ ] **M2** — real NLI backend + GraphRAG baseline on HotpotQA / 2WikiMultiHopQA
- [ ] **M3** — full experiments + **fuzzy-collapse ablation** → go/no-go checkpoint
- [ ] **M4** — FEVER failure-type attribution at scale + open-source release
- [ ] **M5** — learned neutrosophic aggregator (ANFIS/MLP) + clinical case study
- [ ] **M6** — paper + provisional-patent draft + demo dashboard

## Status

Research scaffold. All numbers printed by `demo.py` come from a **deterministic
mock NLI backend** and are illustrative only — swap in a real NLI model and
dataset for paper-grade results.

## License

MIT.
