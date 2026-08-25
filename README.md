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
pip install -e .                          # numpy + networkx (core deps)
python demo.py                            # FEVER premise check + policy + ablation (offline mock NLI)
python experiments/run_pipeline_demo.py   # full pipeline: graph + dense retrieval -> trust layer -> response
pytest -q                                 # 35 unit tests
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
  core.py        # TIF triple: independence, confidence, collapse() for the ablation  [frozen]
  scoring.py     # NLIRetrievalScorer + pluggable NLI backends (Mock / HuggingFace)
  policy.py      # NeutroPolicy: the five-way staged decision controller              [frozen]
  eval.py        # risk-coverage, ECE, failure-type attribution, fuzzy-collapse ablation
  fever.py       # FEVER -> (T,I,F) proof-of-concept (SUPPORTS/REFUTES/NEI == T/F/I)
  retrieval.py   # DenseRetriever: real sentence-embedding + cosine-similarity retrieval
  graph.py       # KnowledgeGraph: in-memory multi-hop graph retrieval (networkx)
  generation.py  # Responder: TemplateResponder (deterministic) / OllamaResponder (real LLM)
tests/           # unit tests for every module above (offline, no network/model download)
demo.py                          # trust-layer-only demonstration (offline mock NLI)
experiments/run_fever_eval.py    # trust layer on real FEVER data + real NLI model
experiments/run_pipeline_demo.py # the full pipeline: retrieval -> trust layer -> generation
```

## Wiring in the real components

The core trust layer (`scoring.py` + `policy.py`) is deliberately dependency-light
and testable in isolation. Everything around it is now real, working code too --
scoped deliberately for a project at this size rather than left as a placeholder:

- **Real NLI** — `from neutrorag import HFNLIBackend; scorer = NLIRetrievalScorer(nli=HFNLIBackend())`
  (downloads `DeBERTa-v3-...-mnli-fever-anli` on first `predict()` call, then caches
  the loaded pipeline; pass `device="cuda"` to use a GPU).
- **Real FEVER** — replace `fever.load_sample()` with `fever.load_fever_hf()` (HuggingFace
  Hub, evidence text included inline -- see below) or `fever.load_fever_jsonl(path)` for
  your own local data.
- **Retrieval** — `neutrorag.retrieval.DenseRetriever`: real sentence embeddings
  (`sentence-transformers`, `all-MiniLM-L6-v2` by default) + cosine similarity over an
  in-memory corpus, producing `Evidence` with genuine relevance scores instead of
  hand-picked ones.
- **Graph retrieval ("GraphRAG")** — `neutrorag.graph.KnowledgeGraph`: an in-memory,
  `networkx`-backed directed graph of (subject, relation, object) facts, each edge
  carrying the sentence that asserts it, with breadth-first multi-hop retrieval and
  hop-distance relevance decay. **Scope, stated plainly**: this is in-memory, not
  Neo4j-backed — a deliberate, permanent design choice at this project's scale, not
  a stand-in awaiting an upgrade. `KnowledgeGraph.add_fact` / `.retrieve` is the
  entire integration surface the rest of the system depends on; a Neo4j-backed
  implementation of that same surface could replace it later without touching
  `scoring.py` or `policy.py`.
- **Response generation** — `neutrorag.generation`: `TemplateResponder` (default,
  deterministic, stdlib-only -- generates the final response per `Action`, so the
  caveat/conflict structure the policy decided on can't be silently dropped) or
  `OllamaResponder` (a real call to a locally running [Ollama](https://ollama.com)
  server's `/api/generate` endpoint, via `urllib` only -- no new dependency; run
  `ollama serve` and `ollama pull llama3.2` to use it).
- **`expand_fn`** — the policy's retrieval-expansion callback now has a real thing to
  call: see `experiments/run_pipeline_demo.py`, where it re-queries both
  `KnowledgeGraph` (wider hop radius) and `DenseRetriever` (larger *k*) under high
  indeterminacy.

## Reproducing the FEVER premise check

`demo.py` uses the mock NLI backend and an embedded 12-example sample -- illustrative
only. For genuine numbers, `experiments/run_fever_eval.py` loads real FEVER claims and
scores them with a real HuggingFace NLI model:

```bash
pip install -e ".[nli]"     # torch, transformers, datasets
python experiments/run_fever_eval.py --limit 200
# optional: --model <hf-model-id> --device cpu|cuda
```

Raw FEVER (Thorne et al., 2018) only records evidence as `(wikipedia_page,
sentence_id)` pointers, not sentence text, so scoring it means joining against the
~5M-sentence Wikipedia dump the annotators used. `neutrorag.fever.load_fever_hf`
sidesteps that by loading
[`copenlu/fever_gold_evidence`](https://huggingface.co/datasets/copenlu/fever_gold_evidence)
from the HuggingFace Hub, which bundles each claim with resolved evidence *text*
inline: gold evidence for SUPPORTS/REFUTES, and a system-retrieved
topically-related-but-non-confirming sentence for NOT ENOUGH INFO (from the
adversarial-claims pipeline in Atanasova et al., EMNLP 2020). See the docstring in
`neutrorag/fever.py` for the exact row format.

## Reproducing the full pipeline (retrieval -> trust layer -> generation)

`experiments/run_pipeline_demo.py` is the one script that runs every stage of the
architecture end-to-end -- graph + dense retrieval feeding real `Evidence` into the
trust layer, the staged policy re-retrieving under indeterminacy, and a response
generator producing the final text:

```bash
python experiments/run_pipeline_demo.py                            # offline: mock NLI, template response
python experiments/run_pipeline_demo.py --nli real                 # real DeBERTa-v3 NLI model
python experiments/run_pipeline_demo.py --nli real --responder ollama --llm-model llama3.2
python experiments/run_pipeline_demo.py --claim "Your own claim here."
```

The first run downloads the retrieval embedding model (`all-MiniLM-L6-v2`, ~90MB,
cached after that). `--responder ollama` requires `ollama serve` running locally
with the requested model pulled.

## Evaluation

- **Failure-type attribution** (the distinctive metric): does `argmax(T, I, F)`
  match the gold SUPPORTS / NEI / REFUTES channel? A collapsed (I+F merged)
  model *structurally cannot* separate missing-evidence from contradiction.
- **Selective prediction**: AURC + selective accuracy at fixed coverage.
- **Calibration**: expected calibration error (ECE).
- **Fuzzy-collapse ablation** (`eval.fuzzy_collapse_ablation`): full neutrosophic
  vs. I+F merged — the experiment that justifies the whole design.

## Roadmap

- [x] **M1–M2** — core library: TIF, scorer, five-way policy, FEVER PoC, tests
- [x] **M2** — real NLI backend (`HFNLIBackend`) + real FEVER data + real dense/graph
  retrieval + response generation, wired end-to-end ← *you are here*
- [ ] **M2.5** — GraphRAG baseline on HotpotQA / 2WikiMultiHopQA (currently the graph
  is a small hand-built in-memory example, not built from a corpus)
- [ ] **M3** — full experiments at scale + calibration (`eval.risk_coverage` /
  `eval.expected_calibration_error` exist but aren't exercised or tested yet)
- [ ] **M4** — FEVER failure-type attribution at scale + open-source release
- [ ] **M5** — learned neutrosophic aggregator (ANFIS/MLP) + clinical case study
- [ ] **M6** — paper + provisional-patent draft + demo dashboard

## Status

The trust layer (`scoring.py` + `policy.py`) is the frozen research contribution.
Around it, retrieval (`retrieval.py`, `graph.py`), real NLI (`HFNLIBackend`), real
FEVER data (`fever.load_fever_hf`), and response generation (`generation.py`) are
all real, working, tested code — not mocks. Two things stay honestly scaffolded:
`demo.py` still runs on the deterministic mock NLI backend by design, for fast,
offline, illustrative output; and `graph.py`'s knowledge graph is hand-built and
in-memory rather than constructed from a real corpus at scale (see the Roadmap).

## License

MIT.
