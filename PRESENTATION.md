# NeutroRAG — Review 1 Runbook

One-page script for tomorrow: pre-flight checks, exact commands, timing, and a
condensed Q&A cheat sheet. Keep this open on a second screen during the review.

## Pre-flight (do this before you walk in)

- [ ] Start Ollama: open a terminal, run `ollama serve` (leave it running). Check it's
      up: `curl http://localhost:11434/api/tags` should return `{"models":[...]}`.
- [ ] Terminal open at the repo root.
- [ ] Sanity check: `pytest -q` → expect `35 passed`. (Don't show this live unless
      asked — it's your own confidence check.)
- [ ] Open the visualization artifact in a browser tab, ready to switch to.
- [ ] Do one full dry run of the commands below, tonight, so nothing is downloading
      live tomorrow (models are already cached from today's session).

## Timing (assume 5–8 minutes total)

1. 30-second pitch — 30s
2. Visualization: architecture + what's real — 60s
3. Live demo #1: default pipeline run — 90s
4. Live demo #2 (if time / if asked): real NLI + real local LLM — 60s
5. Point at the real results chart, state the numbers — 60s
6. Close: contribution + honest limitations — 45s
7. Q&A — use the cheat sheet below

## 1. The 30-second pitch

> "NeutroRAG is a trust layer for RAG systems. Instead of one confidence score,
> it scores every claim on three independent signals — Truth, Indeterminacy,
> Falsity — so it can tell 'I found no evidence' apart from 'I found evidence
> that disagrees with itself,' and act on each differently: answer, expand
> retrieval, surface the conflict, or abstain. The contribution is the scoring
> formula and the five-way decision policy; today it's a real, tested pipeline
> from retrieval through to a generated response."

## 2. Show the visualization

Open the artifact. Walk the pipeline left to right: Query → Graph + Dense
Retrieval → Trust Layer (scoring) → Decision Policy → Response. Say plainly
which stages are the frozen research contribution (scoring + policy) and which
are real supporting infrastructure (retrieval, graph, generation) — don't
present them as the same kind of thing.

## 3. Live demo #1 — default pipeline (offline, fast)

```bash
python experiments/run_pipeline_demo.py
```

Narrate as it prints: retrieval returns ranked evidence (real cosine similarity
+ real graph traversal) → the trust layer aggregates it into (T, I, F) → the
policy picks an action and states why → the responder generates the final text.
With the default claim this lands on `surface_contradiction` — point out that
the response explicitly presents both sides rather than picking one, which is
the entire point of the five-way policy.

## 4. Live demo #2 — real NLI + real local LLM (if time allows)

```bash
python experiments/run_pipeline_demo.py --nli real --responder ollama --llm-model llama3.2:1b
```

Everything is cached from tonight, so this should complete in a few seconds. If
you have time, mention the honest finding from tonight's testing: swapping mock
NLI for the real DeBERTa-v3 model on the *same* evidence changed the final
action (contradiction → expand → abstain), because the mock backend is a crude
lexical heuristic. That's a genuine reason the real backend matters, not a
talking point you made up.

**If asked "is that a real LLM":** yes — `OllamaResponder` makes a real HTTP
call to a locally running Ollama server; nothing about the response is scripted.

## 5. Real results (don't run this live — show the chart)

`experiments/run_fever_eval.py --limit 200` takes about two minutes (loading +
scoring 200 real FEVER claims with the real NLI model) — too slow for a review
slot. Show the results chart instead and state the numbers from memory:

- **Failure-type attribution accuracy: 0.745** — full (T, I, F) model, argmax
  channel vs. gold SUPPORTS/NEI/REFUTES, on 200 real `copenlu/fever_gold_evidence`
  validation examples, real `DeBERTa-v3-base-mnli-fever-anli` NLI.
- **Collapsed (I+F merged) baseline: 0.540.**
- **Delta: +0.205** — the ablation that argues separating indeterminacy from
  contradiction is actually worth something, not just theoretically tidy.

Offer to run it live only if specifically pushed — and say up front it'll take
about two minutes if you do.

## 6. Close

> "The scoring formula and the decision policy are the research contribution,
> and they're frozen and tested. Around them, retrieval, graph traversal, real
> NLI, and response generation are all real, working, tested code as of this
> week — not mocks, and not just a diagram. What's still explicitly scoped: the
> graph is in-memory, not Neo4j-backed, and it's hand-built rather than
> constructed from a real corpus at scale — that's next."

## Q&A cheat sheet (condensed — see chat history for full depth)

- **"Why not fuzzy logic?"** Standard fuzzy sets tie truth/falsity together
  (often F = 1−T); neutrosophic T/I/F are independent, so "strongly supported
  AND strongly contradicted" is a representable state, not forced to the
  middle.
- **"Why not probability / entropy / confidence score?"** All three are
  normalized or scalar — they can't distinguish *no evidence* from
  *conflicting evidence*, which is the entire problem this project targets.
- **"Is this GraphRAG / does it use Neo4j?"** No — `graph.py` is a real,
  working, in-memory graph (`networkx`) with multi-hop retrieval. That's a
  deliberate scope choice for a project this size, not a placeholder; Neo4j
  would be a storage-backend swap behind the same interface, not a rewrite.
  Say this plainly if asked — don't claim Neo4j.
- **"What's mocked vs. real?"** `demo.py` defaults to `MockNLIBackend` by
  design (offline, fast, illustrative). `HFNLIBackend` (real DeBERTa-v3),
  `DenseRetriever` (real sentence-embeddings), `KnowledgeGraph` (real, in-memory),
  and `OllamaResponder` (real local LLM) are all real and tested — verified
  live today, not just written.
- **"What's your contribution vs. infrastructure?"** Contribution: the (T,I,F)
  aggregation formula in `scoring.py` and the five-way staged policy in
  `policy.py` (both frozen). Everything else — retrieval, graph, generation,
  the FEVER loader — is real supporting infrastructure, not the novel part.
- **"Has this already been done?"** Be ready to name intuitionistic fuzzy sets
  (hesitation margin) as the closest prior idea and state the difference
  precisely rather than just asserting novelty — see chat history Part 1 §13
  for the exact framing.
- **"What's the real number?"** 0.745 vs. 0.540, delta +0.205, on 200 real
  FEVER claims, real NLI model — say this from memory, don't fumble for it.

## If something breaks live

- **Ollama not responding** → fall back to the default run (`TemplateResponder`).
  Say plainly: "the Ollama integration is real and tested (`test_generation.py`
  mocks the HTTP layer to verify the real request/response contract), it's just
  not live in this room right now" — true, and not an excuse.
- **No wifi / a download hangs** → run `pytest -q` instead: 35 passing tests is
  concrete proof of real, working code even with nothing live.
- **A number you're asked for doesn't come to mind** → open the visualization
  artifact; the real numbers are on it.
