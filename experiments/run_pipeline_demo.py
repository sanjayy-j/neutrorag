"""Full NeutroRAG pipeline, end-to-end, with every stage of the architecture
diagram actually executed by real code:

    Query -> KnowledgeGraph retrieval -> DenseRetriever retrieval
          -> NLIRetrievalScorer (T,I,F) -> NeutroPolicy (5-way decision,
             re-retrieving under indeterminacy via expand_fn)
          -> Responder (final natural-language response)

Unlike demo.py (which only exercises the trust layer on hand-constructed
Evidence) and run_fever_eval.py (which only exercises the trust layer on FEVER
data), this script is the one that runs the *whole* diagram: retrieval is real
(graph.py / retrieval.py), scoring/policy are the frozen core, and generation is
real (generation.py).

    python experiments/run_pipeline_demo.py
    python experiments/run_pipeline_demo.py --nli real --responder ollama
    python experiments/run_pipeline_demo.py --claim "Drug X reduces the risk of Condition Y in low-risk patients."

Defaults are offline-fast (mock NLI, template responses); the dense retriever's
embedding model still downloads on first run (~90MB, cached after that).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neutrorag import (  # noqa: E402
    DenseRetriever, Document, KnowledgeGraph, Fact,
    HFNLIBackend, MockNLIBackend, NLIRetrievalScorer,
    NeutroPolicy, PolicyConfig,
    TemplateResponder, OllamaResponder,
)

DEFAULT_CLAIM = "Drug X reduces the risk of Condition Y in high-risk patients."


def build_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_facts([
        Fact("Drug X", "studied_in", "Trial A",
             "In a randomized controlled trial, Drug X reduced the risk of Condition Y "
             "by 30 percent in high-risk patients."),
        Fact("Drug X", "studied_in", "Meta-analysis M",
             "A large meta-analysis found no significant reduction in Condition Y risk "
             "among high-risk patients taking Drug X."),
        Fact("Drug X", "prescribed_to", "high-risk patients",
             "Drug X is commonly prescribed to high-risk patients for a range of conditions."),
    ])
    return g


def build_retriever() -> DenseRetriever:
    r = DenseRetriever()
    r.index([
        Document("bg1", "Drug X was approved for cardiovascular risk reduction over a decade ago."),
        Document("bg2", "Condition Y prevalence increases significantly with age."),
        Document("bg3", "The stock market fell sharply on Tuesday amid inflation concerns."),
    ])
    return r


def hr(title: str) -> None:
    print("\n" + "-" * 72)
    print(title)
    print("-" * 72)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--claim", type=str, default=DEFAULT_CLAIM)
    p.add_argument("--nli", choices=["mock", "real"], default="mock")
    p.add_argument("--nli-model", type=str, default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--responder", choices=["template", "ollama"], default="template")
    p.add_argument("--llm-model", type=str, default="llama3.2")
    p.add_argument("--graph-hops", type=int, default=1)
    p.add_argument("--k", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    graph = build_graph()
    retriever = build_retriever()

    print("=" * 72)
    print("NeutroRAG full pipeline demo -- ONE TRACE, NOT AN EVALUATION")
    print("=" * 72)
    print("This runs a single claim end to end to show the stages wiring up.")
    print("It reports no accuracy and is not a result. With the default")
    print("--nli mock the backend is a lexical heuristic, not a model.")
    print("The project's measured numbers are in RESULTS.md.")
    print("=" * 72)
    print("NeutroRAG full pipeline demo")
    print(f"  claim      : {args.claim}")
    print(f"  NLI        : {'MockNLIBackend (offline)' if args.nli == 'mock' else args.nli_model}")
    print(f"  retriever  : DenseRetriever(SentenceTransformerBackend) + KnowledgeGraph (in-memory)")
    print(f"  responder  : {'TemplateResponder' if args.responder == 'template' else 'OllamaResponder (' + args.llm_model + ')'}")

    def retrieve(claim: str, hops: int, k: int):
        graph_ev = graph.retrieve(claim, hops=hops)
        dense_ev = retriever.retrieve(claim, k=k)
        return graph_ev + dense_ev

    hr("1. Retrieval")
    initial_evidence = retrieve(args.claim, args.graph_hops, args.k)
    for e in initial_evidence:
        print(f"  [relevance={e.relevance:.2f}] {e.text}")
    if not initial_evidence:
        print("  (no evidence retrieved -- graph has no matching entities, or corpus is off-topic)")

    nli = MockNLIBackend() if args.nli == "mock" else HFNLIBackend(model=args.nli_model, device=args.device)
    scorer = NLIRetrievalScorer(nli=nli)
    policy = NeutroPolicy(PolicyConfig(max_expansions=2))

    last_evidence = [initial_evidence]  # tracks the evidence set behind the most recent score_fn call

    def expand_fn(claim, evidence, step):
        wider_hops = args.graph_hops + step + 1
        wider_k = args.k + (step + 1) * 3
        print(f"\n  [expanding: graph hops={wider_hops}, dense k={wider_k}]")
        last_evidence[0] = retrieve(claim, wider_hops, wider_k)
        return last_evidence[0]

    hr("2. Neutrosophic trust layer (scoring + staged policy)")
    decision = policy.run(args.claim, initial_evidence, scorer.score, expand_fn)
    for i, step in enumerate(decision.history):
        print(f"  step {i}: {step.tif}  action={step.action.value}")
        print(f"          rationale: {step.rationale}")
    print(f"\n  FINAL ACTION: {decision.action.value}  (expansions_used={decision.expansions_used})")

    hr("3. Response generation")
    responder = TemplateResponder() if args.responder == "template" else OllamaResponder(model=args.llm_model)
    response = responder.respond(args.claim, last_evidence[0], decision)
    print(response)


if __name__ == "__main__":
    main()
