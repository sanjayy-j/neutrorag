import numpy as np

from neutrorag.retrieval import DenseRetriever, Document

# Deterministic bag-of-words embedding for offline tests -- no model download,
# no network. Exercises the real ranking/cosine-similarity logic in
# DenseRetriever with a stand-in EmbeddingBackend, the same way test_scoring.py
# exercises NLIRetrievalScorer with MockNLIBackend instead of HFNLIBackend.
_VOCAB = ["paris", "france", "tower", "mammals", "fish", "sharks", "capital"]


class FakeEmbeddingBackend:
    def embed(self, texts):
        vecs = []
        for t in texts:
            words = set(t.lower().split())
            vecs.append([1.0 if w in words else 0.0 for w in _VOCAB])
        return np.array(vecs, dtype=float)


def make_retriever():
    return DenseRetriever(embedder=FakeEmbeddingBackend())


def test_retrieve_ranks_by_similarity():
    r = make_retriever()
    r.index([
        Document("d1", "The Eiffel Tower is in Paris France."),
        Document("d2", "Sharks are fish not mammals."),
        Document("d3", "Paris is the capital of France."),
    ])
    results = r.retrieve("What is the capital of France", k=2)
    assert len(results) == 2
    assert results[0].text == "Paris is the capital of France."


def test_retrieve_returns_evidence_with_relevance_in_unit_interval():
    r = make_retriever()
    r.index([Document("d1", "Sharks are fish not mammals.")])
    results = r.retrieve("Are sharks mammals", k=1)
    assert len(results) == 1
    assert 0.0 <= results[0].relevance <= 1.0


def test_empty_index_returns_no_evidence():
    r = make_retriever()
    assert r.retrieve("anything") == []


def test_add_is_incremental_and_reranks_with_new_documents():
    r = make_retriever()
    r.index([Document("d1", "Paris is the capital of France.")])
    r.add([Document("d2", "Sharks are fish not mammals.")])
    assert len(r.documents) == 2
    results = r.retrieve("Are sharks mammals", k=1)
    assert results[0].text == "Sharks are fish not mammals."
