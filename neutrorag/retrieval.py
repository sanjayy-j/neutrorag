"""Dense (semantic) retrieval for NeutroRAG -- the 'Dense/vector retrieval' stage
of the architecture.

Before this module, every `Evidence` object fed into `NLIRetrievalScorer` had to
be constructed by hand by the caller. `DenseRetriever` closes that gap: it holds
an in-memory corpus, embeds it with a real sentence-embedding model, and turns a
claim into ranked `Evidence` via cosine similarity -- so `relevance` is a genuine
retrieval score, not a placeholder chosen by the caller.

The embedding model is swapped in behind an `EmbeddingBackend` Protocol, the same
pattern `NLIBackend`/`HFNLIBackend` already use in `scoring.py`: the retriever's
logic (indexing, ranking, the vector store) is the permanent piece; the specific
embedding model is a pluggable backend. `SentenceTransformerBackend` lazily
imports `sentence_transformers` inside `_load()`, so importing this module -- and
`neutrorag` as a whole -- never requires that dependency unless it's actually used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence

import numpy as np

from .scoring import Evidence


class EmbeddingBackend(Protocol):
    """Return an (n, d) array of embeddings for a batch of texts."""

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerBackend:
    """Real sentence-embedding backend (optional heavy dep, downloaded on first use).

    Mirrors HFNLIBackend's lazy-load-and-cache pattern in scoring.py exactly:
    the model is loaded on first `embed()` call, not in `__init__`, and cached
    per (model, device) at the class level.
    """

    _model_cache: dict = {}

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self._model_name = model
        self._device = device
        self._model = None

    def _load(self):
        if self._model is None:
            key = (self._model_name, self._device)
            if key not in SentenceTransformerBackend._model_cache:
                from sentence_transformers import SentenceTransformer  # noqa: local import by design

                SentenceTransformerBackend._model_cache[key] = SentenceTransformer(
                    self._model_name, device=self._device
                )
            self._model = SentenceTransformerBackend._model_cache[key]
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vecs, dtype=float)


@dataclass
class Document:
    """A corpus passage available to be retrieved."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class DenseRetriever:
    """Semantic retrieval over an in-memory corpus: real embeddings + cosine similarity.

    Deliberately in-memory rather than backed by an external vector database --
    for a corpus at research-project scale that's the permanent design, not a
    placeholder; a FAISS/pgvector-backed index would satisfy the same
    `index`/`add`/`retrieve` surface without requiring any change to the
    scoring/policy layers that consume its output.
    """

    embedder: EmbeddingBackend = field(default_factory=SentenceTransformerBackend)
    documents: List[Document] = field(default_factory=list)
    _doc_vecs: Optional[np.ndarray] = field(default=None, repr=False)

    def index(self, documents: Sequence[Document]) -> None:
        """Replace the corpus and (re-)embed it."""
        self.documents = list(documents)
        self._doc_vecs = self.embedder.embed([d.text for d in self.documents]) if self.documents else None

    def add(self, documents: Sequence[Document]) -> None:
        """Add documents to the index incrementally (only the new ones are embedded)."""
        if not documents:
            return
        new_vecs = self.embedder.embed([d.text for d in documents])
        self.documents.extend(documents)
        self._doc_vecs = new_vecs if self._doc_vecs is None else np.vstack([self._doc_vecs, new_vecs])

    def retrieve(self, query: str, k: int = 5) -> List[Evidence]:
        """Return the top-k documents as Evidence, ranked by cosine similarity.

        Cosine similarity lies in [-1, 1]; it's linearly remapped to [0, 1] before
        being used as `Evidence.relevance`, since that field's contract (see
        scoring.py) is a unit-interval retrieval-confidence weight.
        """
        if not self.documents or self._doc_vecs is None:
            return []
        q = self.embedder.embed([query])[0]
        norms = np.linalg.norm(self._doc_vecs, axis=1) * np.linalg.norm(q) + 1e-9
        cos = (self._doc_vecs @ q) / norms
        relevance = np.clip((cos + 1.0) / 2.0, 0.0, 1.0)
        order = np.argsort(-relevance)[:k]
        return [Evidence(text=self.documents[i].text, relevance=float(relevance[i])) for i in order]
