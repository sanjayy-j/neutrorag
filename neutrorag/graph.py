"""In-memory knowledge-graph retrieval for NeutroRAG -- the 'GraphRAG' /
multi-hop retrieval stage of the architecture.

Scope, stated plainly: this is a real, working, in-memory graph (built on
`networkx`), not a Neo4j-backed production graph. That is a deliberate,
permanent design choice for a project at this scale, not a placeholder awaiting
an upgrade -- `KnowledgeGraph.add_fact` / `.retrieve` is the entire integration
surface the rest of NeutroRAG depends on, and a Neo4j-backed implementation of
that same surface could be substituted later without changing anything in
`scoring.py` or `policy.py`.

Design: each edge stores the natural-language sentence that asserted it, so a
retrieved graph *path* becomes `Evidence` text directly -- no separate document
corpus has to be kept in sync with the graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

from .scoring import Evidence


@dataclass
class Fact:
    """One (subject, relation, object) triple plus the sentence that asserts it."""

    subject: str
    relation: str
    obj: str
    text: str


class KnowledgeGraph:
    """A directed multigraph of entities connected by relation-labelled, textual edges."""

    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()

    def add_fact(self, subject: str, relation: str, obj: str, text: str) -> None:
        s, o = subject.strip().lower(), obj.strip().lower()
        self._g.add_node(s, label=subject)
        self._g.add_node(o, label=obj)
        self._g.add_edge(s, o, relation=relation, text=text)

    def add_facts(self, facts: Sequence[Fact]) -> None:
        for f in facts:
            self.add_fact(f.subject, f.relation, f.obj, f.text)

    def _match_entities(self, query: str) -> List[str]:
        """Entity linking: a node is 'mentioned' in the query if its label appears
        as a case-insensitive substring. Deliberately simple -- adequate for a
        small in-memory graph; a production system would use NER + an entity
        linker here instead, behind the same method signature."""
        ql = query.lower()
        return [n for n, data in self._g.nodes(data=True) if data.get("label", n).lower() in ql]

    def entities_in(self, text: str) -> List[str]:
        """Public wrapper over the entity linker, for callers that need to know
        *which* graph entities a claim mentions without retrieving evidence.

        Used by :mod:`neutrorag.sufficiency` to compute graph connectivity as a
        retrieval-side signal. Returns internal node keys (lowercased labels).
        """
        return self._match_entities(text)

    def hops_between(self, a: str, b: str, max_hops: int = 3) -> Optional[int]:
        """Shortest path length between two entities, or None if they are not
        connected within ``max_hops``.

        Edge direction is ignored: for a sufficiency signal what matters is
        whether the two entities are related in the graph at all, not which way
        the relation points (``Drug X --studied_in--> Trial A`` connects the two
        regardless of reading order).
        """
        if a not in self._g or b not in self._g:
            return None
        if a == b:
            return 0
        undirected = self._g.to_undirected(as_view=True)
        try:
            length = nx.shortest_path_length(undirected, a, b)
        except nx.NetworkXNoPath:
            return None
        return length if length <= max_hops else None

    def retrieve(self, query: str, hops: int = 2, decay: float = 0.75) -> List[Evidence]:
        """Multi-hop retrieval: start from entities mentioned in the query, walk
        outward up to `hops` edges (breadth-first), and return every traversed
        fact as Evidence. Relevance decays geometrically with hop distance from
        the nearest query entity -- a fact directly about a queried entity is
        more relevant than one reached by traversing through it.
        """
        seeds = self._match_entities(query)
        if not seeds:
            return []

        seen_edges: Set[Tuple[str, str, int]] = set()
        evidence: List[Evidence] = []
        frontier: Dict[str, int] = {s: 0 for s in seeds}
        visited: Set[str] = set(seeds)

        while frontier:
            next_frontier: Dict[str, int] = {}
            for node, dist in frontier.items():
                if dist >= hops:
                    continue
                for _, nbr, key, data in self._g.out_edges(node, keys=True, data=True):
                    edge_id = (node, nbr, key)
                    if edge_id in seen_edges:
                        continue
                    seen_edges.add(edge_id)
                    relevance = max(0.05, decay**dist)
                    evidence.append(Evidence(text=data["text"], relevance=relevance))
                    if nbr not in visited:
                        visited.add(nbr)
                        next_frontier[nbr] = dist + 1
            frontier = next_frontier
        return evidence
