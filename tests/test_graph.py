from neutrorag.graph import Fact, KnowledgeGraph


def make_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_facts([
        Fact("Aspirin", "studied_in", "Trial A", "Aspirin was studied in Trial A."),
        Fact("Trial A", "enrolled", "High-risk patients", "Trial A enrolled high-risk patients."),
        Fact("Trial A", "found", "30 percent stroke reduction",
             "Trial A found a 30 percent reduction in stroke risk."),
        Fact("Ibuprofen", "studied_in", "Trial B", "Ibuprofen was studied in Trial B."),
    ])
    return g


def test_one_hop_retrieval_finds_direct_facts_only():
    g = make_graph()
    texts = [e.text for e in g.retrieve("Aspirin", hops=1)]
    assert "Aspirin was studied in Trial A." in texts
    assert "Trial A enrolled high-risk patients." not in texts  # that's 2 hops away
    assert "Ibuprofen was studied in Trial B." not in texts     # unrelated entity


def test_multi_hop_retrieval_follows_paths():
    g = make_graph()
    texts = [e.text for e in g.retrieve("Aspirin", hops=2)]
    assert "Aspirin was studied in Trial A." in texts
    assert "Trial A enrolled high-risk patients." in texts
    assert "Trial A found a 30 percent reduction in stroke risk." in texts
    assert "Ibuprofen was studied in Trial B." not in texts


def test_relevance_decays_with_hop_distance():
    g = make_graph()
    by_text = {e.text: e.relevance for e in g.retrieve("Aspirin", hops=2)}
    assert by_text["Aspirin was studied in Trial A."] > by_text["Trial A enrolled high-risk patients."]


def test_no_matching_entity_returns_no_evidence():
    g = make_graph()
    assert g.retrieve("Completely unrelated query") == []
