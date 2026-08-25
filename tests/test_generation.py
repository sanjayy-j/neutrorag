import json

from neutrorag.core import TIF
from neutrorag.generation import OllamaResponder, TemplateResponder
from neutrorag.policy import Action, Decision
from neutrorag.scoring import Evidence


def make_decision(action, tif, rationale="test rationale", expansions=0):
    return Decision(action=action, rationale=rationale, tif=tif, expansions_used=expansions)


def test_answer_response_states_support():
    r = TemplateResponder()
    d = make_decision(Action.ANSWER, TIF(0.8, 0.1, 0.1))
    text = r.respond("Sharks are fish.", [Evidence("Sharks are fish not mammals.", 0.9)], d)
    assert "Sharks are fish." in text
    assert "T=0.80" in text


def test_surface_contradiction_includes_all_retrieved_evidence():
    r = TemplateResponder()
    ev = [Evidence("Source A says yes.", 0.9), Evidence("Source B says no.", 0.9)]
    d = make_decision(Action.SURFACE_CONTRADICTION, TIF(0.5, 0.1, 0.6))
    text = r.respond("Claim.", ev, d)
    assert "Source A says yes." in text
    assert "Source B says no." in text
    assert "conflict" in text.lower()


def test_abstain_mentions_expansions_used_and_does_not_answer():
    r = TemplateResponder()
    d = make_decision(Action.ABSTAIN, TIF(0.1, 0.85, 0.05), expansions=2)
    text = r.respond("Claim.", [], d)
    assert "2 time(s)" in text
    assert "don't have enough" in text.lower()


def test_hedge_flags_as_provisional():
    r = TemplateResponder()
    d = make_decision(Action.HEDGE, TIF(0.3, 0.3, 0.2))
    text = r.respond("Claim.", [], d)
    assert "provisional" in text.lower()


def test_no_evidence_renders_explicitly_not_silently():
    r = TemplateResponder()
    d = make_decision(Action.SURFACE_CONTRADICTION, TIF(0.5, 0.1, 0.6))
    text = r.respond("Claim.", [], d)
    assert "none retrieved" in text.lower()


def test_ollama_responder_calls_real_api_shape_and_parses_response(monkeypatch):
    """No live server: mocks urlopen, but exercises the real request/response
    construction against Ollama's actual /api/generate contract."""
    captured = {}

    class FakeHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"response": "The evidence on this claim conflicts."}).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeHTTPResponse()

    monkeypatch.setattr("neutrorag.generation.urllib.request.urlopen", fake_urlopen)

    responder = OllamaResponder(model="llama3.2", host="http://localhost:11434")
    decision = make_decision(Action.SURFACE_CONTRADICTION, TIF(0.5, 0.1, 0.6))
    text = responder.respond("Claim.", [Evidence("Some evidence.", 0.9)], decision)

    assert text == "The evidence on this claim conflicts."
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["body"]["model"] == "llama3.2"
    assert captured["body"]["stream"] is False
    assert "Claim." in captured["body"]["prompt"]
