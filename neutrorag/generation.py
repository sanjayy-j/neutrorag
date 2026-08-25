"""Turn a NeutroRAG Decision into a final, user-facing response -- the 'LLM
Response' stage of the architecture.

Nothing upstream of this module produces prose: `scoring.py` decides *what the
evidence supports*, `policy.py` decides *what to do about it*. This module
decides *what to say*, and says something different for each Action -- that's
the point of the five-way policy: ANSWER, SURFACE_CONTRADICTION, HEDGE and
ABSTAIN are not interchangeable "here's an answer" responses, so the response
generator has to respect that distinction rather than flattening it.

Two responders are provided, and both are real -- neither is a stand-in for a
"proper" implementation:

- `TemplateResponder` (default): deterministic, policy-driven natural-language
  templates, stdlib only. This is a legitimate production choice for a
  trust-sensitive system, not a mock -- it guarantees the caveat/conflict
  structure the policy decided on cannot be silently dropped or reworded by a
  generative model.
- `OllamaResponder`: a real call to a locally running Ollama server
  (https://ollama.com) via its actual `/api/generate` HTTP endpoint, using only
  `urllib` from the standard library (no new dependency). It is prompted with
  the claim, the evidence, and the policy's decision, and explicitly instructed
  to respect the decision rather than freelance past it. Lazy: it never touches
  the network until `respond()` is called.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Protocol, Sequence

from .policy import Action, Decision
from .scoring import Evidence


class Responder(Protocol):
    def respond(self, claim: str, evidence: Sequence[Evidence], decision: Decision) -> str: ...


_TEMPLATES = {
    Action.ANSWER: (
        "{claim}\n\n"
        "This is supported by the retrieved evidence (T={t:.2f}, I={i:.2f}, F={f:.2f}). "
        "{rationale}."
    ),
    Action.SURFACE_CONTRADICTION: (
        "The evidence on \"{claim}\" conflicts and should not be resolved by picking a side "
        "(T={t:.2f}, I={i:.2f}, F={f:.2f}; {rationale}).\n\n"
        "Retrieved evidence:\n{evidence_list}"
    ),
    Action.HEDGE: (
        "{claim}\n\n"
        "The evidence is weak or mixed (T={t:.2f}, I={i:.2f}, F={f:.2f}; {rationale}). "
        "Treat this as provisional, not confirmed."
    ),
    Action.ABSTAIN: (
        "I don't have enough reliable evidence to answer \"{claim}\" "
        "(T={t:.2f}, I={i:.2f}, F={f:.2f}; {rationale}), even after expanding retrieval "
        "{expansions} time(s). I'm abstaining rather than guessing."
    ),
    Action.EXPAND: (
        "Still gathering evidence for \"{claim}\" (I={i:.2f}); no answer yet."
    ),
}


@dataclass
class TemplateResponder:
    """Deterministic, policy-driven response generation. See module docstring.

    Note on SURFACE_CONTRADICTION: the scorer aggregates all evidence into one
    (T, I, F) triple (scoring.py) rather than labelling which individual source
    supports vs. contradicts, so the response can point at the retrieved
    evidence set as a whole, not attribute specific sentences to specific sides.
    """

    def respond(self, claim: str, evidence: Sequence[Evidence], decision: Decision) -> str:
        tif = decision.tif
        evidence_list = "\n".join(f"  - {e.text}" for e in evidence) or "  (none retrieved)"
        template = _TEMPLATES[decision.action]
        return template.format(
            claim=claim,
            t=tif.T, i=tif.I, f=tif.F,
            rationale=decision.rationale,
            expansions=decision.expansions_used,
            evidence_list=evidence_list,
        )


@dataclass
class OllamaResponder:
    """Real generation via a locally running Ollama server's /api/generate endpoint.

    Requires `ollama serve` running locally with `model` pulled (e.g.
    `ollama pull llama3.2`); nothing else. No new dependency -- built on
    `urllib.request` from the standard library.
    """

    model: str = "llama3.2"
    host: str = "http://localhost:11434"
    timeout: float = 60.0

    def _prompt(self, claim: str, evidence: Sequence[Evidence], decision: Decision) -> str:
        tif = decision.tif
        ev_block = "\n".join(
            f"- {e.text} (relevance={e.relevance:.2f})" for e in evidence
        ) or "(no evidence retrieved)"
        return (
            "You are the response-generation stage of a fact-checking pipeline. "
            "A neutrosophic trust layer has already scored the evidence and decided "
            f"on the action '{decision.action.value}' for the claim below. Write the "
            "final response to the user. Follow the decision -- do not override it, "
            "and do not claim more certainty than the action implies.\n\n"
            f"Claim: {claim}\n"
            f"Evidence:\n{ev_block}\n\n"
            f"Trust scores: T(support)={tif.T:.2f}, I(indeterminacy)={tif.I:.2f}, "
            f"F(contradiction)={tif.F:.2f}\n"
            f"Decision: {decision.action.value} -- {decision.rationale}\n\n"
            "Response:"
        )

    def respond(self, claim: str, evidence: Sequence[Evidence], decision: Decision) -> str:
        payload = json.dumps({
            "model": self.model,
            "prompt": self._prompt(claim, evidence, decision),
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["response"].strip()
