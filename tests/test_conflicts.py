"""Tests for the CONFLICTS loader -- inline fixture, no file, no network, no model.

Mirrors tests/test_fever.py. The relevance function is injected, so nothing here
touches sentence-transformers.
"""
import pytest

from neutrorag.conflicts import (
    CONFLICT_TYPES,
    TOO_FEW_TO_EVALUATE,
    TYPE_TO_EXPECTED_ACTION,
    _examples_from_rows,
    passage_text,
    result_domain,
)

# Shaped exactly like the released conflicts.jsonl (schema verified against the
# repo's README and the real file, not assumed).
_ROWS = [
    {
        "source": "conflicting_qa",
        "question": "Does drug X reduce risk?",
        "conflict_type": "Conflicting opinions and research outcomes",
        "correct_answer": "",
        "search_results": [
            {"title": "Trial A", "url": "https://www.nejm.org/a",
             "snippet": "short snip", "date": "2024-01-01",
             "response_str": "full page text", "short_text": "Drug X reduced risk by 30%."},
            {"title": "Meta M", "url": "https://bmj.com/b",
             "snippet": "another snip", "date": "2025-02-02",
             "response_str": "full page", "short_text": "No significant reduction was found."},
        ],
    },
    {
        "source": "freshqa",
        "question": "Who is the head of state?",
        "conflict_type": "No conflict",
        "correct_answer": "Someone",
        "search_results": [
            {"title": "Gov", "url": "https://en.wikipedia.org/wiki/X",
             "snippet": "snip", "date": "2025-03-03",
             "response_str": "page", "short_text": "The head of state is Someone."},
        ],
    },
    {
        "source": "qacc",
        "question": "Skipped, unknown type.",
        "conflict_type": "Not a real released type",
        "search_results": [{"title": "t", "url": "https://x.com/1", "short_text": "text"}],
    },
]


def _fake_relevance(question, passages):
    """Deterministic stand-in: descending by position, no model involved."""
    return [1.0 - 0.1 * i for i in range(len(passages))]


def test_released_type_counts_are_pinned():
    assert CONFLICT_TYPES == {
        "No conflict": 161,
        "Complementary information": 115,
        "Conflicting opinions and research outcomes": 115,
        "Conflict due to outdated information": 62,
        "Conflict due to misinformation": 5,
    }
    assert sum(CONFLICT_TYPES.values()) == 458


def test_only_two_types_map_to_an_action():
    assert TYPE_TO_EXPECTED_ACTION == {
        "Conflicting opinions and research outcomes": "surface_contradiction",
        "No conflict": "answer",
    }
    # the unmapped three must not acquire a mapping by accident
    for t in ("Complementary information", "Conflict due to outdated information",
              "Conflict due to misinformation"):
        assert t not in TYPE_TO_EXPECTED_ACTION


def test_misinformation_is_marked_too_small_to_evaluate():
    assert TOO_FEW_TO_EVALUATE == {"Conflict due to misinformation"}
    assert CONFLICT_TYPES["Conflict due to misinformation"] == 5


def test_examples_are_built_with_evidence_relevance_and_source():
    examples = _examples_from_rows(_ROWS, _fake_relevance)
    assert len(examples) == 2  # the unknown-type row is skipped, not relabelled

    conflicting, no_conflict = examples
    assert conflicting.conflict_type == "Conflicting opinions and research outcomes"
    assert conflicting.expected_action == "surface_contradiction"
    assert [e.text for e in conflicting.evidence] == [
        "Drug X reduced risk by 30%.", "No significant reduction was found."]
    assert [e.relevance for e in conflicting.evidence] == [1.0, 0.9]
    assert [e.source for e in conflicting.evidence] == ["nejm.org", "bmj.com"]

    assert no_conflict.expected_action == "answer"
    assert no_conflict.source == "freshqa"
    assert no_conflict.correct_answer == "Someone"


def test_passage_text_prefers_short_text_then_snippet_then_title():
    assert passage_text({"short_text": "A", "snippet": "B", "title": "C"}) == "A"
    assert passage_text({"short_text": "  ", "snippet": "B", "title": "C"}) == "B"
    assert passage_text({"snippet": "", "title": "C"}) == "C"
    assert passage_text({"url": "https://x.com"}) == ""


def test_results_without_any_text_are_dropped():
    rows = [{
        "question": "q", "conflict_type": "No conflict",
        "search_results": [
            {"url": "https://a.com", "short_text": "kept"},
            {"url": "https://b.com"},                      # no text at all
            {"url": "https://c.com", "short_text": "   "},  # whitespace only
        ],
    }]
    (example,) = _examples_from_rows(rows, _fake_relevance)
    assert [e.text for e in example.evidence] == ["kept"]


def test_domain_strips_www_and_handles_missing_urls():
    assert result_domain({"url": "https://www.nature.com/articles/x"}) == "nature.com"
    assert result_domain({"url": "http://BMJ.com/b"}) == "bmj.com"
    assert result_domain({}) == ""


def test_max_examples_limits_the_result():
    rows = [dict(_ROWS[1], question=f"q{i}") for i in range(5)]
    assert len(_examples_from_rows(rows, _fake_relevance, max_examples=3)) == 3


def test_relevance_is_not_a_constant_by_construction():
    """The F1 defect was a hardcoded constant relevance. Guard against its return."""
    examples = _examples_from_rows(_ROWS, _fake_relevance)
    values = [e.relevance for ex in examples for e in ex.evidence]
    assert len(set(values)) > 1, "relevance must vary, not be a fabricated constant"


def test_unmapped_types_expose_no_expected_action():
    rows = [dict(_ROWS[0], conflict_type="Complementary information")]
    (example,) = _examples_from_rows(rows, _fake_relevance)
    assert example.expected_action is None
