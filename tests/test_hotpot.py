"""Offline tests for the HotpotQA loader.

Every test runs against the inline fixtures below -- no network, no model -- so
the parsing logic is covered independently of whether the dataset downloads.
Mirrors tests/test_fever.py and tests/test_conflicts.py.
"""
import pytest

from neutrorag.hotpot import (
    MIN_DISTRACTORS, Paragraph, eligible_for_unanswerable, examples_from_rows,
    exact_match, normalize_answer, split_answerable, stratified_sample,
    token_f1, type_counts,
)


def _row(qid, question, answer, qtype, titles, sentences, sf_titles, sf_ids,
         level="hard"):
    return {
        "id": qid, "question": question, "answer": answer,
        "type": qtype, "level": level,
        "context": {"title": titles, "sentences": sentences},
        "supporting_facts": {"title": sf_titles, "sent_id": sf_ids},
    }


#: Two gold paragraphs plus four distractors, i.e. eligible for the
#: unanswerable condition (4 >= MIN_DISTRACTORS).
def _eligible_row(qid, qtype="bridge", answer="Doctor Strange"):
    titles = ["Gold A", "Gold B", "Dist 1", "Dist 2", "Dist 3", "Dist 4"]
    sentences = [
        ["Gold A is directed by Scott Derrickson.", "It came out in 2012."],
        ["Gold B mentions Doctor Strange explicitly."],
        ["Distractor one is unrelated."],
        ["Distractor two is unrelated."],
        ["Distractor three is unrelated."],
        ["Distractor four is unrelated."],
    ]
    return _row(qid, f"Question {qid}?", answer, qtype, titles, sentences,
                ["Gold A", "Gold B"], [0, 0])


#: Only one distractor -> NOT eligible for the unanswerable condition.
_THIN_ROW = _row(
    "thin", "A question with almost no distractors?", "yes", "comparison",
    ["Gold A", "Gold B", "Dist 1"],
    [["Gold A sentence."], ["Gold B sentence."], ["Only distractor."]],
    ["Gold A", "Gold B"], [0, 0])

#: Supporting-fact title absent from its own context -> unusable gold.
_BAD_GOLD_ROW = _row(
    "nogold", "A question whose gold title is missing?", "nothing", "bridge",
    ["Some Paragraph"], [["A sentence."]],
    ["A Title Not In Context"], [0])

FIXTURE = [_eligible_row("q1", "comparison", "yes"),
           _eligible_row("q2", "bridge", "Doctor Strange"),
           _THIN_ROW, _BAD_GOLD_ROW]


def _many(n, bridge_share=0.5):
    """n synthetic eligible examples with distinct ids and a known type mix."""
    rows = []
    for i in range(n):
        qtype = "bridge" if i < int(n * bridge_share) else "comparison"
        rows.append(_eligible_row(f"id{i:04d}", qtype))
    return examples_from_rows(rows)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def test_parses_the_well_formed_rows():
    assert [e.qid for e in examples_from_rows(FIXTURE)] == ["q1", "q2", "thin"]


def test_row_whose_gold_title_is_absent_is_skipped():
    """A row with an empty gold set would score 0 recall for reasons that have
    nothing to do with the retriever, so it must not enter the sample."""
    assert "nogold" not in {e.qid for e in examples_from_rows(FIXTURE)}


def test_gold_is_exposed_at_both_granularities():
    q1 = examples_from_rows(FIXTURE)[0]
    assert q1.gold_titles == {"Gold A", "Gold B"}
    assert q1.gold_sentences == {("Gold A", 0), ("Gold B", 0)}


def test_paragraph_text_prepends_the_title():
    p = Paragraph("Ed Wood", ("Edward Davis Wood Jr. was a filmmaker.",))
    assert p.text == "Ed Wood. Edward Davis Wood Jr. was a filmmaker."


def test_max_examples_truncates():
    assert len(examples_from_rows(FIXTURE, max_examples=1)) == 1


def test_malformed_context_is_skipped_not_raised():
    bad = [{"id": "x", "question": "q", "answer": "a", "type": "bridge",
            "level": "hard",
            "context": {"title": ["T1", "T2"], "sentences": [["only one"]]},
            "supporting_facts": {"title": ["T1"], "sent_id": [0]}}]
    assert examples_from_rows(bad) == []


# ---------------------------------------------------------------------------
# The span-reader ceiling
# ---------------------------------------------------------------------------
def test_yes_no_is_flagged_and_unreachable_by_a_span_reader():
    q1, q2, _ = examples_from_rows(FIXTURE)
    assert q1.is_yes_no and not q1.answer_in_context()
    assert not q2.is_yes_no


def test_answer_in_context_detects_a_present_span():
    _, q2, _ = examples_from_rows(FIXTURE)
    assert q2.answer_in_context()


# ---------------------------------------------------------------------------
# Eligibility for the unanswerable condition
# ---------------------------------------------------------------------------
def test_thin_instance_is_not_eligible_for_the_unanswerable_condition():
    """21 real validation instances have no distractors at all; removing the
    gold there measures the empty-corpus path, not hallucination."""
    thin = [e for e in examples_from_rows(FIXTURE) if e.qid == "thin"][0]
    assert not eligible_for_unanswerable(thin)
    assert len(thin.without_gold().paragraphs) < MIN_DISTRACTORS


def test_normal_instance_is_eligible():
    q1 = examples_from_rows(FIXTURE)[0]
    assert eligible_for_unanswerable(q1)


def test_split_answerable_refuses_ineligible_input_loudly():
    with pytest.raises(ValueError, match="unanswerable condition"):
        split_answerable(examples_from_rows(FIXTURE), seed=1)


# ---------------------------------------------------------------------------
# The Task 3 construction
# ---------------------------------------------------------------------------
def test_without_gold_removes_exactly_the_gold_paragraphs():
    q1 = examples_from_rows(FIXTURE)[0]
    stripped = q1.without_gold()
    assert stripped.titles == ["Dist 1", "Dist 2", "Dist 3", "Dist 4"]
    assert stripped.gold_removed is True
    assert stripped.answerable is False


def test_without_gold_keeps_the_gold_set_for_metric_reporting():
    q1 = examples_from_rows(FIXTURE)[0]
    assert q1.without_gold().gold_titles == {"Gold A", "Gold B"}


def test_without_gold_does_not_mutate_the_original():
    q1 = examples_from_rows(FIXTURE)[0]
    before = list(q1.titles)
    q1.without_gold()
    assert q1.titles == before and q1.answerable is True


def test_split_answerable_is_a_partition_and_is_seeded():
    pool = _many(20)
    a1, u1 = split_answerable(pool, seed=7)
    a2, u2 = split_answerable(pool, seed=7)
    assert len(a1) + len(u1) == len(pool)
    assert [e.qid for e in a1] == [e.qid for e in a2]
    assert [e.qid for e in u1] == [e.qid for e in u2]
    assert all(e.answerable for e in a1)
    assert all(not e.answerable for e in u1)


def test_split_answerable_is_disjoint():
    a, u = split_answerable(_many(20), seed=7)
    assert set(e.qid for e in a).isdisjoint(e.qid for e in u)


def test_split_answerable_preserves_the_type_mix():
    a, u = split_answerable(_many(20), seed=3)
    assert type_counts(a) == type_counts(u)


def test_split_answerable_rounds_the_fraction_per_stratum():
    """The fraction is applied within each question type, not globally.

    With 10 bridge + 10 comparison at fraction 0.25 that is round(2.5) = 2 per
    type under Python's round-half-to-even, so 4 unanswerable rather than the 5
    a global split would give. Preserving the type mix exactly is worth the
    off-by-one; pinned here because the discrepancy is surprising.
    """
    pool = _many(20)
    a, u = split_answerable(pool, seed=3, fraction=0.25)
    assert len(a) + len(u) == len(pool)
    assert type_counts(u) == {"bridge": 2, "comparison": 2}
    assert type_counts(a) == {"bridge": 8, "comparison": 8}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def test_stratified_sample_is_seeded_and_sized():
    pool = _many(60)
    s1 = stratified_sample(pool, 10, seed=1)
    s2 = stratified_sample(pool, 10, seed=1)
    assert len(s1) == 10
    assert [e.qid for e in s1] == [e.qid for e in s2]


def test_stratified_sample_keeps_both_types():
    assert set(type_counts(stratified_sample(_many(60), 20, seed=2))) == {
        "bridge", "comparison"}


def test_stratified_sample_drops_ineligible_by_default():
    pool = examples_from_rows(FIXTURE)          # q1, q2 eligible; thin is not
    got = stratified_sample(pool, 3, seed=1)
    assert "thin" not in {e.qid for e in got}


def test_stratified_sample_can_keep_ineligible_when_asked():
    pool = examples_from_rows(FIXTURE)
    got = stratified_sample(pool, 3, seed=1,
                            require_unanswerable_eligible=False)
    assert "thin" in {e.qid for e in got}


# ---------------------------------------------------------------------------
# Answer scoring -- the official normalisation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("The Beatles", "beatles"),
    ("  a  Doctor   Strange!  ", "doctor strange"),
    ("Yes.", "yes"),
    # SQuAD normalisation strips articles as whole words, so the trailing "A"
    # of an initialism is removed. Pinned because it is surprising.
    ("U.S.A.", "u s"),
])
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


@pytest.mark.parametrize("pred,gold,ok", [
    ("Doctor Strange", "doctor strange", True),
    ("the Doctor Strange", "Doctor Strange", True),
    ("Doctor Strange 2", "Doctor Strange", False),
    ("", "Doctor Strange", False),
])
def test_exact_match(pred, gold, ok):
    assert exact_match(pred, gold) is ok


def test_token_f1_partial_credit():
    assert token_f1("Doctor Strange", "Doctor Strange") == pytest.approx(1.0)
    assert token_f1("Doctor Strange 2016", "Doctor Strange") == pytest.approx(0.8)
    assert token_f1("Sinister", "Doctor Strange") == 0.0


def test_token_f1_both_empty_is_one():
    """Needed for the unanswerable condition: abstaining where the gold answer
    is also empty is a correct match, not a vacuous one."""
    assert token_f1("", "") == 1.0
    assert token_f1("something", "") == 0.0
