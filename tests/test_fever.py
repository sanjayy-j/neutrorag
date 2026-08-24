from neutrorag.fever import LABEL_TO_CHANNEL, _examples_from_rows

# Tiny inline fixture shaped like a `copenlu/fever_gold_evidence` row:
# claim, label, evidence=[[wikipedia_page, sentence_id, sentence_text], ...].
# No network call and no dataset download -- this only exercises row parsing.
_ROWS = [
    {
        "claim": "The sky is blue.",
        "label": "SUPPORTS",
        "evidence": [["Sky", "0", "The sky appears blue due to Rayleigh scattering."]],
    },
    {
        "claim": "Cats are reptiles.",
        "label": "REFUTES",
        "evidence": [["Cat", "1", "Cats are mammals, not reptiles."]],
    },
    {
        "claim": "Someone did something noteworthy.",
        "label": "NOT ENOUGH INFO",
        "evidence": [],
    },
]


def test_label_to_channel_mapping():
    assert LABEL_TO_CHANNEL == {"SUPPORTS": "T", "NOT ENOUGH INFO": "I", "REFUTES": "F"}


def test_examples_from_rows_builds_evidence_and_gold_channel():
    examples = _examples_from_rows(_ROWS)
    assert len(examples) == 3

    supports, refutes, nei = examples
    assert supports.gold_channel == "T"
    assert len(supports.evidence) == 1
    assert supports.evidence[0].text == "The sky appears blue due to Rayleigh scattering."
    assert supports.evidence[0].relevance == 0.9

    assert refutes.gold_channel == "F"
    assert refutes.evidence[0].text == "Cats are mammals, not reptiles."

    assert nei.gold_channel == "I"
    assert nei.evidence == []


def test_examples_from_rows_skips_unknown_labels_and_respects_limit():
    rows = [{"claim": f"claim {i}", "label": "SUPPORTS", "evidence": []} for i in range(5)]
    rows.insert(2, {"claim": "disputed claim", "label": "DISPUTED", "evidence": []})

    examples = _examples_from_rows(rows, max_examples=3)

    assert len(examples) == 3
    assert all(e.gold_label == "SUPPORTS" for e in examples)


def test_examples_from_rows_is_robust_to_malformed_or_missing_evidence():
    rows = [
        {"claim": "no evidence key at all", "label": "NOT ENOUGH INFO"},
        {"claim": "malformed evidence entries", "label": "SUPPORTS",
         "evidence": [["OnlyPage"], None, ["Page", "0", ""], ["Page", "1", "Usable sentence."]]},
    ]

    examples = _examples_from_rows(rows)

    assert examples[0].evidence == []
    assert [e.text for e in examples[1].evidence] == ["Usable sentence."]
