"""
Unit tests for parse_entities — focused on label validation.
"""

import json
import pytest
from models.ner_label import NerLabel
from utils.parse_entities import parse_entities


def make_raw(*entities: dict) -> str:
    return json.dumps(entities)


# ---------------------------------------------------------------------------
# Valid labels
# ---------------------------------------------------------------------------

class TestValidLabels:
    def test_all_13_labels_are_accepted(self):
        for label in NerLabel:
            raw = make_raw({"label": label, "text": "sample"})
            result = parse_entities(raw)
            assert len(result) == 1
            assert result[0]["label"] == label

    def test_accepted_entity_preserves_text(self):
        raw = make_raw({"label": "PERS", "text": "Петро Порошенко"})
        result = parse_entities(raw)
        assert result[0]["text"] == "Петро Порошенко"


# ---------------------------------------------------------------------------
# Invalid labels are stripped
# ---------------------------------------------------------------------------

class TestInvalidLabels:
    @pytest.mark.parametrize("bad_label", [
        "PERSON", "LOCATION", "ORGANIZATION",   # common LLM hallucinations
        "person", "loc", "Pers",                 # wrong case
        "", "UNKNOWN", "FAKE", "123",
    ])
    def test_invalid_label_is_stripped(self, bad_label: str):
        raw = make_raw({"label": bad_label, "text": "sample"})
        assert parse_entities(raw) == []

    def test_mixed_valid_and_invalid_keeps_only_valid(self):
        raw = make_raw(
            {"label": "PERS", "text": "Петро"},
            {"label": "PERSON", "text": "should be stripped"},
            {"label": "LOC",  "text": "Київ"},
        )
        result = parse_entities(raw)
        assert len(result) == 2
        assert all(r["label"] in set(NerLabel) for r in result)

    def test_all_invalid_returns_empty_list(self):
        raw = make_raw(
            {"label": "FAKE", "text": "a"},
            {"label": "BAD",  "text": "b"},
        )
        assert parse_entities(raw) == []
