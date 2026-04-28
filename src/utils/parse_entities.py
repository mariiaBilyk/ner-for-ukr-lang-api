import json
import re
from dataclasses import dataclass, field

from models.ner_label import NerLabel

_VALID_LABELS = set(NerLabel)


@dataclass
class ParseResult:
    entities: list[dict] = field(default_factory=list)
    has_json: bool = False          # a JSON array was found and parsed
    bad_labels: set[str] = field(default_factory=set)  # labels present but not in NerLabel


def parse_entities_full(raw: str) -> ParseResult:
    """Parse raw LLM output and return full diagnostics.

    Use this when the caller needs to know *why* parsing failed
    (e.g. for retry feedback in NERAgent).
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if not match:
        return ParseResult()

    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return ParseResult()

    valid, bad_labels = [], set()
    for e in parsed:
        if not isinstance(e, dict) or "text" not in e or "label" not in e:
            continue
        if e["label"] in _VALID_LABELS:
            valid.append({"text": e["text"], "label": e["label"]})
        else:
            bad_labels.add(e["label"])

    return ParseResult(entities=valid, has_json=True, bad_labels=bad_labels)


def parse_entities(raw: str) -> list[dict]:
    """Extract a JSON entity array from raw LLM output.

    Handles markdown code fences and surrounding text.
    Strips any entity whose label is not in the allowed 13 (NerLabel).
    Returns a list of {"label": ..., "text": ...} dicts.
    """
    return parse_entities_full(raw).entities
