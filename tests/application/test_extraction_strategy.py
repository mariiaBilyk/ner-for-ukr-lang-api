"""
Unit tests for ExtractionStrategy implementations.

The two loops are tested in isolation:
- SimpleExtractionStrategy: single LLM call, returns None on structural failure
- SelfCritiqueExtractionStrategy:
    * _first_pass fails  → returns None  (agent retries)
    * _first_pass ok, _critique fails  → returns first_pass  (agent does NOT retry)
    * _first_pass ok, _critique ok     → returns critique result

Boundaries mocked:
- llm_client  — AsyncMock controlling raw LLM responses
- registry    — MagicMock controlling the critique prompt template
parse_entities_full() is NOT mocked.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from application.extraction_strategy import (
    SimpleExtractionStrategy,
    SelfCritiqueExtractionStrategy,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

VALID_ENTITY  = {"label": "PERS", "text": "Петро"}
VALID_ENTITY2 = {"label": "LOC",  "text": "Київ"}

VALID_JSON    = json.dumps([VALID_ENTITY])
VALID_JSON2   = json.dumps([VALID_ENTITY, VALID_ENTITY2])
INVALID_JSON  = "I found some people in the text."
BAD_LABELS    = json.dumps([{"label": "PERSON", "text": "Петро"}])

CRITIQUE_PROMPT_TEMPLATE = "Review: {{ENTITIES_JSON}}"


def make_simple(responses: list[str]) -> SimpleExtractionStrategy:
    llm_client = AsyncMock()
    llm_client.generate = AsyncMock(side_effect=responses)
    return SimpleExtractionStrategy(llm_client)


def make_self_critique(responses: list[str]) -> SelfCritiqueExtractionStrategy:
    llm_client = AsyncMock()
    llm_client.generate = AsyncMock(side_effect=responses)

    registry = MagicMock()
    registry.get.return_value = CRITIQUE_PROMPT_TEMPLATE

    return SelfCritiqueExtractionStrategy(llm_client, registry)


# ---------------------------------------------------------------------------
# SimpleExtractionStrategy
# ---------------------------------------------------------------------------

class TestSimpleExtractionStrategy:
    async def test_returns_entities_on_valid_json(self):
        s = make_simple([VALID_JSON])
        assert await s.extract("prompt") == [VALID_ENTITY]

    async def test_returns_none_on_invalid_json(self):
        s = make_simple([INVALID_JSON])
        assert await s.extract("prompt") is None

    async def test_returns_none_on_bad_labels(self):
        s = make_simple([BAD_LABELS])
        assert await s.extract("prompt") is None

    async def test_llm_called_once(self):
        s = make_simple([VALID_JSON])
        await s.extract("prompt")
        s._llm_client.generate.assert_called_once_with("prompt")

    async def test_last_failure_set_on_no_json(self):
        s = make_simple([INVALID_JSON])
        await s.extract("prompt")
        assert s.last_failure is not None
        assert not s.last_failure.has_json

    async def test_last_failure_set_on_bad_labels(self):
        s = make_simple([BAD_LABELS])
        await s.extract("prompt")
        assert s.last_failure is not None
        assert "PERSON" in s.last_failure.bad_labels

    async def test_last_failure_cleared_on_success(self):
        s = make_simple([INVALID_JSON, VALID_JSON])
        await s.extract("first")
        await s.extract("second")
        assert s.last_failure is None

    async def test_multiple_entities(self):
        s = make_simple([VALID_JSON2])
        assert await s.extract("prompt") == [VALID_ENTITY, VALID_ENTITY2]


# ---------------------------------------------------------------------------
# SelfCritiqueExtractionStrategy — first pass fails → returns None
# ---------------------------------------------------------------------------

class TestSelfCritiqueFirstPassFails:
    async def test_returns_none_on_invalid_json(self):
        s = make_self_critique([INVALID_JSON])
        assert await s.extract("prompt") is None

    async def test_returns_none_on_bad_labels(self):
        s = make_self_critique([BAD_LABELS])
        assert await s.extract("prompt") is None

    async def test_only_one_llm_call_on_first_pass_failure(self):
        # Critique must not be called if first pass fails
        s = make_self_critique([INVALID_JSON])
        await s.extract("prompt")
        assert s._llm_client.generate.call_count == 1

    async def test_last_failure_set_on_first_pass_failure(self):
        s = make_self_critique([INVALID_JSON])
        await s.extract("prompt")
        assert s.last_failure is not None


# ---------------------------------------------------------------------------
# SelfCritiqueExtractionStrategy — first pass ok, critique ok
# ---------------------------------------------------------------------------

class TestSelfCritiqueSuccess:
    async def test_returns_critique_result(self):
        s = make_self_critique([VALID_JSON, VALID_JSON2])
        result = await s.extract("prompt")
        assert result == [VALID_ENTITY, VALID_ENTITY2]

    async def test_two_llm_calls_made(self):
        s = make_self_critique([VALID_JSON, VALID_JSON2])
        await s.extract("prompt")
        assert s._llm_client.generate.call_count == 2

    async def test_critique_prompt_includes_first_pass_entities(self):
        s = make_self_critique([VALID_JSON, VALID_JSON2])
        await s.extract("base prompt")
        critique_call_prompt = s._llm_client.generate.call_args_list[1][0][0]
        assert "Петро" in critique_call_prompt
        assert "PERS" in critique_call_prompt

    async def test_critique_prompt_appended_to_original(self):
        s = make_self_critique([VALID_JSON, VALID_JSON2])
        await s.extract("base prompt")
        critique_call_prompt = s._llm_client.generate.call_args_list[1][0][0]
        assert critique_call_prompt.startswith("base prompt")

    async def test_last_failure_none_after_success(self):
        s = make_self_critique([VALID_JSON, VALID_JSON2])
        await s.extract("prompt")
        assert s.last_failure is None


# ---------------------------------------------------------------------------
# SelfCritiqueExtractionStrategy — first pass ok, critique fails → fallback
# ---------------------------------------------------------------------------

class TestSelfCritiqueFallback:
    async def test_returns_first_pass_when_critique_fails(self):
        # first pass succeeds, critique returns unparseable text
        s = make_self_critique([VALID_JSON, INVALID_JSON])
        result = await s.extract("prompt")
        assert result == [VALID_ENTITY]

    async def test_returns_first_pass_when_critique_has_bad_labels(self):
        s = make_self_critique([VALID_JSON, BAD_LABELS])
        result = await s.extract("prompt")
        assert result == [VALID_ENTITY]

    async def test_does_not_return_none_on_critique_failure(self):
        # Agent must NOT retry — result is structurally valid (first pass)
        s = make_self_critique([VALID_JSON, INVALID_JSON])
        result = await s.extract("prompt")
        assert result is not None

    async def test_two_llm_calls_on_critique_failure(self):
        s = make_self_critique([VALID_JSON, INVALID_JSON])
        await s.extract("prompt")
        assert s._llm_client.generate.call_count == 2
