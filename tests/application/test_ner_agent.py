"""
Unit tests for NERAgent.

NERAgent owns the structural-retry loop. It calls strategy.extract() and
retries when None is returned (structural failure). Specific feedback is
built from strategy.last_failure (a ParseResult).

Boundaries mocked:
- strategy    — MagicMock with AsyncMock extract() and last_failure attribute
- registry    — MagicMock, controls what feedback prompts are returned
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, call

from application.ner_agent import NERAgent
from utils.parse_entities import ParseResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_ENTITY  = {"label": "PERS", "text": "Петро"}
VALID_ENTITY2 = {"label": "LOC",  "text": "Київ"}

FAILURE_NO_JSON    = ParseResult(entities=[], has_json=False, bad_labels=set())
FAILURE_BAD_LABELS = ParseResult(entities=[], has_json=True, bad_labels={"PERSON"})

FEEDBACK_NO_JSON    = "Please return valid JSON."
FEEDBACK_BAD_LABELS = "Bad labels: {{BAD_LABELS}}. Use allowed labels."


def make_agent(extract_responses: list, last_failure=FAILURE_NO_JSON,
               max_attempts: int = 3) -> NERAgent:
    strategy = MagicMock()
    strategy.extract = AsyncMock(side_effect=extract_responses)
    strategy.last_failure = last_failure

    registry = MagicMock()
    registry.get.side_effect = lambda name, *_: (
        FEEDBACK_NO_JSON if name == NERAgent._FEEDBACK_NO_JSON_PROMPT
        else FEEDBACK_BAD_LABELS
    )

    return NERAgent(strategy=strategy, registry=registry, max_attempts=max_attempts)


# ---------------------------------------------------------------------------
# Happy path — strategy succeeds on first attempt
# ---------------------------------------------------------------------------

class TestSuccessOnFirstAttempt:
    async def test_returns_entities(self):
        agent = make_agent([[VALID_ENTITY]])
        assert await agent.run("prompt") == [VALID_ENTITY]

    async def test_strategy_called_once(self):
        agent = make_agent([[VALID_ENTITY]])
        await agent.run("prompt")
        agent._strategy.extract.assert_called_once_with("prompt")

    async def test_no_feedback_on_first_success(self):
        agent = make_agent([[VALID_ENTITY]])
        await agent.run("base")
        agent._strategy.extract.assert_called_once_with("base")

    async def test_multiple_entities_forwarded(self):
        agent = make_agent([[VALID_ENTITY, VALID_ENTITY2]])
        assert await agent.run("prompt") == [VALID_ENTITY, VALID_ENTITY2]

    async def test_empty_list_returned_without_retry(self):
        # Empty list is a valid (non-None) result — agent must not retry it
        agent = make_agent([[]])
        assert await agent.run("prompt") == []
        agent._strategy.extract.assert_called_once()


# ---------------------------------------------------------------------------
# Retry on structural failure — no JSON
# ---------------------------------------------------------------------------

class TestRetryOnNoJson:
    async def test_retries_and_succeeds(self):
        agent = make_agent([None, [VALID_ENTITY]], last_failure=FAILURE_NO_JSON)
        assert await agent.run("prompt") == [VALID_ENTITY]

    async def test_strategy_called_twice(self):
        agent = make_agent([None, [VALID_ENTITY]], last_failure=FAILURE_NO_JSON)
        await agent.run("prompt")
        assert agent._strategy.extract.call_count == 2

    async def test_no_json_feedback_appended(self):
        agent = make_agent([None, [VALID_ENTITY]], last_failure=FAILURE_NO_JSON)
        await agent.run("base")
        calls = agent._strategy.extract.call_args_list
        assert calls[1] == call("base\n\n" + FEEDBACK_NO_JSON)

    async def test_no_json_feedback_loaded_from_registry(self):
        agent = make_agent([None, [VALID_ENTITY]], last_failure=FAILURE_NO_JSON)
        await agent.run("prompt")
        agent._registry.get.assert_any_call(NERAgent._FEEDBACK_NO_JSON_PROMPT)


# ---------------------------------------------------------------------------
# Retry on structural failure — bad labels
# ---------------------------------------------------------------------------

class TestRetryOnBadLabels:
    async def test_retries_and_succeeds(self):
        agent = make_agent([None, [VALID_ENTITY]], last_failure=FAILURE_BAD_LABELS)
        assert await agent.run("prompt") == [VALID_ENTITY]

    async def test_bad_labels_substituted_in_feedback(self):
        agent = make_agent([None, [VALID_ENTITY]], last_failure=FAILURE_BAD_LABELS)
        await agent.run("base")
        second_prompt = agent._strategy.extract.call_args_list[1][0][0]
        assert "PERSON" in second_prompt

    async def test_bad_labels_feedback_loaded_from_registry(self):
        agent = make_agent([None, [VALID_ENTITY]], last_failure=FAILURE_BAD_LABELS)
        await agent.run("prompt")
        agent._registry.get.assert_any_call(NERAgent._FEEDBACK_BAD_LABELS_PROMPT)


# ---------------------------------------------------------------------------
# Exhausted retries
# ---------------------------------------------------------------------------

class TestExhaustedRetries:
    async def test_returns_empty_after_all_fail(self):
        agent = make_agent([None, None, None], max_attempts=3)
        assert await agent.run("prompt") == []

    async def test_called_exactly_max_attempts(self):
        agent = make_agent([None] * 5, max_attempts=3)
        await agent.run("prompt")
        assert agent._strategy.extract.call_count == 3

    async def test_max_attempts_one_no_retry(self):
        agent = make_agent([None], max_attempts=1)
        assert await agent.run("prompt") == []
        agent._strategy.extract.assert_called_once()
