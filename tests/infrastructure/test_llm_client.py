"""
Unit tests for LLMClient.

Strategy:
- Mock OllamaInference at the boundary where LLMClient imports it.
- Never touch the real Ollama server.
- Tests cover: normal path, async execution, error propagation,
  and that the singleton is fetched (not re-created) on construction.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

PROMPT = "Знайди сутності в тексті: Київ є столицею України."
EXPECTED_RESPONSE = '[{"label": "LOC", "text": "Київ"}, {"label": "LOC", "text": "України"}]'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_inference():
    """A fully-configured mock of OllamaInference returned by get_instance()."""
    inference = MagicMock()
    inference.chat.return_value = EXPECTED_RESPONSE
    return inference


@pytest.fixture()
def llm_client(mock_inference):
    """LLMClient with its singleton dependency replaced by mock_inference."""
    with patch(
        "infrastructure.llm_client.OllamaInference.get_instance",
        return_value=mock_inference,
    ):
        from infrastructure.llm_client import LLMClient
        yield LLMClient()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLLMClientConstruction:
    def test_fetches_singleton_on_init(self, mock_inference):
        """LLMClient must call get_instance() exactly once during construction."""
        with patch(
            "infrastructure.llm_client.OllamaInference.get_instance",
            return_value=mock_inference,
        ) as get_instance:
            from infrastructure.llm_client import LLMClient
            LLMClient()
            get_instance.assert_called_once()

    def test_raises_if_singleton_not_initialized(self):
        """Construction must propagate RuntimeError when singleton is missing."""
        with patch(
            "infrastructure.llm_client.OllamaInference.get_instance",
            side_effect=RuntimeError("not initialized"),
        ):
            from infrastructure.llm_client import LLMClient
            with pytest.raises(RuntimeError, match="not initialized"):
                LLMClient()


class TestLLMClientGenerate:
    async def test_returns_inference_response(self, llm_client, mock_inference):
        result = await llm_client.generate(PROMPT)
        assert result == EXPECTED_RESPONSE

    async def test_passes_prompt_to_inference(self, llm_client, mock_inference):
        await llm_client.generate(PROMPT)
        mock_inference.chat.assert_called_once_with(PROMPT)

    async def test_generate_is_coroutine(self, llm_client):
        """generate() must be awaitable (not accidentally sync)."""
        coro = llm_client.generate(PROMPT)
        assert asyncio.iscoroutine(coro)
        await coro

    async def test_runs_in_thread_pool(self, mock_inference):
        """
        OllamaInference.chat() is blocking; LLMClient must offload it to a
        thread so the event loop stays unblocked. Verify via asyncio.to_thread.
        """
        with patch(
            "infrastructure.llm_client.OllamaInference.get_instance",
            return_value=mock_inference,
        ), patch(
            "infrastructure.llm_client.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=EXPECTED_RESPONSE,
        ) as to_thread:
            from infrastructure.llm_client import LLMClient
            client = LLMClient()
            await client.generate(PROMPT)

            to_thread.assert_called_once_with(mock_inference.chat, PROMPT)

    async def test_propagates_inference_error(self, llm_client, mock_inference):
        """Errors from the inference layer must bubble up to the caller."""
        mock_inference.chat.side_effect = ConnectionError("Ollama unreachable")

        with pytest.raises(ConnectionError, match="Ollama unreachable"):
            await llm_client.generate(PROMPT)

    async def test_empty_prompt_is_forwarded(self, llm_client, mock_inference):
        """LLMClient must not filter or validate the prompt — pass it as-is."""
        mock_inference.chat.return_value = "[]"
        result = await llm_client.generate("")
        mock_inference.chat.assert_called_once_with("")
        assert result == "[]"
