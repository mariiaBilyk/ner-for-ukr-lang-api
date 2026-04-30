"""
Unit tests for LLMClient.

LLMClient receives an InferenceBackend via constructor injection.
Tests verify: response forwarding, prompt passing, async execution
via asyncio.to_thread, error propagation, and metrics/ping delegation.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infrastructure.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

PROMPT = "Знайди сутності в тексті: Київ є столицею України."
EXPECTED_RESPONSE = '[{"label": "LOC", "text": "Київ"}, {"label": "LOC", "text": "України"}]'


def make_backend(response: str = EXPECTED_RESPONSE) -> MagicMock:
    backend = MagicMock()
    backend.chat.return_value = response
    backend.ping.return_value = True
    backend.metrics.return_value = {"method": "LLM", "provider": "ollama", "model": "test"}
    return backend


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------

class TestLLMClientGenerate:
    async def test_returns_backend_response(self):
        client = LLMClient(make_backend())
        assert await client.generate(PROMPT) == EXPECTED_RESPONSE

    async def test_passes_prompt_to_backend(self):
        backend = make_backend()
        await LLMClient(backend).generate(PROMPT)
        backend.chat.assert_called_once_with(PROMPT)

    async def test_generate_is_coroutine(self):
        client = LLMClient(make_backend())
        coro = client.generate(PROMPT)
        assert asyncio.iscoroutine(coro)
        await coro

    async def test_runs_in_thread_pool(self):
        """chat() is blocking — LLMClient must offload it via asyncio.to_thread."""
        backend = make_backend()
        with patch(
            "infrastructure.llm_client.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=EXPECTED_RESPONSE,
        ) as to_thread:
            client = LLMClient(backend)
            await client.generate(PROMPT)
            to_thread.assert_called_once_with(backend.chat, PROMPT)

    async def test_propagates_backend_error(self):
        backend = make_backend()
        backend.chat.side_effect = ConnectionError("Ollama unreachable")
        with pytest.raises(ConnectionError, match="Ollama unreachable"):
            await LLMClient(backend).generate(PROMPT)

    async def test_empty_prompt_forwarded(self):
        backend = make_backend(response="[]")
        result = await LLMClient(backend).generate("")
        backend.chat.assert_called_once_with("")
        assert result == "[]"


# ---------------------------------------------------------------------------
# ping()
# ---------------------------------------------------------------------------

class TestLLMClientPing:
    async def test_returns_true_when_backend_healthy(self):
        backend = make_backend()
        backend.ping.return_value = True
        assert await LLMClient(backend).ping() is True

    async def test_returns_false_when_backend_unreachable(self):
        backend = make_backend()
        backend.ping.return_value = False
        assert await LLMClient(backend).ping() is False


# ---------------------------------------------------------------------------
# metrics()
# ---------------------------------------------------------------------------

class TestLLMClientMetrics:
    def test_delegates_to_backend(self):
        backend = make_backend()
        result = LLMClient(backend).metrics()
        backend.metrics.assert_called_once()
        assert result == {"method": "LLM", "provider": "ollama", "model": "test"}
