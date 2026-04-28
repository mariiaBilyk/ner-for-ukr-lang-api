import json
from typing import Protocol, runtime_checkable

import structlog

from prompts.registry import PromptRegistry
from utils.parse_entities import ParseResult, parse_entities_full

logger = structlog.get_logger()


@runtime_checkable
class ExtractionStrategy(Protocol):
    @property
    def last_failure(self) -> ParseResult | None: ...

    async def extract(self, prompt: str) -> list[dict] | None: ...


class SimpleExtractionStrategy:
    def __init__(self, llm_client) -> None:
        self._llm_client   = llm_client
        self._last_failure: ParseResult | None = None

    @property
    def last_failure(self) -> ParseResult | None:
        return self._last_failure

    async def extract(self, prompt: str) -> list[dict] | None:
        raw    = await self._llm_client.generate(prompt)
        result = parse_entities_full(raw)
        if not result.has_json or result.bad_labels:
            self._last_failure = result
            return None
        self._last_failure = None
        return result.entities


class SelfCritiqueExtractionStrategy:
    _CRITIQUE_PROMPT      = "critique-review"
    _ENTITIES_PLACEHOLDER = "{{ENTITIES_JSON}}"

    def __init__(self, llm_client, registry: PromptRegistry) -> None:
        self._llm_client   = llm_client
        self._registry     = registry
        self._last_failure: ParseResult | None = None

    @property
    def last_failure(self) -> ParseResult | None:
        return self._last_failure

    async def extract(self, prompt: str) -> list[dict] | None:
        first_pass = await self._first_pass(prompt)
        if first_pass is None:
            return None  # structural failure — agent will retry

        critique = await self._critique(first_pass, prompt)
        if critique is None:
            logger.warning("self_critique_failed", fallback="first_pass")
            return first_pass  # structurally valid — agent must not retry

        return critique

    async def _first_pass(self, prompt: str) -> list[dict] | None:
        raw    = await self._llm_client.generate(prompt)
        result = parse_entities_full(raw)
        if not result.has_json or result.bad_labels:
            self._last_failure = result
            return None
        self._last_failure = None
        return result.entities

    async def _critique(self, entities: list[dict], prompt: str) -> list[dict] | None:
        critique_prompt = self._build_critique_prompt(entities, prompt)
        raw    = await self._llm_client.generate(critique_prompt)
        result = parse_entities_full(raw)
        if not result.has_json or result.bad_labels:
            return None
        return result.entities

    def _build_critique_prompt(self, entities: list[dict], prompt: str) -> str:
        tmpl = self._registry.get(self._CRITIQUE_PROMPT)
        return prompt + "\n\n" + tmpl.replace(
            self._ENTITIES_PLACEHOLDER,
            json.dumps(entities, ensure_ascii=False),
        )
