from functools import lru_cache
from pathlib import Path

from infrastructure.inference.factory import InferenceFactory
from infrastructure.llm_client import LLMClient
from infrastructure.config import get_settings
from prompts.registry import PromptRegistry
from application.extraction_strategy import (
    ExtractionStrategy,
    SimpleExtractionStrategy,
    SelfCritiqueExtractionStrategy,
)
from application.ner_service import NerService


class _PromptRegistryFactory:
    _PROMPTS_DIR: Path = Path(__file__).parent.parent.parent / "prompts"

    @classmethod
    def build(cls) -> PromptRegistry:
        return PromptRegistry(cls._PROMPTS_DIR)


class _ExtractionStrategyFactory:
    @staticmethod
    def build(settings, llm_client, registry: PromptRegistry) -> ExtractionStrategy:
        if settings.extraction_strategy == "self_critique":
            return SelfCritiqueExtractionStrategy(llm_client, registry)
        return SimpleExtractionStrategy(llm_client)


def get_llm_client() -> LLMClient:
    return LLMClient(InferenceFactory.get_backend())


@lru_cache(maxsize=1)
def get_prompt_registry() -> PromptRegistry:
    return _PromptRegistryFactory.build()


@lru_cache(maxsize=1)
def get_ner_service() -> NerService:
    settings   = get_settings()
    llm_client = get_llm_client()
    registry   = get_prompt_registry()
    strategy   = _ExtractionStrategyFactory.build(settings, llm_client, registry)
    return NerService(
        llm_client,
        registry,
        prompt_name=settings.prompt_name,
        prompt_version=settings.prompt_version or None,
        strategy=strategy,
        max_attempts=settings.ner_agent_max_attempts,
    )
