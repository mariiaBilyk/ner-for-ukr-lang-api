from application.extraction_strategy import ExtractionStrategy
from application.ner_agent import NERAgent
from models.health_status import HealthStatus
from prompts.registry import PromptRegistry


class NerService:
    _TEXT_PLACEHOLDER = "{{TEXT_HERE}}"

    def __init__(self, llm_client, registry: PromptRegistry,
                 prompt_name: str, prompt_version: str | None = None,
                 strategy: ExtractionStrategy | None = None,
                 max_attempts: int = 3) -> None:
        self.llm_client = llm_client
        self._prompt_template = registry.get(prompt_name, prompt_version)
        self._agent = NERAgent(strategy, registry, max_attempts)

    def _build_prompt(self, text: str) -> str:
        return self._prompt_template.replace(self._TEXT_PLACEHOLDER, text)

    def metrics(self) -> dict:
        return self.llm_client.metrics()

    async def health(self) -> HealthStatus:
        return HealthStatus(reachable=await self.llm_client.ping())

    async def generate(self, text: str) -> list[dict]:
        return await self._agent.run(self._build_prompt(text))
