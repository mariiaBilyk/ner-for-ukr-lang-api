import structlog

from application.extraction_strategy import ExtractionStrategy
from prompts.registry import PromptRegistry

logger = structlog.get_logger()


class NERAgent:
    _FEEDBACK_NO_JSON_PROMPT    = "feedback-invalid-json"
    _FEEDBACK_BAD_LABELS_PROMPT = "feedback-invalid-labels"
    _BAD_LABELS_PLACEHOLDER     = "{{BAD_LABELS}}"

    def __init__(self, strategy: ExtractionStrategy, registry: PromptRegistry,
                 max_attempts: int) -> None:
        self._strategy     = strategy
        self._registry     = registry
        self._max_attempts = max_attempts

    async def run(self, prompt: str) -> list[dict]:
        feedback = ""
        for attempt in range(1, self._max_attempts + 1):
            result = await self._strategy.extract(prompt + feedback)

            if result is not None:
                if attempt > 1:
                    logger.info("ner_agent_recovered", attempt=attempt)
                return result

            failure = self._strategy.last_failure
            feedback = self._build_feedback(failure)
            logger.warning(
                "ner_agent_retry",
                attempt=attempt,
                max_attempts=self._max_attempts,
                has_json=failure.has_json if failure else False,
                bad_labels=sorted(failure.bad_labels) if failure else [],
            )

        logger.error("ner_agent_failed", max_attempts=self._max_attempts)
        return []

    def _build_feedback(self, result) -> str:
        if result is None or not result.has_json:
            return "\n\n" + self._registry.get(self._FEEDBACK_NO_JSON_PROMPT)
        tmpl = self._registry.get(self._FEEDBACK_BAD_LABELS_PROMPT)
        return "\n\n" + tmpl.replace(
            self._BAD_LABELS_PLACEHOLDER,
            ", ".join(sorted(result.bad_labels)),
        )
