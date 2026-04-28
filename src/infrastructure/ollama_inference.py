import ollama
from infrastructure.config import OLLAMA_MODEL, OLLAMA_HOST


class OllamaInference:
    """
    Singleton that holds the Ollama model reference.
    Initialized once at startup; never recreated per request.
    """

    _instance: "OllamaInference | None" = None

    def __init__(self, model: str, host: str) -> None:
        self.model = model
        self._client = ollama.Client(host=host)

    @classmethod
    def get_instance(cls) -> "OllamaInference":
        if cls._instance is None:
            raise RuntimeError(
                "OllamaInference has not been initialized. Call OllamaInference.initialize() at startup."
            )
        return cls._instance

    @classmethod
    def initialize(cls, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST) -> "OllamaInference":
        if cls._instance is None:
            instance = cls(model=model, host=host)
            instance._verify_model()
            cls._instance = instance
        return cls._instance

    def _verify_model(self) -> None:
        available = [m.model for m in self._client.list().models]
        if self.model not in available:
            raise RuntimeError(
                f"Model '{self.model}' is not available in Ollama. "
                f"Run: ollama pull {self.model}"
            )

    def metrics(self) -> dict:
        return {
            "method": "LLM",
            "provider": "ollama",
            "model": self.model,
        }

    def ping(self) -> bool:
        """Returns True if the Ollama server is reachable and the model is available."""
        try:
            available = [m.model for m in self._client.list().models]
            return self.model in available
        except Exception:
            return False

    def chat(self, prompt: str) -> str:
        response = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content
