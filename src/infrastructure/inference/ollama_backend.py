import ollama


class OllamaBackend:
    def __init__(self, model: str, host: str) -> None:
        self.model = model
        self._client = ollama.Client(host=host)

    @classmethod
    def build(cls, model: str, host: str) -> "OllamaBackend":
        backend = cls(model=model, host=host)
        backend._verify_model()
        return backend

    def _verify_model(self) -> None:
        available = [m.model for m in self._client.list().models]
        if self.model not in available:
            raise RuntimeError(
                f"Model '{self.model}' is not available in Ollama. "
                f"Run: ollama pull {self.model}"
            )

    def chat(self, prompt: str) -> str:
        response = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content

    def ping(self) -> bool:
        try:
            available = [m.model for m in self._client.list().models]
            return self.model in available
        except Exception:
            return False

    def metrics(self) -> dict:
        return {
            "method": "LLM",
            "provider": "ollama",
            "model": self.model,
        }
