from infrastructure.config import Settings
from infrastructure.inference.base import InferenceBackend


class InferenceFactory:
    _backend: InferenceBackend | None = None

    @classmethod
    def initialize(cls, config: Settings) -> None:
        if cls._backend is not None:
            return

        backend = config.inference_backend

        if backend == "ollama":
            from infrastructure.inference.ollama_backend import OllamaBackend
            cls._backend = OllamaBackend.build(
                model=config.ollama_model,
                host=config.ollama_host,
            )
        else:
            raise ValueError(
                f"Unknown inference_backend: {backend!r}. "
                f"Supported: 'ollama'"
            )

    @classmethod
    def get_backend(cls) -> InferenceBackend:
        if cls._backend is None:
            raise RuntimeError("InferenceFactory not initialized. Call initialize() at startup.")
        return cls._backend
