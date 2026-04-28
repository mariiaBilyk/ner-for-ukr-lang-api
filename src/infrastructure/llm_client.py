import asyncio
from infrastructure.inference.base import InferenceBackend


class LLMClient:
    def __init__(self, backend: InferenceBackend) -> None:
        self._backend = backend

    async def generate(self, prompt: str) -> str:
        return await asyncio.to_thread(self._backend.chat, prompt)

    async def ping(self) -> bool:
        return await asyncio.to_thread(self._backend.ping)

    def metrics(self) -> dict:
        return self._backend.metrics()
