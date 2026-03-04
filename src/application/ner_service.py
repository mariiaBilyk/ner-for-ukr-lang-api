import json

class NerService:

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def _parse(self, raw: str):
        try:
            entities = json.loads(raw)
            return [{"text": entity["text"], "label": entity["type"]} for entity in entities]
        except json.JSONDecodeError:
            return []

    async def generate(self, text: str):
        raw = await self.llm_client.generate(text)
        return self._parse(raw)