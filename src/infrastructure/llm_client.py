class LLMClient:
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def generate(self, prompt: str) -> str:
        return '[{"text": "John Doe", "type": "PERSON"}, {"text": "New York", "type": "LOCATION"}]'  # Dummy response