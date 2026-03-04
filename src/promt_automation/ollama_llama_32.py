import dspy
import ollama

class OllamaLlama32(dspy.Module):
    def __init__(self, model_name="llama3.2"):
        super().__init__()
        self.model_name = model_name

    def complete(self, prompt: str, **kwargs) -> str:
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
