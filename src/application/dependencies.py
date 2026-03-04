from infrastructure.llm_client import LLMClient
from application.ner_service import NerService

def get_ner_service():
    client = LLMClient(model_name="mistral")
    return NerService(client)