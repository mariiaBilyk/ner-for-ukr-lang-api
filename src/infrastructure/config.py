from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    inference_backend: str = "ollama"   # ollama | huggingface | azure_ml

    # Ollama
    ollama_model: str
    ollama_host: str  = "http://localhost:11434"

    # HuggingFace (future)
    hf_model_name: str = ""

    # Azure ML (future)
    azure_endpoint_url: str = ""
    azure_api_key: str      = ""

    # Prompt
    prompt_name: str    = "ner"
    prompt_version: str = ""    # empty → registry picks latest

    # NER Agent
    ner_agent_max_attempts: int = 3
    extraction_strategy: str    = "simple"   # simple | self_critique


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
