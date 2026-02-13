from typing import Optional
from pydantic_settings import BaseSettings

class BaseProviderSettings(BaseSettings):
    temperature: float = 0
    top_p: float = 0.9
    max_tokens: Optional[int] = None

class OllamaSettings(BaseProviderSettings):
    default_model: str = "gemma:2b"
    embedding_model: str = "gemma.2b"

    # Connection settings
    host: Optional[str] = None  # None = local (http://localhost:11434), or set to cloud URL
    api_key: Optional[str] = None  # Required for Ollama Cloud