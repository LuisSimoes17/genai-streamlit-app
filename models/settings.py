import os
from typing import Optional

from pydantic_settings import BaseSettings

class BaseProviderSettings(BaseSettings):
    temperature: int = 0
    top_p: float = 0.9
    max_tokens: Optional[int] = None

class OllamaSettings(BaseProviderSettings):
    default_model: str = "gemma3"
    embedding_model: str = "llama3.2"