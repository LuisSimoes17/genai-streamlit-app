import ollama

from models.settings import OllamaSettings
from models.base_provider import BaseProvider


class OllamaProvider(BaseProvider):
    """Provider for Ollama LLM."""

    def __init__(self, settings: OllamaSettings) -> None:
        self.settings = settings
        
        # Setup client with optional host and API key
        client_kwargs = {}
        
        # for local Ollama, host is None and defaults to http://localhost:11434
        # for Ollama Cloud, host should be set to https://ollama.com and API key must be provided
        if settings.host:
            client_kwargs['host'] = settings.host
            
        if settings.api_key:
            client_kwargs['headers'] = {
                'Authorization': f'Bearer {settings.api_key}'
            }
        
        self.client = ollama.AsyncClient(**client_kwargs)

    async def chat(self, prompt: str) -> str:
        """Send a chat completion request to Ollama."""
        response = await self.client.chat(
            model=self.settings.default_model,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': self.settings.temperature,
                'num_predict': self.settings.max_tokens,  # Ollama uses num_predict instead of max_tokens
            }
        )
        return response['message']['content']

    async def content_embedding(self, content: str) -> list[float]:
        """Generate embeddings for the given content."""
        response = await self.client.embeddings(
            model=self.settings.embedding_model,
            prompt=content
        )
        return response['embedding']

    @property
    def model_name(self) -> str:
        return self.settings.default_model

    def get_model_parameters(self) -> dict:
        """Return the parameters for the model."""
        return {
            'model': self.settings.default_model,
            'temperature': self.settings.temperature,
            'max_tokens': self.settings.max_tokens,
            'host': self.settings.host or 'local',
        }