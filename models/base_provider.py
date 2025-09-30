from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat(self, prompt: str) -> str:
        """Invoke the LLM with the given prompt and return the response."""
        pass

    async def content_embedding(self, content: str) -> list[float]:
        """Generate an embedding for the given content."""
        pass

    @property
    def model_name(self) -> str:
        """Return the name of the model used by this provider."""
        pass

    def get_model_parameters(self) -> dict:
        """Return the parameters for the model."""
        pass
