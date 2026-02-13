"""
Simple test script for OllamaProvider.
Tests both local Ollama and Ollama Cloud configurations.
"""
import asyncio
from models.settings import OllamaSettings
from models.ollama import OllamaProvider
from dotenv import load_dotenv
import os

# load environment variables from .env file
load_dotenv()


async def test_local_ollama():
    """Test with local Ollama installation."""
    print("=" * 50)
    print("Testing LOCAL Ollama")
    print("=" * 50)
    
    # Settings for local Ollama (default)
    settings = OllamaSettings(
        default_model="gemma:2b",
        embedding_model="gemma:2b",
        temperature=0.7,
        max_tokens=100
    )
    
    provider = OllamaProvider(settings)
    
    print(f"Model: {provider.model_name}")
    print(f"Parameters: {provider.get_model_parameters()}")
    print()
    
    # Test chat
    try:
        prompt = "What is the capital of France? Answer in one sentence."
        print(f"Prompt: {prompt}")
        response = await provider.chat(prompt)
        print(f"Response: {response}")
        print()
    except Exception as e:
        print(f"Chat error: {e}")
        print()
    
    # Test embedding
    try:
        text = "Hello world"
        print(f"Generating embedding for: '{text}'")
        embedding = await provider.content_embedding(text)
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        print()
    except Exception as e:
        print(f"Embedding error: {e}")
        print()


async def test_ollama_cloud():
    """Test with Ollama Cloud."""
    print("=" * 50)
    print("Testing OLLAMA CLOUD")
    print("=" * 50)
    
    # Settings for Ollama Cloud
    # NOTE: Replace 'your-api-key-here' with your actual API key
    settings = OllamaSettings(
        default_model="gpt-oss:120b",
        host="https://ollama.com",  # Ollama Cloud endpoint
        api_key=os.getenv("OLLAMA_API_KEY"),  # Get from ollama.com
        temperature=0.7,
        max_tokens=100
    )
    
    provider = OllamaProvider(settings)
    
    print(f"Model: {provider.model_name}")
    print(f"Parameters: {provider.get_model_parameters()}")
    print()
    
    # Test chat
    try:
        prompt = "What is 2+2? Answer briefly."
        print(f"Prompt: {prompt}")
        response = await provider.chat(prompt)
        print(f"Response: {response}")
        print()
    except Exception as e:
        print(f"Chat error: {e}")
        print()


async def main():
    print("\n")
    print("🧪 OLLAMA PROVIDER TEST")
    print("\n")
    
    # Test local Ollama
    await test_local_ollama()
    
    print("\n")
    
    # Uncomment to test Ollama Cloud (requires API key)
    await test_ollama_cloud()
    
    print("=" * 50)
    print("Tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
