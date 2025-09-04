# import packages
from dotenv import load_dotenv
import openai
import os

# load environment variables from .env file
load_dotenv()
perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

# initalize OpenAI client
client = openai.OpenAI(
    api_key=perplexity_api_key,
    base_url="https://api.perplexity.ai"
)

response = client.chat.completions.create(
    model="sonar",
    messages=[
        {"role": "user", "content": "Hello! How can I use GPT-4o in my applications?"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)