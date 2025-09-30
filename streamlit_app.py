# import packages
from dotenv import load_dotenv
#import openai
import os
import streamlit as st

import asyncio

from models.ollama import OllamaProvider
from models.settings import OllamaSettings

@st.cache_data
def get_response(user_prompt, temperature):
    response = client.chat.completions.create(
            model="sonar",
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=100
        )
    return response

# load environment variables from .env file
load_dotenv()
#perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

# initalize OpenAI client
#client = openai.OpenAI(
#    api_key=perplexity_api_key,
#    base_url="https://api.perplexity.ai"
#)

# intializa Ollama class
settings = OllamaSettings()
ollama_object = OllamaProvider(settings)


st.title("Hello, GenAI!")
st.write("This is a simple Streamlit app using GPT-4o via Perplexity API.")

# add a text input box for the user prompt
user_prompt = st.text_input("Enter your prompt:", "Explain Generative AI in one sentence.")

# add a slider for temperature
temperature = st.slider(
    "Model Temperature:",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,
    help="Controls randomness: 0 = deterministic, 1 = very creative"
)

with st.spinner("Generating response..."):
    # make a request to the Perplexity API
    #response = get_response(user_prompt, temperature)
    ollama_object.settings.temperature = temperature
    response = asyncio.run(ollama_object.chat(user_prompt))
    # print the response in the Streamlit app
    #st.write(response.choices[0].message.content)
    st.write(response)