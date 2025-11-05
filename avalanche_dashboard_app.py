# import packages
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import asyncio

from models.ollama import OllamaProvider
from models.settings import OllamaSettings

from data_handling import *
from templates.model_responses import analyze_all_reviews

# 2 button app -> load dataset + analyse sentiment

# load environment variables from .env file
load_dotenv()

# intializa Ollama class
settings = OllamaSettings()
ollama_object = OllamaProvider(settings)


def run_sentiment_analysis():
    """
    Run sentiment analysis on the dataframe in session state.
    """
    if "df" not in st.session_state:
        st.warning("Please ingest the dataset first.")
        return False
    
    # Check if cleaned column exists
    if "CLEANED_SUMMARY" not in st.session_state["df"].columns:
        st.warning("Please clean the data first (this is done automatically during ingestion).")
        return False
    
    try:
        # Create Streamlit-specific callbacks
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def streamlit_progress_callback(current: int, total: int, message: str):
            """Progress callback for Streamlit UI."""
            progress_bar.progress(current / total)
            status_text.text(message)
        
        def streamlit_warning_callback(message: str):
            """Warning callback for Streamlit UI."""
            st.warning(message)
        
        # Create a lambda that wraps the ollama chat method
        chat_fn = lambda prompt: ollama_object.chat(prompt)
        
        # Run async sentiment analysis with callbacks
        df_with_sentiment = asyncio.run(
            analyze_all_reviews(
                st.session_state["df"],
                chat_function=chat_fn,
                text_column="CLEANED_SUMMARY",
                progress_callback=streamlit_progress_callback,
                warning_callback=streamlit_warning_callback
            )
        )
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.session_state["df"] = df_with_sentiment
        return True
    except Exception as e:
        st.error(f"Error during sentiment analysis: {str(e)}")
        return False


st.title("Avalanche Dashboard App")
st.write(f"This app loads avalanche data and analyses sentiment using {ollama_object.model_name} via Ollama.")

# Layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Ingest Dataset"):
        if load_dataset():
            if clean_text_column("SUMMARY", "CLEANED_SUMMARY"):
                st.success("Dataset loaded successfully and Reviews parsed and cleaned!")

with col2:
    if st.button("🧹 Analyze Sentiments"):
        with st.spinner("Analyzing sentiments with Ollama..."):
            if run_sentiment_analysis():
                st.success("Sentiment analysis completed!")
                st.balloons()

# Display the dataframe if it exists in session state
if "df" in st.session_state:

    # Product filter dropdown
    filtered_df = create_product_filter()
        
    if filtered_df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(filtered_df)

        # Display sentiment chart
        display_sentiment_chart()
        if 'SENTIMENT' in filtered_df.columns:
            display_sentiment_pie_chart(filtered_df)