"""
Reusable Streamlit Data Loading and Cleaning Module

This module provides reusable functions for loading and cleaning datasets
in Streamlit applications. Import these functions into your apps as needed.

Usage in other apps:
    from data_loader import load_dataset, clean_text_column, create_product_filter
"""

import streamlit as st
import pandas as pd
import re
import os
from typing import Optional, Callable
import plotly.express as px


# ============================================================================
# REUSABLE FUNCTIONS - Import these into other Streamlit apps
# ============================================================================

def get_dataset_path(filename: str = "customer_reviews.csv", 
                     subfolder: str = "data") -> str:
    """
    Construct path to dataset file relative to current script.
    
    Args:
        filename: Name of the CSV file
        subfolder: Subdirectory containing the file
        
    Returns:
        Full path to the dataset file
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, subfolder, filename)
    return csv_path


def clean_text(text: str) -> str:
    """
    Clean text by converting to lowercase, removing punctuation.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def load_dataset(file_path: Optional[str] = None, 
                 session_key: str = "df",
                 filename: str = "customer_reviews.csv") -> bool:
    """
    Load dataset into Streamlit session state.
    
    Args:
        file_path: Path to CSV file (if None, uses get_dataset_path)
        session_key: Session state key to store dataframe
        filename: Filename to load if file_path not provided
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if file_path is None:
            file_path = get_dataset_path(filename)
        
        st.session_state[session_key] = pd.read_csv(file_path)
        return True
    except FileNotFoundError:
        st.error(f"Dataset not found at: {file_path}")
        return False
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return False


def clean_text_column(source_column: str, 
                      target_column: str,
                      session_key: str = "df",
                      cleaning_func: Callable = clean_text) -> bool:
    """
    Apply text cleaning to a dataframe column.
    
    Args:
        source_column: Name of column to clean
        target_column: Name of new column for cleaned data
        session_key: Session state key containing dataframe
        cleaning_func: Function to apply for cleaning
        
    Returns:
        True if successful, False otherwise
    """
    if session_key not in st.session_state:
        st.warning("Please load the dataset first.")
        return False
    
    try:
        df = st.session_state[session_key]
        df[target_column] = df[source_column].apply(cleaning_func)
        st.session_state[session_key] = df
        return True
    except Exception as e:
        st.error(f"Error cleaning column: {str(e)}")
        return False


def create_product_filter(column_name: str = "PRODUCT",
                         session_key: str = "df",
                         all_option: str = "All Products") -> Optional[pd.DataFrame]:
    """
    Create a product filter dropdown and return filtered dataframe.
    
    Args:
        column_name: Name of column containing product names
        session_key: Session state key containing dataframe
        all_option: Label for "all products" option
        
    Returns:
        Filtered dataframe or None if no data available
    """
    if session_key not in st.session_state:
        return None
    
    df = st.session_state[session_key]
    
    st.subheader("🔍 Filter by Product")
    products = [all_option] + list(df[column_name].unique())
    selected_product = st.selectbox("Choose a product", products)
    
    if selected_product != all_option:
        return df[df[column_name] == selected_product]
    else:
        return df


def display_sentiment_chart(product_column: str = "PRODUCT",
                           sentiment_column: str = "SENTIMENT_SCORE",
                           session_key: str = "df") -> None:
    """
    Display bar chart of average sentiment scores by product.
    
    Args:
        product_column: Column containing product names
        sentiment_column: Column containing sentiment scores
        session_key: Session state key containing dataframe
    """
    if session_key not in st.session_state:
        return
    
    df = st.session_state[session_key]
    
    st.subheader("Sentiment Score by Product")
    grouped = df.groupby(product_column)[sentiment_column].mean().reset_index()
    st.bar_chart(grouped.set_index(product_column))


def display_sentiment_pie_chart(df: pd.DataFrame,
                            session_key: str = "df",
                            sentiment_column: str = "SENTIMENT") -> None:
    """
    Display bar chart of average sentiment scores by product.
    
    Args:
        product_column: Column containing product names
        sentiment_column: Column containing sentiment scores
        session_key: Session state key containing dataframe
    """
    if session_key not in st.session_state:
        return
    
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df[sentiment_column].value_counts()
    fig_pie = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Overall Sentiment Distribution",
        color_discrete_map={
            "positive": "#2ecc71",
            "neutral": "#95a5a6",
            "negative": "#e74c3c"
        }
    )
    st.plotly_chart(fig_pie, use_container_width=True)



# ============================================================================
# MAIN APP - Runs when executing: streamlit run data_loader.py
# ============================================================================

def main():
    """Main application function."""
    
    st.title("Hello, GenAI!")
    st.write("This is your GenAI-powered data processing app.")

    # Layout two buttons side by side
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Ingest Dataset"):
            if load_dataset():
                st.success("Dataset loaded successfully!")

    with col2:
        if st.button("🧹 Parse Reviews"):
            if clean_text_column("SUMMARY", "CLEANED_SUMMARY"):
                st.success("Reviews parsed and cleaned!")

    # Display the dataframe if it exists in session state
    if "df" in st.session_state:
        # Product filter dropdown
        filtered_df = create_product_filter()
        
        if filtered_df is not None:
            st.subheader("Dataset Preview")
            st.dataframe(filtered_df)

            # Display sentiment chart
            display_sentiment_chart()


if __name__ == "__main__":
    main()