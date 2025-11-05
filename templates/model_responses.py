import json
from typing import Dict, Any, Callable, Awaitable, Optional, Protocol
from dataclasses import dataclass
import pandas as pd

from .prompt_templates import create_sentiment_prompt


class WarningCallback(Protocol):
    """Protocol for warning callback functions."""
    def __call__(self, message: str) -> None:
        """
        Report a warning message.
        
        Args:
            message: Warning message to display
        """
        ...

class ProgressCallback(Protocol):
    """Protocol for progress callback functions."""
    def __call__(self, current: int, total: int, message: str) -> None:
        """
        Report progress of an operation.
        
        Args:
            current: Current item number
            total: Total number of items
            message: Progress message
        """
        ...

@dataclass
class SentimentResult:
    """Result of sentiment analysis for a single review."""
    sentiment: str  # "positive", "negative", "neutral", or "error"
    score: float    # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0

async def analyze_single_review(
    review_text: str,
    chat_function: Callable[[str], Awaitable[Any]],
    warning_callback: Optional[WarningCallback] = None
) -> SentimentResult:
    """
    Analyze sentiment of a single review using any LLM provider.
    
    Args:
        review_text: The review text to analyze
        chat_function: Async function that takes a prompt and returns a response
                      Example: lambda prompt: ollama_object.chat(prompt)
        warning_callback: Optional callback for warning messages
        
    Returns:
        SentimentResult object with sentiment, score, and confidence
    """
    try:
        prompt = create_sentiment_prompt(review_text)
        response = await chat_function(prompt)
        
        # Parse the response - handle both string and dict responses
        if isinstance(response, str):
            # Try to extract JSON from response (in case there's extra text)
            # Look for JSON pattern between curly braces
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx + 1]
                result = json.loads(json_str)
            else:
                result = json.loads(response)
        else:
            result = response
            
        return SentimentResult(
            sentiment=result.get("sentiment", "neutral"),
            score=float(result.get("score", 0.0)),
            confidence=float(result.get("confidence", 0.5))
        )
    except json.JSONDecodeError as e:
        if warning_callback:
            warning_callback(f"Error parsing JSON response: {str(e)}")
        return SentimentResult(sentiment="error", score=0.0, confidence=0.0)
    except Exception as e:
        if warning_callback:
            warning_callback(f"Error analyzing review: {str(e)}")
        return SentimentResult(sentiment="error", score=0.0, confidence=0.0)


async def analyze_all_reviews(
    df: pd.DataFrame,
    chat_function: Callable[[str], Awaitable[Any]],
    text_column: str = "CLEANED_SUMMARY",
    progress_callback: Optional[ProgressCallback] = None,
    warning_callback: Optional[WarningCallback] = None
) -> pd.DataFrame:
    """
    Analyze sentiment for all reviews in the dataframe.
    
    Args:
        df: DataFrame containing reviews
        chat_function: Async function that takes a prompt and returns a response
                      Example: lambda prompt: ollama_object.chat(prompt)
        text_column: Column name containing text to analyze
        progress_callback: Optional callback for progress updates
        warning_callback: Optional callback for warnings (only first review)
        
    Returns:
        DataFrame with added sentiment columns (SENTIMENT, SENTIMENT_SCORE, CONFIDENCE)
    """
    results = []
    total_reviews = len(df)
    
    for idx, row in df.iterrows():
        # Report progress
        if progress_callback:
            progress_callback(
                current=idx + 1,
                total=total_reviews,
                message=f"Analyzing review {idx + 1} of {total_reviews}..."
            )
        
        # Analyze the review (only show warnings for first review)
        sentiment_result = await analyze_single_review(
            row[text_column],
            chat_function,
            warning_callback=warning_callback if idx == 0 else None
        )
        results.append(sentiment_result)
    
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Add results to dataframe
    df_copy["SENTIMENT"] = [r.sentiment for r in results]
    df_copy["SENTIMENT_SCORE"] = [r.score for r in results]
    df_copy["CONFIDENCE"] = [r.confidence for r in results]
    
    return df_copy