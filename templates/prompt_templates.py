def create_sentiment_prompt(review_text: str) -> str:
    """
    Create a prompt for sentiment analysis.
    
    Args:
        review_text: The review text to analyze
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Analyze the sentiment of the following product review and respond with ONLY a JSON object in this exact format:
{{"sentiment": "positive" or "negative" or "neutral", "score": float between -1.0 and 1.0, "confidence": float between 0.0 and 1.0}}

Review: {review_text}

Remember: Respond with ONLY the JSON object, no additional text."""
    
    return prompt