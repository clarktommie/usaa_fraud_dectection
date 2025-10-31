import os
import sys
import torch
import pandas as pd
import streamlit as st
from textblob import TextBlob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

from transformers import pipeline

# Force CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.modules["torch"] = torch

# storytelling_summary.py
# -----------------
# Cache Summarizer
# -----------------
@st.cache_resource
def load_local_summarizer():
    """
    Load and cache a local summarization model using Hugging Face transformers.
    This uses a distilled BART model (fast, small, no API key required).
    """
    try:
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=-1  # CPU mode
        )
        return summarizer
    except Exception as e:
        st.error(f"Failed to load local summarizer: {e}")
        return None


# -----------------
# Storytelling Summary Function
# -----------------
def summarize_patterns_storytelling(patterns, query_sentence):
    """
    Generate a narrative summary for detected fraud or compliance trends
    using a locally hosted lightweight summarizer (DistilBART).
    """
    if not patterns.get("keyword_counts") or not patterns.get("top_phrases"):
        return f"No strong narrative patterns detected for '{query_sentence}'."

    # Combine top keywords + phrases into a contextual text block
    context_text = " ".join(
        list(patterns["keyword_counts"].keys()) +
        [p for p, _ in patterns["top_phrases"]]
    )

    summarizer = load_local_summarizer()
    if summarizer is None:
        return "Summarizer not available — please check model installation."

    try:
        # No prompt template needed — the model naturally summarizes
        text_input = f"{query_sentence}. {context_text}"
        result = summarizer(
            text_input,
            max_length=120,
            min_length=30,
            do_sample=False
        )
        summary_text = result[0]['summary_text']
    except Exception as e:
        summary_text = f"(Summarization error: {e})"

    # Sentiment analysis
    sentiment = TextBlob(summary_text).sentiment.polarity
    if sentiment > 0.15:
        tone = "optimistic tone — reflecting improved oversight or reduced fraud risk."
    elif sentiment < -0.15:
        tone = "cautious tone — suggesting elevated concern or systemic vulnerabilities."
    else:
        tone = "neutral tone — indicating balanced or stable trends."

    return f"{summary_text} Overall, these narratives carry a {tone}"
