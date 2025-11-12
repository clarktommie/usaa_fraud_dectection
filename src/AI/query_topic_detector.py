"""
query_topic_detector.py
---------------------------------
Detects fraud-related topics or categories from user questions.
Used to filter complaint data dynamically so visuals change with context.
"""

import re

def detect_fraud_topics(query: str):
    """
    Detect relevant fraud topics from the user's query text.
    Returns a list of matching keywords (normalized).
    """
    query = query.lower()

    topic_map = {
        "identity theft": ["identity", "id theft", "stolen identity"],
        "wire fraud": ["wire", "transfer", "money wiring"],
        "cyber fraud": ["cyber", "phishing", "impersonation", "breach", "ransomware"],
        "scam": ["scam", "scheme", "deceptive", "fraudulent"],
        "aml compliance": ["aml", "money laundering", "bsa", "kyc"],
        "consumer protection": ["consumer", "complaint", "dispute", "unauthorized"],
    }

    detected = []
    for label, words in topic_map.items():
        if any(re.search(rf"\b{w}\b", query) for w in words):
            detected.append(label)

    # default fallback
    if not detected:
        detected = ["general fraud"]

    return detected
