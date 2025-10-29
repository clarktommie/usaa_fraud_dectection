import re
from collections import Counter
import pandas as pd

def analyze_articles(articles):
    """Analyze retrieved articles for key trends and term frequencies."""
    if not articles:
        return None

    years = []
    for a in articles:
        if a.get("date"):
            found = re.findall(r"\b(20\d{2}|19\d{2})\b", str(a["date"]))
            if found:
                years.append(int(found[0]))

    year_counts = Counter(years)

    # Combine all text and extract keywords
    text = " ".join(a.get("content", "").lower() for a in articles)
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text)
    stopwords = set([
        "https", "federal", "reserve", "board",
        "press", "release", "bank", "financial"
    ])
    filtered = [w for w in words if w not in stopwords]
    common_words = Counter(filtered).most_common(20)

    return {
        "year_counts": year_counts,
        "common_words": common_words,
        "total_articles": len(articles),
        "text": " ".join(filtered),
    }
