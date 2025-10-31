import re
from collections import Counter
import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk

# Download stopwords once (safe to leave here — runs only if not downloaded)
nltk.download('stopwords', quiet=True)

def analyze_articles(articles):
    """Analyze retrieved articles for temporal, thematic, and sentiment insights."""
    if not articles:
        return None

    # ---------------------------
    # 1. Temporal Trend Analysis
    # ---------------------------
    years = []
    for a in articles:
        if a.get("date"):
            found = re.findall(r"\b(20\d{2}|19\d{2})\b", str(a["date"]))
            if found:
                years.append(int(found[0]))
    year_counts = Counter(years)

    # ---------------------------
    # 2. Keyword Frequency
    # ---------------------------
    text = " ".join(a.get("content", "").lower() for a in articles)
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text)

    # Combine NLTK + domain stopwords
    nltk_stop = set(stopwords.words("english"))
    domain_stop = set([
        "https","federal","reserve","board","press","release","bank","banks",
        "financial","system","statement","policy","committee","public",
        "report","meeting","minutes","news","update","service","division",
        "department","media","contact","agency","institution","data",
        "figure","section","table","appendix","document"
    ])
    stop_set = nltk_stop.union(domain_stop)

    filtered = [w for w in words if w not in stop_set]
    common_words = Counter(filtered).most_common(25)

    # ---------------------------
    # 3. Fraud-Related Signal Extraction
    # ---------------------------
    fraud_signals = [
        "fraud","scam","bribery","cyber","identity","aml","launder",
        "sanction","suspicious","scheme","misconduct","investigation",
        "enforcement","penalty","settlement","whistleblower","compliance",
        "security","risk","loss","arrest"
    ]

    signal_hits = {word: text.count(word) for word in fraud_signals if text.count(word) > 0}
    top_signals = dict(sorted(signal_hits.items(), key=lambda x: x[1], reverse=True)[:10])

    # ---------------------------
    # 4. Sentiment Snapshot
    # ---------------------------
    sentiment = TextBlob(text).sentiment
    sentiment_label = (
        "Negative" if sentiment.polarity < -0.1
        else "Positive" if sentiment.polarity > 0.1
        else "Neutral"
    )

    # ---------------------------
    # 5. Executive Summary (Auto-Generated)
    # ---------------------------
    if top_signals:
        dominant_themes = ", ".join(list(top_signals.keys())[:5])
    else:
        dominant_themes = ", ".join([w for w, _ in common_words[:5]])

    summary = (
        f"Across {len(articles)} press releases, key compliance and fraud-related activity centers on "
        f"{dominant_themes}. Overall sentiment is **{sentiment_label.lower()}**, "
        f"indicating {'heightened risk awareness' if sentiment_label == 'Negative' else 'stable enforcement tone' if sentiment_label == 'Neutral' else 'positive remediation outcomes'}. "
        f"Observed activity spans {', '.join(map(str, year_counts.keys())) or 'recent years'}, "
        f"reflecting evolving oversight and regulatory communication priorities."
    )

    # ---------------------------
    # 6. Output Object
    # ---------------------------
    return {
        "year_counts": year_counts,
        "common_words": common_words,
        "fraud_signals": top_signals,
        "sentiment": sentiment_label,
        "summary": summary,
        "total_articles": len(articles),
        "text": " ".join(filtered),
    }
