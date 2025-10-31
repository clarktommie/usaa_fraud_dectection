import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords", quiet=True)

def detect_patterns(all_articles: pd.DataFrame, focus_term: str = "fraud"):
    """
    Detect frequency trends and related phrases for a specific focus term
    (e.g., 'fraud', 'cyber', 'money laundering').
    Returns keyword counts, related phrases, and yearly trend data.
    """
    if all_articles.empty or "content" not in all_articles:
        return {
            "keyword_counts": {},
            "top_phrases": [],
            "yearly_trend": pd.DataFrame(),
        }

    # --- Normalize ---
    all_articles = all_articles.copy()
    all_articles["content"] = all_articles["content"].astype(str).str.lower()
    all_articles["year"] = pd.to_datetime(all_articles["date"], errors="coerce").dt.year

    # -----------------------------
    # 1. Detect keyword trends
    # -----------------------------
    query = focus_term.lower().strip()
    all_articles["match"] = all_articles["content"].apply(
        lambda t: len(re.findall(rf"\b{re.escape(query)}\b", t))
    )

    yearly_trend = (
        all_articles.groupby("year")["match"].sum().reset_index().sort_values("year")
    )
    yearly_trend.columns = ["Year", f"Mentions of '{focus_term}'"]

    # -----------------------------
    # 2. Extract top keywords
    # -----------------------------
    text = " ".join(all_articles["content"])
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text)
    stop = set(stopwords.words("english")).union({
        "federal","reserve","board","press","release","bank",
        "financial","system","statement","policy","committee",
        "public","meeting","minutes","update","news","report",
        "https","gov","page","joint","agencies","statement"
    })
    filtered = [w for w in words if w not in stop]
    keyword_counts = dict(Counter(filtered).most_common(30))

    # -----------------------------
    # 3. Extract related phrases (1–3 grams)
    # -----------------------------
    text = re.sub(r'\s+', ' ', text)  # collapse excessive whitespace

    vec = CountVectorizer(stop_words=list(stop), ngram_range=(2, 5), max_features=500)
    X = vec.fit_transform([text])
    phrase_freq = list(zip(vec.get_feature_names_out(), X.toarray()[0]))

    top_phrases = [
        (p, c) for p, c in sorted(phrase_freq, key=lambda x: x[1], reverse=True)
        if query in p
    ][:15]

    # If none found with query, fall back to most frequent general phrases
    if not top_phrases:
        top_phrases = sorted(phrase_freq, key=lambda x: x[1], reverse=True)[:15]

    return {
        "keyword_counts": keyword_counts,
        "top_phrases": top_phrases,
        "yearly_trend": yearly_trend,
    }
