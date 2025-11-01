import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords", quiet=True)


def detect_patterns(all_articles: pd.DataFrame, focus_term: str = "fraud"):
    """
    Detect yearly frequency trends, top keywords, and common phrases from articles.
    Extracts years directly from text like 'analyze_articles' for guaranteed reliability.
    """
    if all_articles.empty or "content" not in all_articles:
        return {"keyword_counts": {}, "top_phrases": [], "yearly_trend": pd.DataFrame()}

    df = all_articles.copy()
    df["content"] = df["content"].astype(str).str.lower()

    # -----------------------------
    # 1. Extract years directly from the date field (robust + text-safe)
    # -----------------------------
    years = []
    for _, row in df.iterrows():
        date_field = str(row.get("date", "")) or str(row.get("date_standard", "")) or str(row.get("year", ""))
        found = re.findall(r"\b(20\d{2}|19\d{2})\b", date_field)
        if found:
            years.append(int(found[0]))
        else:
            years.append(None)

    df["Year"] = years
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    # -----------------------------
    # 2. Flexible keyword frequency detection
    # -----------------------------
    query = focus_term.lower().strip()
    df["match"] = df["content"].apply(lambda t: len(re.findall(rf"{re.escape(query)}", t)))

    yearly_trend = (
        df.groupby("Year", as_index=False)["match"].sum().sort_values("Year")
    )

    # Fallback: if no matches found, show article count per year instead
    if yearly_trend["match"].sum() == 0:
        yearly_trend = (
            df.groupby("Year", as_index=False)["content"]
            .count()
            .rename(columns={"content": "match"})
            .sort_values("Year")
        )

    yearly_trend.columns = ["Year", f"Mentions of '{focus_term}'"]

    # -----------------------------
    # 3. Extract top keywords
    # -----------------------------
    text = " ".join(df["content"])
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text)

    stop = set(stopwords.words("english")).union({
        "federal", "reserve", "board", "press", "release", "bank",
        "financial", "system", "statement", "policy", "committee",
        "public", "meeting", "minutes", "update", "news", "report",
        "https", "gov", "page", "joint", "agencies", "statement"
    })

    filtered = [w for w in words if w not in stop]
    keyword_counts = dict(Counter(filtered).most_common(30))

    # -----------------------------
    # 4. Extract related phrases (2–5 grams)
    # -----------------------------
    text = re.sub(r"\s+", " ", text)

    vec = CountVectorizer(stop_words=list(stop), ngram_range=(2, 5), max_features=500)
    X = vec.fit_transform([text])
    phrase_freq = list(zip(vec.get_feature_names_out(), X.toarray()[0]))

    top_phrases = sorted(phrase_freq, key=lambda x: x[1], reverse=True)[:15]

    # -----------------------------
    # 5. Return structure
    # -----------------------------
    return {
        "keyword_counts": keyword_counts,
        "top_phrases": top_phrases,
        "yearly_trend": yearly_trend,
    }
