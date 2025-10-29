import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime, timedelta

def detect_patterns(all_articles, years=5):
    df = all_articles.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Filter to last `years` if specified
    if years:
        cutoff = datetime.now() - timedelta(days=years * 365)
        df = df[df["date"] >= cutoff]

    if df.empty or df["content"].dropna().empty:
        return {"keyword_counts": {}, "top_phrases": [], "hourly_pattern": {}}

    # ---- Unigrams (keywords) ----
    unigram_vec = CountVectorizer(stop_words="english", ngram_range=(1, 1))
    try:
        X_uni = unigram_vec.fit_transform(df["content"].fillna(""))
        uni_words = unigram_vec.get_feature_names_out()
        uni_freqs = X_uni.toarray().sum(axis=0)
        uni_df = pd.DataFrame({"term": uni_words, "count": uni_freqs})
        top_words = dict(uni_df.sort_values("count", ascending=False).head(10).values)
    except ValueError:
        top_words = {}

    # ---- Bigrams/Trigrams (phrases) ----
    ngram_vec = CountVectorizer(stop_words="english", ngram_range=(2, 3))
    try:
        X_ngrams = ngram_vec.fit_transform(df["content"].fillna(""))
        ngram_terms = ngram_vec.get_feature_names_out()
        ngram_freqs = X_ngrams.toarray().sum(axis=0)
        ngram_df = pd.DataFrame({"term": ngram_terms, "count": ngram_freqs})
        top_phrases = list(
            ngram_df.sort_values("count", ascending=False)
            .head(10)
            .itertuples(index=False, name=None)
        )
    except ValueError:
        top_phrases = []

    # ---- Hourly pattern ----
    df["hour"] = pd.to_datetime(df["date"], errors="coerce").dt.hour
    hour_pattern = df["hour"].value_counts().sort_index().to_dict()

    return {
        "keyword_counts": top_words,
        "top_phrases": top_phrases,
        "hourly_pattern": hour_pattern,
    }
