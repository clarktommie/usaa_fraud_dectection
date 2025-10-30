# src/pattern_detector.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from fastembed import TextEmbedding

def detect_patterns(all_articles, years=5, n_components=2, top_k=200):
    df = all_articles.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ---- 1. Filter to timeframe ----
    if years:
        cutoff = datetime.now() - timedelta(days=365 * years)
        df = df[df["date"] >= cutoff]

    if df.empty or df["content"].dropna().empty:
        return {
            "keyword_counts": {},
            "top_phrases": [],
            "hourly_pattern": {},
            "pca_embedding": None,
        }

    # ---- 2. Build embeddings for each article ----
    embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
    texts = df["content"].fillna("").tolist()
    embeddings = np.array(list(embedder.embed(texts)))

    # ---- 3. Learn the "fraud direction" automatically ----
    # Get embedding of the word 'fraud' as a reference anchor
    anchor_vec = np.array(list(embedder.embed(["fraud"])))[0]
    sims = cosine_similarity(embeddings, anchor_vec.reshape(1, -1)).flatten()

    # Rank by semantic closeness to "fraud"
    df["fraud_score"] = sims
    df_sorted = df.sort_values("fraud_score", ascending=False)
    top_df = df_sorted.head(min(top_k, len(df_sorted)))

    # ---- 4. Extract most informative terms from top semantic articles ----
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(top_df["content"].fillna(""))
    terms = np.array(tfidf.get_feature_names_out())
    scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    tfidf_ranking = pd.DataFrame({"term": terms, "score": scores})
    top_terms = dict(tfidf_ranking.sort_values("score", ascending=False).head(15).values)

    # ---- 5. N-gram phrases (optional) ----
    ngram_vec = CountVectorizer(stop_words="english", ngram_range=(2, 3))
    X_ngrams = ngram_vec.fit_transform(top_df["content"].fillna(""))
    ngram_terms = ngram_vec.get_feature_names_out()
    ngram_freqs = X_ngrams.toarray().sum(axis=0)
    ngram_df = pd.DataFrame({"term": ngram_terms, "count": ngram_freqs})
    top_phrases = list(
        ngram_df.sort_values("count", ascending=False)
        .head(10)
        .itertuples(index=False, name=None)
    )

    # ---- 6. Hourly pattern ----
    top_df["hour"] = pd.to_datetime(top_df["date"], errors="coerce").dt.hour
    hour_pattern = top_df["hour"].value_counts().sort_index().to_dict()

    # ---- 7. PCA visualization ----
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    pca_df = pd.DataFrame({
        "date": df["date"].astype(str),
        "PC1": reduced[:, 0],
        "PC2": reduced[:, 1],
        "fraud_score": sims
    })

    return {
        "keyword_counts": top_terms,
        "top_phrases": top_phrases,
        "hourly_pattern": hour_pattern,
        "pca_embedding": pca_df.to_dict(orient="records"),
    }
