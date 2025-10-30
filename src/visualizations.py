import re
from collections import Counter
from itertools import combinations
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
from wordcloud import WordCloud


# -----------------
# 1. Keyword Momentum Chart
# -----------------
def plot_keyword_momentum(all_articles, keywords):
    df_list = []
    for kw in keywords:
        yearly = all_articles[all_articles["content"].str.contains(kw, case=False, na=False)]
        yearly = yearly.groupby("year").size().reset_index(name="count")
        yearly["keyword"] = kw
        df_list.append(yearly)
    df = pd.concat(df_list)

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("count:Q", title="Frequency"),
            color="keyword:N",
            tooltip=["year", "keyword", "count"]
        )
        .properties(title="📈 Fraud Keyword Momentum Over Time")
        .configure_axis(labelFontSize=12, titleFontSize=13)
    )
    st.altair_chart(chart, use_container_width=True)


# -----------------
# 2. Top Fraud Terms Bar Chart
# -----------------
def plot_top_terms(all_articles, top_n=15, focus_terms=None):
    """
    Displays the most frequent fraud-related terms in article text,
    excluding common English stopwords and filler words.
    """

    # Combine all text
    text = " ".join(all_articles["content"].dropna().tolist()).lower()

    # Tokenize and keep alphabetic words (min 4 letters)
    words = re.findall(r"\b[a-z]{4,}\b", text)

    # Basic stopword list (expand as needed)
    stopwords = {
        "this", "that", "with", "from", "have", "been", "their", "will", "they",
        "which", "would", "could", "should", "about", "there", "where", "when",
        "were", "what", "these", "those", "because", "while", "your", "ours",
        "into", "also", "some", "than", "then", "them", "very", "only", "such",
        "other", "more", "over", "under", "within", "upon", "after", "before",
        "said", "many", "most", "each", "any", "been", "its", "may", "might",
        "must", "can", "shall", "was", "for", "and", "the", "that", "this", 
        "those", "these", "there", "here", "were", "has", "had", "been", "than",
        "year", "month", "day"
    }

    # Optional fraud-context focus
    if focus_terms:
        words = [w for w in words if any(term in w for term in focus_terms)]

    # Filter out stopwords and short words
    words = [w for w in words if w not in stopwords and len(w) > 3]

    # Count frequency
    common = Counter(words).most_common(top_n)
    df = pd.DataFrame(common, columns=["word", "count"])

    # Plot
    fig = px.bar(
        df.sort_values("count"),
        x="count",
        y="word",
        orientation="h",
        title="🏦 Top Fraud-Related Terms in Press Releases",
        text="count",
        color="count",
        color_continuous_scale="blues"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis=dict(title=""), xaxis=dict(title="Frequency"))
    st.plotly_chart(fig, use_container_width=True)


# -----------------
# 3. Keyword Co-Occurrence Network (Plotly-style)
# -----------------
def plot_keyword_network(all_articles, min_count=10):
    text_data = all_articles["content"].dropna().str.lower().tolist()
    words = [re.findall(r"\b[a-z]{4,}\b", t) for t in text_data]
    pairs = Counter([tuple(sorted(p)) for w in words for p in combinations(set(w), 2)])
    df = pd.DataFrame(pairs.most_common(100), columns=["pair", "count"])
    df = df[df["count"] >= min_count]
    df[["source", "target"]] = pd.DataFrame(df["pair"].tolist(), index=df.index)

    fig = px.scatter(
        df,
        x="source",
        y="target",
        size="count",
        color="count",
        title="🔗 Fraud Keyword Co-Occurrence Network",
        hover_data=["count"],
        color_continuous_scale="purples"
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------
# 4. Rolling + Cumulative Trend Summary
# -----------------
def plot_trend_summary(all_articles):
    df = all_articles.groupby("year").size().reset_index(name="count")
    df["rolling_avg"] = df["count"].rolling(window=3, min_periods=1).mean()
    df["cumulative"] = df["count"].cumsum()

    fig = px.line(
        df,
        x="year",
        y=["count", "rolling_avg", "cumulative"],
        title="📊 Fraud Press Release Trend Overview",
        labels={"value": "Articles", "variable": "Metric"},
        markers=True,
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------
# 5. Thematic Word Cloud (Fraud-Focused)
# -----------------
def plot_thematic_wordcloud(all_articles, keywords):
    text = " ".join(all_articles["content"].dropna().tolist()).lower()
    fraud_text = " ".join(
        [w for w in re.findall(r"\b[a-z]{4,}\b", text) if any(k in w for k in keywords)]
    )
    wc = WordCloud(
        width=1000, height=500, background_color="white",
        colormap="plasma", max_words=100
    ).generate(fraud_text)
    st.image(wc.to_array(), caption="💬 Dominant Fraud Vocabulary", use_column_width=True)
