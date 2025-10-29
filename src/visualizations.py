import altair as alt
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import re
from collections import Counter
from itertools import combinations
import networkx as nx

def plot_heatmap(all_articles, keywords):
    heatmap_data = []
    for kw in keywords:
        yearly_kw = all_articles[all_articles["content"].str.contains(kw, case=False, na=False)]
        counts = yearly_kw.groupby("year").size().reset_index(name="count")
        counts["keyword"] = kw
        heatmap_data.append(counts)
    df = pd.concat(heatmap_data)

    chart = alt.Chart(df).mark_rect().encode(
        x="year:O", y="keyword:N", color="count:Q",
        tooltip=["year", "keyword", "count"]
    ).properties(title="Keyword Frequency by Year (Heatmap)")
    st.altair_chart(chart, use_container_width=True)


def plot_rolling_trend(all_articles):
    df = all_articles.groupby("year").size().reset_index(name="count")
    df["rolling_avg"] = df["count"].rolling(window=3, min_periods=1).mean()
    fig = px.line(df, x="year", y="rolling_avg",
                  title="3-Year Rolling Average: Press Release Volume",
                  markers=True)
    st.plotly_chart(fig, use_container_width=True)


def plot_wordcloud(all_articles):
    text = " ".join(all_articles["content"].dropna().tolist())
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    st.image(wc.to_array(), caption="Most Frequent Terms in All Articles")


def plot_network(all_articles):
    words = [re.findall(r"\b[a-z]{4,}\b", text.lower()) for text in all_articles["content"].dropna()]
    pairs = Counter([tuple(sorted(pair)) for w in words for pair in combinations(set(w), 2)])
    df = pd.DataFrame(pairs.most_common(30), columns=["pair", "count"])
    df[["source", "target"]] = pd.DataFrame(df["pair"].tolist(), index=df.index)
    G = nx.from_pandas_edgelist(df, "source", "target", "count")
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx(G, node_size=500, font_size=8, with_labels=True)
    st.pyplot(fig)
