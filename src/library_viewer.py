# src/library_viewer.py  (fixed)
import streamlit as st
import pandas as pd

from src.semantic_library import (
    get_topic_library,
    get_library_by_article_ids,
    topic_variants,
)

def render_library_viewer():
    st.markdown("## 📚 Semantic Library Viewer")

    topic = st.text_input("Enter topic to view:", placeholder="e.g., phishing, aml, sanctions")
    semantic_k = st.number_input(
        "Max semantic articles",
        min_value=10,
        max_value=500,
        value=80,
        step=10,
        help="Number of similar articles to consider when matching saved library entries.",
    )

    if st.button("Load Library"):
        if not topic.strip():
            st.warning("Enter a topic keyword to search the library.")
            return

        df = pd.DataFrame()
        semantic_notes = []

        # Attempt semantic lookup via article embeddings -> library article_ids
        try:
            from src.article_index import ArticleIndex  # local import to avoid hard failure when API key missing

            index = ArticleIndex.load()
            search_results = index.search(topic, top_k=int(semantic_k))
            article_ids = search_results["id"].dropna().unique().tolist()
            if article_ids:
                df = get_library_by_article_ids(article_ids)
                semantic_notes.append(f"Semantic match: {len(article_ids)} related articles, {len(df)} saved library entries.")
            else:
                semantic_notes.append("Semantic match found 0 related articles (no library entries).")
        except Exception as exc:
            st.warning(f"Semantic search unavailable: {exc}")

        # Fallback to topic variants if semantic lookup returned nothing
        if df.empty:
            variants = topic_variants(topic)
            if variants:
                semantic_notes.append(f"Fallback topic search for: {', '.join(variants)}")
            df = get_topic_library(topic)

        for note in semantic_notes:
            st.caption(note)

        if df.empty:
            st.warning("No entries found for that topic.")
            return

        # ✅ only display existing columns
        display_cols = [c for c in ["title", "url", "summary", "created_at"] if c in df.columns]
        df_display = df[display_cols].copy()

        # Sort newest first
        if "created_at" in df_display.columns:
            df_display = df_display.sort_values("created_at", ascending=False)

        st.dataframe(df_display)

        # Optional CSV download
        csv = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"{topic}_library.csv",
            mime="text/csv",
        )
