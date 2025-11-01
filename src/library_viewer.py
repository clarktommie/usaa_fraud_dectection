# src/library_viewer.py  (fixed)
import streamlit as st
import pandas as pd
from src.semantic_library import get_topic_library

def render_library_viewer():
    st.markdown("## 📚 Semantic Library Viewer")

    topic = st.text_input("Enter topic to view:", placeholder="e.g., phishing, aml, sanctions")

    if st.button("Load Library"):
        df = get_topic_library(topic)

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
