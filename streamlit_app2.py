import os
import io
import numpy as np
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from fastembed import TextEmbedding
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -----------------
# Setup
# -----------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource(show_spinner=False)
def get_embedder():
    return TextEmbedding("BAAI/bge-small-en-v1.5")

embedder = get_embedder()

st.set_page_config(page_title="USAA Fraud Semantic Search", layout="wide")
st.title("🔎 USAA Fraud Semantic Search")

# -----------------
# Sidebar controls
# -----------------
with st.sidebar:
    st.header("Filters")
    preset = st.selectbox(
        "Preset query",
        [
            "— none —",
            "AML",
            "Bank Secrecy Act compliance",
            "money laundering",
            "check fraud",
            "synthetic identity fraud",
            "enforcement action",
            "consumer fraud complaints",
        ],
        index=0,
    )
    year = st.text_input("Year (optional)", value="")
    keyword = st.text_input("Keyword in content (optional)", value="")
    threshold = st.slider("Match threshold", 0.0, 1.0, 0.60, 0.05)
    top_k = st.slider("Results to return", 5, 50, 20, 5)

# If preset chosen and no query, use preset
default_query = "" if preset == "— none —" else preset
query = st.text_input(
    "Enter your query (e.g., 'bank fraud', 'AML enforcement')",
    value=default_query,
    placeholder="Type a query or pick a preset in the sidebar…",
)

col_btn1, col_btn2 = st.columns([1, 1], vertical_alignment="center")
with col_btn1:
    search_btn = st.button("Search", type="primary")
with col_btn2:
    clear_btn = st.button("Clear")

if clear_btn:
    st.experimental_rerun()

# -----------------
# Search logic
# -----------------
def run_search(q: str, year_filter: str, keyword_filter: str, top_k: int, threshold: float):
    q_vec = np.array(list(embedder.embed([q]))[0])

    resp = supabase.rpc(
        "match_press_releases",
        {
            "query_embedding": q_vec.tolist(),
            "match_threshold": float(threshold),
            "match_count": int(max(top_k, 50)),
        },
    ).execute()

    results = resp.data or []

    if year_filter:
        results = [r for r in results if r.get("date") and str(year_filter) in str(r["date"])]
    if keyword_filter:
        kw = keyword_filter.lower()
        results = [r for r in results if kw in (r.get("content") or "").lower()]

    results = sorted(results, key=lambda x: x.get("similarity", 0.0), reverse=True)[:top_k]
    return results

# -----------------
# PDF generator
# -----------------
def generate_pdf(results, query):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("USAA Fraud Detection Project – UNC Charlotte", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Semantic Search Results for: {query}", styles["Heading2"]))
    story.append(Spacer(1, 12))

    for i, r in enumerate(results, 1):
        title = r.get("title", "(No Title)")
        score = round(r.get("similarity", 0.0), 3)
        url = r.get("url", "N/A")
        date = r.get("date", "N/A")
        content = (r.get("content") or "")[:600] + "..."
        story.append(Paragraph(f"{i}. {title}", styles["Heading3"]))
        story.append(Paragraph(f"Score: {score} | Date: {date}", styles["Normal"]))
        story.append(Paragraph(f"URL: <a href='{url}'>{url}</a>", styles["Normal"]))
        story.append(Paragraph(content, styles["BodyText"]))
        story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer

# -----------------
# Execute search
# -----------------
if search_btn and query.strip():
    with st.spinner("Searching..."):
        hits = run_search(query.strip(), year, keyword, top_k, threshold)

    if not hits:
        st.warning("No results found. Try lowering the threshold or broadening your query.")
    else:
        st.subheader(f"Top {len(hits)} matches")

        pdf_buffer = generate_pdf(hits, query)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer,
            file_name=f"semantic_results_{query.replace(' ', '_')}.pdf",
            mime="application/pdf",
        )

        for i, r in enumerate(hits, 1):
            st.markdown(
                f"**{i}. {r.get('title','(no title)')}**  \n"
                f"Score: `{r.get('similarity',0):.3f}` · Date: `{r.get('date','N/A')}`  \n"
                f"[Open Link]({r.get('url')})"
            )
            with st.expander("Preview"):
                snippet = (r.get("content") or "").strip()
                st.write(snippet[:1500] + ("…" if len(snippet) > 1500 else ""))

else:
    st.info("Enter a query (or choose a preset in the sidebar), then click **Search**.")
