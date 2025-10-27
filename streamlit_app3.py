# Importing necessary libraries
import os  # For interacting with the operating system
import io  # For I/O operations
import re  # For regular expressions
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For plotting
from collections import Counter  # For counting elements in an iterable
from dotenv import load_dotenv  # For loading environment variables from a .env file
from supabase import create_client  # For interacting with Supabase
from fastembed import TextEmbedding  # For text embedding
from wordcloud import WordCloud  # For generating word clouds
import plotly.express as px  # For data visualization
import streamlit as st  # For building interactive web apps with Streamlit
from reportlab.lib.pagesizes import letter  # For specifying page size in PDF generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer  # For creating PDF documents
from reportlab.lib.styles import getSampleStyleSheet  # For styling text in PDF documents

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

st.set_page_config(page_title="USAA Semantic Search", layout="wide")
st.title("Financial Compliance Insight System")

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

    # Replace slider with more flexible choices
    top_k = st.selectbox(
        "Results to return",
        [5, 10, 20, 50, 100, 500, 1000, 5000, 10000],
        index=2
    )


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

    story.append(Paragraph("USAA Semantic Search Project – UNC Charlotte", styles["Title"]))
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
# Insights generator
# -----------------
def analyze_articles(articles):
    if not articles:
        return None

    years = []
    for a in articles:
        if a.get("date"):
            found = re.findall(r"\b(20\d{2}|19\d{2})\b", str(a["date"]))
            if found:
                years.append(int(found[0]))

    year_counts = Counter(years)

    # Combine all text and extract keywords
    text = " ".join(a.get("content", "").lower() for a in articles)
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text)
    stopwords = set(["https", "federal", "reserve", "board", "press", "release", "bank", "financial"])
    filtered = [w for w in words if w not in stopwords]
    common_words = Counter(filtered).most_common(20)

    return {
        "year_counts": year_counts,
        "common_words": common_words,
        "total_articles": len(articles),
        "text": " ".join(filtered),
    }


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

        # PDF Download
        pdf_buffer = generate_pdf(hits, query)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer,
            file_name=f"semantic_results_{query.replace(' ', '_')}.pdf",
            mime="application/pdf",
        )

        # Display each article
        for i, r in enumerate(hits, 1):
            st.markdown(
                f"**{i}. {r.get('title','(no title)')}**  \n"
                f"Match Score: `{r.get('similarity',0):.3f}` · Date: `{r.get('date','N/A')}`  \n"
                f"[Open Link]({r.get('url')})"
            )
            with st.expander("Preview"):
                snippet = (r.get("content") or "").strip()
                st.write(snippet[:1500] + ("…" if len(snippet) > 1500 else ""))

        # -----------------
        # Dynamic insights
        # -----------------
        analysis = analyze_articles(hits)

        if analysis:
            st.divider()
            st.subheader("📊 Article Insights")

            st.metric("Total Articles Retrieved", analysis["total_articles"])

            # Year trend
            if analysis["year_counts"]:
                df_trend = pd.DataFrame(list(analysis["year_counts"].items()), columns=["Year", "Articles"])
                fig = px.bar(
                    df_trend,
                    x="Year",
                    y="Articles",
                    title="Number of Matching Articles by Year",
                    color="Articles",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Top words
            common_df = pd.DataFrame(analysis["common_words"], columns=["Word", "Count"])
            fig2 = px.bar(
                common_df,
                x="Word",
                y="Count",
                title="Top Common Terms in Retrieved Articles",
                color="Count",
                color_continuous_scale="greens",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Word cloud
            st.write("**Word Cloud (Context Overview):**")
            wc = WordCloud(width=800, height=400, background_color="white").generate(analysis["text"])
            fig_wc, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)

else:
    st.info("Enter a query (or choose a preset in the sidebar), then click **Search**.")
