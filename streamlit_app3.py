# streamlit_app3.py
"""
USAA Fraud Research Dashboard
---------------------------------
Analyzes financial compliance and fraud-related content from Federal Reserve
and CFPB press releases. Combines semantic storytelling, domain tagging,
and OpenAI reasoning to provide insight into fraud, compliance, and risk trends.
"""

import os
import json
from datetime import datetime
from io import StringIO
import pandas as pd
import pydeck as pdk
import streamlit as st
import altair as alt
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

# -------------------------------
# Imports from your project
# -------------------------------
from src.sidebar_controls import sidebar_controls
from src.library_viewer import render_library_viewer
from src.usaa_logo import display_usaa_logo
from src.topic_dashboard import render_topic_dashboard
from src.ai.fraud_insights import generate_fraud_insights
from src.ai.article_preprocessing import prepare_articles_for_ai
from src.cfpb_loader import fetch_complaints
from src.topic_visuals import generate_topic_visual_data
from src.topic_keyword_utils import (
    infer_focus_phrase,
    build_state_heatmap_data,
)
from src.article_index import ArticleIndex


def format_apa_citation(row: dict) -> str:
    """Build a lightweight APA-style citation using available metadata."""
    source = row.get("source") or "Press Release"
    title = row.get("title") or "Untitled article"
    date_value = row.get("date") or row.get("scraped_at") or row.get("updated_at")
    formatted_date = "n.d."
    if date_value:
        try:
            parsed = pd.to_datetime(date_value)
            if isinstance(parsed, pd.Timestamp) and not pd.isna(parsed):
                formatted_date = parsed.strftime("%Y, %B %-d")
        except Exception:
            pass
    citation = f"{source}. ({formatted_date}). {title}."
    url = row.get("url")
    if url:
        citation += f" {url}"
    return citation


def build_apa_citations(df: pd.DataFrame, limit: int = 3) -> list[str]:
    subset = df.copy()
    if "date" in subset.columns:
        subset = subset.sort_values("date", ascending=False)
    rows = subset.head(limit).to_dict(orient="records")
    return [format_apa_citation(r) for r in rows if r]


def build_fallback_trends(
    ts_df: pd.DataFrame,
    sources_df: pd.DataFrame,
    keywords_df: pd.DataFrame,
    focus_phrase: str,
) -> list[dict]:
    """Derive trend statements from quantitative data when AI narratives are unavailable."""
    trends: list[dict] = []

    if not ts_df.empty and len(ts_df) >= 2:
        cur = ts_df.iloc[-1]
        prev = ts_df.iloc[-2]
        delta = cur["article_count"] - prev["article_count"]
        direction = "increased" if delta > 0 else "declined" if delta < 0 else "held steady"
        summary = (
            f"Monthly coverage {direction} to {cur['article_count']} articles "
            f"in {cur['timestamp']:%b %Y} compared with {prev['article_count']} previously."
        )
        trends.append(
            {
                "headline": "Momentum across releases",
                "summary": summary,
            }
        )

    if not sources_df.empty:
        top_sources = sources_df.head(2)
        src_desc = "; ".join(
            f"{row['source']} ({row['article_count']} articles)"
            for _, row in top_sources.iterrows()
        )
        trends.append(
            {
                "headline": "Dominant sources",
                "summary": f"{src_desc} drive the {focus_phrase} topic coverage.",
            }
        )

    if not keywords_df.empty:
        top_terms = ", ".join(keywords_df.head(5)["term"].tolist())
        trends.append(
            {
                "headline": "Top recurring terms",
                "summary": f"Most frequent terms in matched articles: {top_terms}.",
            }
        )

    return trends[:3]


def build_stakeholder_brief(
    stats: dict,
    sources_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    focus_phrase: str,
) -> list[str]:
    """Craft bullet-friendly stakeholder talking points from visual data."""
    lines: list[str] = []
    coverage = "n/a"
    if stats.get("earliest_date") and stats.get("latest_date"):
        coverage = f"{stats['earliest_date']:%b %Y} → {stats['latest_date']:%b %Y}"

    lines.append(
        f"- **Focus**: {focus_phrase} surfaces {stats.get('article_count', 0)} high-similarity releases "
        f"(avg score {stats.get('avg_similarity', 0):.2f}) across {coverage}."
    )

    if not ts_df.empty and len(ts_df) >= 2:
        cur = ts_df.iloc[-1]
        prev = ts_df.iloc[-2]
        delta = cur["article_count"] - prev["article_count"]
        pct = None
        if prev["article_count"]:
            pct = (delta / prev["article_count"]) * 100
        trend_icon = "📈" if delta > 0 else "📉" if delta < 0 else "➖"
        change_text = f"{delta:+} articles"
        if pct is not None:
            change_text += f" ({pct:+.0f}%)"
        lines.append(
            f"- {trend_icon} **Momentum**: Latest month recorded {cur['article_count']} articles "
            f"({change_text} vs. previous month)."
        )

    if not sources_df.empty:
        top_source = sources_df.iloc[0]
        lines.append(
            f"- **Top source**: {top_source['source']} contributed {top_source['article_count']} articles "
            f"(avg similarity {top_source['avg_similarity']:.2f})."
        )

    return lines


def generate_ai_topic_trends(
    articles_df: pd.DataFrame, focus_phrase: str, max_articles: int = 8
) -> dict:
    """Use OpenAI to derive keywords/trends strictly from provided article snippets."""
    if openai_client is None or not OPENAI_API_KEY:
        return {"keywords": [], "trends": [], "error": "OPENAI_API_KEY not configured."}

    if articles_df.empty:
        return {"keywords": [], "trends": []}

    subset = articles_df.head(max_articles).copy()
    payload = []
    for row in subset.to_dict(orient="records"):
        snippet = (
            row.get("summary")
            or row.get("content")
            or row.get("snippet")
            or ""
        )
        snippet = str(snippet).strip()
        if not snippet:
            continue
        payload.append(
            {
                "title": row.get("title") or "Untitled release",
                "source": row.get("source") or "Unknown",
                "date": str(row.get("date") or ""),
                "snippet": snippet[:900],
            }
        )

    if not payload:
        return {"keywords": [], "trends": [], "error": "No article excerpts available."}

    articles_json = json.dumps(payload, ensure_ascii=False)
    user_prompt = f"""
Focus phrase: {focus_phrase}

Use ONLY the content in the JSON array below to determine important keywords and trend statements.
If data is insufficient, return empty arrays.

ARTICLES_JSON:
{articles_json}

Return strictly valid JSON with this schema:
{{
  "keywords": [
    {{"term": "short keyword from evidence", "supporting_titles": ["title1", "title2"]}}
  ],
  "trends": [
    {{
      "headline": "concise trend title grounded in evidence",
      "summary": "one sentence referencing exact article facts",
      "supporting_titles": ["title1", "title2"]
    }}
  ]
}}
Provide up to 5 keywords and up to 3 trends, but only include items supported by the excerpts.
Do not invent facts outside the provided snippets.
"""

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_TRENDS_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a risk intelligence analyst. Only use supplied article snippets. "
                        "Never infer beyond provided evidence and always return compact JSON."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=700,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        keywords = (data.get("keywords") or [])[:5]
        trends = (data.get("trends") or [])[:3]
        return {"keywords": keywords, "trends": trends}
    except Exception as exc:
        return {"keywords": [], "trends": [], "error": str(exc)}

# -----------------
# Setup
# -----------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TRENDS_MODEL = os.getenv("OPENAI_TRENDS_MODEL", "gpt-4o-mini")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -----------------
# Initialization
# -----------------
st.set_page_config(page_title="USAA Semantic Search", layout="wide")
display_usaa_logo()
st.title("Financial Compliance Insight System")

focus_label, preset_query, insight_choice = sidebar_controls()
default_query = preset_query or ""
st.info("Enter a query (or choose a fraud focus in the sidebar), then click **Search**.")

# -----------------
# Sidebar: Topic Dashboard Button
# -----------------
st.sidebar.markdown("### Analysis Modules")
if st.sidebar.button("📘 Topic Dashboard"):
    st.session_state["show_topic_dashboard"] = True
if st.session_state.get("show_topic_dashboard"):
    render_topic_dashboard()

# -----------------
# Fetch Articles (Cached Index)
# -----------------
refresh_cache = st.sidebar.button("♻️ Refresh Article Cache", help="Forces the local embedding cache to rebuild from Supabase.")
try:
    index = ArticleIndex.load(force_refresh=refresh_cache)
    all_articles = index.to_dataframe(include_embeddings=True)
    built_time = datetime.fromtimestamp(index.built_at)
    st.success(f"✅ Loaded {len(all_articles)} cached articles (updated {built_time.strftime('%Y-%m-%d %H:%M')}).")
except Exception as exc:
    st.error(f"⚠️ Unable to load article cache: {exc}")
    from src.data_loader import fetch_articles  # lazy import fallback

    all_articles = fetch_articles()
    if all_articles.empty:
        st.stop()
    else:
        st.warning("Using live Supabase data (no local cache).")

# -----------------
# Load Complaints
# -----------------
complaints_df = fetch_complaints()
if complaints_df.empty:
    st.warning("⚠️ No complaints found in Supabase.")
else:
    st.success(f"✅ Loaded {len(complaints_df)} classified CFPB complaints.")

# -----------------
# SEMANTIC + INSIGHT ENGINE
# -----------------
st.markdown("### ⚙️ Semantic Exploration & Insight Engine")

query_sentence = st.text_input(
    "Enter a question or sentence to explore:",
    value=default_query,
    placeholder="e.g., How are banks addressing AML and third-party risks?"
)
run_semantic = st.button("Run Semantic Search")

if run_semantic and query_sentence.strip():
    df = all_articles.copy()

    if df.empty:
        st.warning("No articles match the selected filters.")
    else:
        with st.spinner(f"Analyzing '{query_sentence}' across articles and complaints..."):

            # Step 1: Prepare articles for AI
            clean_articles = prepare_articles_for_ai(df.to_dict(orient="records"))

            # Step 2: Generate AI insights
            result = generate_fraud_insights(
                articles=clean_articles,
                user_query=query_sentence,
                date_range=("N/A", "N/A")
            )

        # -----------------------------
        # DISPLAY UNIFIED AI SUMMARY
        # -----------------------------
        st.divider()
        st.markdown("### 🧠 AI-Generated Unified Fraud Insight")
        st.write(result.get("summary_text", "No insight generated."))
        citations = build_apa_citations(df)
        if citations:
            st.markdown("**APA References**")
            for cite in citations:
                st.markdown(f"- {cite}")
        st.divider()

        # -----------------------------
        # TOPIC-FOCUSED VISUAL STUDIO
        # -----------------------------
        st.divider()
        st.markdown("### 🔭 Topic Visual Studio")
        focus_hint = focus_label if focus_label != "— none —" else ""
        focus_phrase = infer_focus_phrase(query_sentence, focus_hint, [], df)

        if not focus_phrase:
            st.info("Select a fraud focus or mention a phrase in your query to unlock visuals.")
        else:
            topic_viz = generate_topic_visual_data(df, focus_phrase)
            articles_subset = topic_viz["articles"]
            if articles_subset.empty:
                st.info("Not enough embedding coverage to render visuals for this topic.")
            else:
                st.caption(f"Visuals tailored to **{focus_phrase}** ({len(articles_subset)} high-similarity articles).")

                ts = topic_viz["time_series"]
                sources = topic_viz["source_breakdown"]
                keyword_scores = topic_viz["keyword_scores"]
                projection = topic_viz["embedding_projection"]
                stats = topic_viz.get("stats", {})
                ai_trends = generate_ai_topic_trends(articles_subset, focus_phrase)
                ai_keywords_raw = []
                if isinstance(ai_trends, dict):
                    ai_keywords_raw = ai_trends.get("keywords") or []
                fallback_keywords: list[dict] = []
                if ai_keywords_raw:
                    keyword_chart_df = pd.DataFrame(
                        [
                            {
                                "term": entry.get("term"),
                                "score": max(len(entry.get("supporting_titles") or []), 1),
                            }
                            for entry in ai_keywords_raw
                            if entry.get("term")
                        ]
                    )
                    keywords_source = "ai"
                else:
                    keyword_chart_df = keyword_scores.copy()

                if not keyword_chart_df.empty:
                    fallback_keywords = [
                        {"term": term, "supporting_titles": []}
                        for term in keyword_chart_df.head(5)["term"].tolist()
                        if term
                    ]

                fallback_trend_items = build_fallback_trends(ts, sources, keyword_chart_df, focus_phrase)

                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Articles in focus",
                    stats.get("article_count", 0),
                    help="Number of articles most similar to the current query.",
                )
                m2.metric(
                    "Avg similarity",
                    f"{stats.get('avg_similarity', 0):.2f}",
                    help="Normalized cosine similarity (0-1) against the query.",
                )
                date_range = "n/a"
                if stats.get("earliest_date") and stats.get("latest_date"):
                    date_range = f"{stats['earliest_date']:%b %Y} → {stats['latest_date']:%b %Y}"
                m3.metric("Coverage window", date_range)

                col_ts, col_src = st.columns(2)
                if not ts.empty:
                    ts_copy = ts.copy()
                    ts_copy["timestamp"] = pd.to_datetime(ts_copy["timestamp"])
                    base = alt.Chart(ts_copy).encode(x=alt.X("timestamp:T", title="Month"))
                    similarity_line = base.mark_line(point=True, color="#2F80ED").encode(
                        y=alt.Y("avg_similarity:Q", title="Avg similarity"),
                        tooltip=[
                            alt.Tooltip("timestamp:T", title="Month"),
                            alt.Tooltip("avg_similarity:Q", format=".2f"),
                            alt.Tooltip("article_count:Q", title="Articles"),
                        ],
                    )
                    volume_bars = base.mark_bar(color="#7FDBFF", opacity=0.3).encode(
                        y=alt.Y("article_count:Q", title="Article count")
                    )
                    col_ts.altair_chart(
                        alt.layer(volume_bars, similarity_line).resolve_scale(y="independent"),
                        use_container_width=True,
                    )
                    col_ts.caption("Monthly distribution of similarity (line) against article volume (bars).")
                else:
                    col_ts.info("No dated articles to plot the timeline.")

                if not sources.empty:
                    source_chart = (
                        alt.Chart(sources)
                        .mark_bar()
                        .encode(
                            x=alt.X("article_count:Q", title="Articles"),
                            y=alt.Y("source:N", sort="-x", title="Source"),
                            color=alt.Color(
                                "avg_similarity:Q",
                                title="Avg similarity",
                                scale=alt.Scale(scheme="blues"),
                            ),
                            tooltip=[
                                alt.Tooltip("source:N", title="Source"),
                                alt.Tooltip("article_count:Q", title="Articles"),
                                alt.Tooltip("avg_similarity:Q", format=".2f", title="Avg similarity"),
                            ],
                        )
                        .properties(height=320)
                    )
                    col_src.altair_chart(source_chart, use_container_width=True)
                    col_src.caption("Ranking of top domain sources powering this topic.")
                else:
                    col_src.info("Source information unavailable for this topic.")

                if not keyword_chart_df.empty:
                    keyword_view = keyword_chart_df.head(15).copy()
                    keyword_chart = (
                        alt.Chart(keyword_view)
                        .mark_bar()
                        .encode(
                            x=alt.X("score:Q", title="Keyword weight"),
                            y=alt.Y("term:N", sort="-x", title="Keyword"),
                            color=alt.Color("score:Q", scale=alt.Scale(scheme="teals"), legend=None),
                            tooltip=[
                                alt.Tooltip("term:N", title="Keyword"),
                                alt.Tooltip("score:Q", format=".3f", title="Weight"),
                            ],
                        )
                        .properties(height=380)
                    )
                    st.altair_chart(keyword_chart, use_container_width=True)
                    chips = " · ".join(f"`{term}`" for term in keyword_view["term"].head(6).tolist())
                    if chips:
                        st.caption(f"Focus keywords: {chips}")
                    if ai_keywords_raw:
                        st.caption("Keywords derived from evidence-backed LLM summaries.")
                    else:
                        st.caption("Keywords generated from TF-IDF weighting within matched articles.")
                else:
                    st.info("Not enough content to extract meaningful keywords.")

                if not projection.empty:
                    scatter_data = projection.copy()
                    scatter_data["date"] = pd.to_datetime(scatter_data.get("date"), errors="coerce")
                    scatter_data["dim_3_size"] = scatter_data["dim_3"].abs() * 80 + 20
                    scatter_chart = (
                        alt.Chart(scatter_data)
                        .mark_circle(opacity=0.8)
                        .encode(
                            x=alt.X("dim_1:Q", title="Latent dimension 1"),
                            y=alt.Y("dim_2:Q", title="Latent dimension 2"),
                            size=alt.Size("dim_3_size:Q", legend=None),
                            color=alt.Color("similarity:Q", scale=alt.Scale(scheme="viridis"), title="Similarity"),
                            tooltip=[
                                alt.Tooltip("title:N", title="Article"),
                                alt.Tooltip("similarity:Q", format=".3f", title="Similarity"),
                                alt.Tooltip("date:T", title="Date", format="%b %Y"),
                            ],
                        )
                        .properties(height=420)
                        .interactive()
                    )
                    st.altair_chart(scatter_chart, use_container_width=True)
                    st.caption("Embedding neighborhood — each point is an article colored by semantic proximity.")
                else:
                    st.info("Cannot project embeddings for this topic.")

                state_map = build_state_heatmap_data(complaints_df, focus_phrase)
                if not state_map.empty:
                    st.subheader("Complaint Hotspots")
                    st.pydeck_chart(
                        pdk.Deck(
                            layers=[
                                pdk.Layer(
                                    "ColumnLayer",
                                    data=state_map,
                                    get_position="[longitude, latitude]",
                                    get_elevation="mentions * 1000",
                                    elevation_scale=0.5,
                                    radius=30000,
                                    get_fill_color="[255, 99, 71, 160]",
                                    pickable=True,
                                )
                            ],
                            initial_view_state=pdk.ViewState(
                                latitude=state_map["latitude"].mean(),
                                longitude=state_map["longitude"].mean(),
                                zoom=3.2,
                                pitch=20,
                            ),
                            tooltip={"text": "{state}: {mentions} complaints"},
                        )
                    )
                    with st.expander("State-level detail"):
                        st.dataframe(state_map, hide_index=True)
                else:
                    st.info("No complaint hotspots found for this focus phrase.")

                top_articles = (
                    articles_subset[["date", "title", "source", "similarity", "url"]]
                    .head(15)
                    .copy()
                )
                top_articles["similarity"] = top_articles["similarity"].round(3)
                st.subheader("Top Articles Driving This Topic")
                st.dataframe(
                    top_articles.rename(columns={"similarity": "semantic_score"}),
                    use_container_width=True,
                    hide_index=True,
                )

                ai_keywords = ai_keywords_raw[:5]
                ai_trend_items = []
                ai_error = None
                if isinstance(ai_trends, dict):
                    ai_trend_items = (ai_trends.get("trends") or [])[:3]
                    ai_error = ai_trends.get("error")
                keyword_display_items = ai_keywords if ai_keywords else fallback_keywords
                trend_display_items = ai_trend_items if ai_trend_items else fallback_trend_items

                st.subheader("Keywords & Trend Highlights")
                if ai_error:
                    st.warning(f"Trend summarization unavailable: {ai_error}")
                if keyword_display_items:
                    st.markdown(
                        "**Keywords:** "
                        + " · ".join(
                            f"`{entry.get('term')}`"
                            for entry in keyword_display_items
                            if entry.get("term")
                        )
                    )
                    with st.expander("Keyword evidence"):
                        for entry in keyword_display_items:
                            titles = entry.get("supporting_titles") or []
                            title_list = ", ".join(titles) if titles else "No article references provided."
                            st.write(f"- **{entry.get('term')}** — sources: {title_list}")
                    if ai_keywords and len(ai_keywords) < 5:
                        st.caption("Fewer than five keywords due to limited supporting articles.")
                    elif not ai_keywords:
                        st.caption("Keywords derived from TF-IDF coverage because AI summaries were unavailable.")
                else:
                    st.info("No keywords could be derived from the selected articles.")

                if trend_display_items:
                    source_note_added = False
                    for idx, entry in enumerate(trend_display_items, 1):
                        st.markdown(f"**Trend {idx}: {entry.get('headline','Trend')}**")
                        st.write(entry.get("summary") or "No summary available.")
                        titles = entry.get("supporting_titles") or []
                        if titles:
                            st.caption("Sources: " + ", ".join(titles))
                        st.divider()
                    if ai_trend_items and len(ai_trend_items) < 3:
                        st.caption("Fewer than three trend statements were supported by the evidence.")
                    elif not ai_trend_items:
                        st.caption("Trend statements estimated from topic metrics (no AI summary).")
                else:
                    st.info("No trend statements could be generated.")

                st.subheader("Stakeholder Briefing & Exports")
                briefing_lines = build_stakeholder_brief(stats, sources, ts, focus_phrase)
                if briefing_lines:
                    for line in briefing_lines:
                        st.markdown(line)
                else:
                    st.info("Not enough signal to generate stakeholder-ready talking points.")

                summary_md = "\n".join(
                    [
                        f"# Stakeholder Brief — {focus_phrase}",
                        f"*Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*",
                        "",
                        *[line.replace("- ", "* ", 1) for line in briefing_lines],
                        "",
                        "## Focus Keywords",
                        *[
                            f"* `{entry.get('term')}` — sources: {', '.join(entry.get('supporting_titles') or [])}"
                            for entry in keyword_display_items
                            if entry.get("term")
                        ],
                        "",
                        "## Trend Highlights",
                        *[
                            f"* **{entry.get('headline','Trend')}** — {entry.get('summary', '')}"
                            for entry in trend_display_items
                        ],
                        "",
                        "Data source: USAA Fraud Research Dashboard",
                    ]
                )
                csv_content = ""
                if not articles_subset.empty:
                    csv_buffer = StringIO()
                    articles_subset.to_csv(csv_buffer, index=False)
                    csv_content = csv_buffer.getvalue()

                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button(
                        "⬇️ Download Topic Articles (CSV)",
                        data=csv_content or "title,source,similarity\n",
                        file_name=f"{focus_phrase.lower().replace(' ', '_')}_articles.csv",
                        mime="text/csv",
                        disabled=not csv_content,
                    )
                with dl2:
                    st.download_button(
                        "⬇️ Download Stakeholder Brief (Markdown)",
                        data=summary_md,
                        file_name=f"{focus_phrase.lower().replace(' ', '_')}_brief.md",
                        mime="text/markdown",
                        disabled=not briefing_lines,
                    )
else:
    st.info("Enter a query and click **Run Semantic Search** to begin.")

# -----------------
# Library Viewer Integration
# -----------------
if st.sidebar.button("📚 Open Library Viewer"):
    st.session_state["view_library"] = True
if st.session_state.get("view_library"):
    render_library_viewer()
