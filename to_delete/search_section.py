import streamlit as st
from unused_scripts.embedding_search import run_search
from unused_scripts.pdf_generator import generate_pdf

def search_section(default_query, year, keyword, top_k, threshold):
    """Renders the Streamlit section for article semantic search and results."""
    st.markdown("### 🔎 Search and Explore Articles")

    # Query input
    query = st.text_input(
        "Enter your query (e.g., 'bank fraud', 'AML enforcement')",
        value=default_query,
        placeholder="Type a search term or choose a preset from the sidebar…",
    )

    # Action buttons
    col_btn1, col_btn2 = st.columns([1, 1], vertical_alignment="center")
    with col_btn1:
        search_btn = st.button("Search", type="primary")
    with col_btn2:
        clear_btn = st.button("Clear")

    if clear_btn:
        st.rerun()


    hits = []

    # Perform search
    if search_btn and query.strip():
        with st.spinner("Searching..."):
            hits = run_search(query.strip(), year, keyword, top_k, threshold)

        if not hits:
            st.warning("No results found. Try lowering the threshold or broadening your query.")
        else:
            st.subheader(f"Top {len(hits)} Results for: {query}")

            # PDF Download
            pdf_buffer = generate_pdf(hits, query)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_buffer,
                file_name=f"semantic_results_{query.replace(' ', '_')}.pdf",
                mime="application/pdf",
            )

            # Display results
            for i, r in enumerate(hits, 1):
                st.markdown(
                    f"**{i}. {r.get('title', '(no title)')}**  \n"
                    f"Match Score: `{r.get('similarity', 0):.3f}` · Date: `{r.get('date', 'N/A')}`  \n"
                    f"[Open Link]({r.get('url')})"
                )
                with st.expander("Preview"):
                    snippet = (r.get("content") or "").strip()
                    st.write(snippet[:1500] + ("…" if len(snippet) > 1500 else ""))

    return hits
