import streamlit as st

def sidebar_controls():
    """Render the Streamlit sidebar and return filter selections."""
    with st.sidebar:
        st.header("🔍 Search Filters")

        preset = st.selectbox(
            "Choose a preset query (optional):",
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

        year = st.text_input("Filter by Year (optional):", placeholder="e.g., 2024")
        keyword = st.text_input(
            "Filter by Keyword in Content (optional):",
            placeholder="e.g., cyber, bank, scam",
        )

        threshold = st.slider(
            "Match Threshold (similarity score):",
            min_value=0.0,
            max_value=1.0,
            value=0.60,
            step=0.05,
            help="Higher values = stricter match filtering",
        )

        top_k = st.selectbox(
            "Number of Results to Return:",
            options=[5, 10, 20, 50, 100, 500, 1000, 5000, 10000],
            index=2,
            help="Controls how many matching articles are displayed.",
        )

        st.markdown("---")
        st.caption("Adjust these filters to refine your semantic search results.")

    return preset, year, keyword, threshold, top_k
