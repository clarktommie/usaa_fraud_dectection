import streamlit as st

def sidebar_controls():
    """
    Render the Streamlit sidebar for the fraud research app
    and return the selected filters and focus option.
    """
    with st.sidebar:
        st.header("🎯 Fraud Research Controls")

        # --- Common preset queries ---
        preset = st.selectbox(
            "Choose a preset topic:",
            [
                "— none —",
                "How are banks improving AML compliance?",
                "How are regulators addressing money laundering trends?",
                "What fraud risks are emerging from digital payments?",
                "How are institutions responding to BSA/AML enforcement actions?",
                "What patterns exist in synthetic identity fraud?",
                "How are consumers reporting financial scams or fraud?",
                "What role does AI play in detecting financial crimes?",
            ],
            index=0,
        )

        # --- Optional year or keyword filters ---
        st.subheader("📅 Optional Filters")
        year = st.text_input("Year (optional):", placeholder="e.g., 2024")
        keyword = st.text_input("Keyword (optional):", placeholder="e.g., FinCEN, wire transfer, scam")

        # --- Quick Insight Prompts ---
        st.markdown("### 💡 Quick Insights")
        insight_choice = st.radio(
            "Focus the summary on:",
            [
                "Trends and patterns over time",
                "Emerging risk areas",
                "Policy and regulatory tone",
                "Consumer or institutional impact",
            ],
            index=0,
            help="Select the analytical focus for the OpenAI summary.",
        )

        st.markdown("---")
        st.caption("Use presets or filters to guide your semantic storytelling search.")

    return preset, year, keyword, insight_choice
