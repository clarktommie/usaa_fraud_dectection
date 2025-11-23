import streamlit as st

FOCUS_OPTIONS = [
    ("— none —", ""),
    ("Synthetic identity fraud", "What patterns exist in synthetic identity fraud?"),
    ("Check fraud & paper-based scams", "How are banks mitigating check fraud and counterfeit schemes?"),
    ("Wire & ACH payment fraud", "What risks are emerging in wire transfer and ACH payment fraud?"),
    ("Money mule networks", "How are regulators combating money mule schemes tied to fraud?"),
    ("Elder financial exploitation", "How are institutions protecting seniors from financial scams?"),
    ("Cyber & ransomware threats", "How are banks responding to cyber fraud and ransomware attacks?"),
    ("AML & sanctions compliance", "How are institutions improving AML and sanctions controls?"),
]


def sidebar_controls():
    """
    Render the Streamlit sidebar for the fraud research app
    and return the selected focus option plus insight preference.
    """
    with st.sidebar:
        st.header("🎯 Fraud Research Controls")

        labels = [label for label, _ in FOCUS_OPTIONS]
        default_index = 0
        focus_label = st.selectbox(
            "Choose a fraud focus type:",
            labels,
            index=default_index,
            help="Select the fraud theme you want to explore.",
        )
        preset_query = dict(FOCUS_OPTIONS).get(focus_label, "")

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
        st.caption("Pick a fraud focus to guide semantic storytelling.")

    return focus_label, preset_query, insight_choice
