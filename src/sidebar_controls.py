import streamlit as st

FOCUS_OPTIONS = [
    ("Synthetic ID", "What trends define synthetic identity fraud in banking?"),
    ("Check fraud", "What is the current state of check fraud in banking?"),
    ("Wire/ACH", "How are wire and ACH fraud risks evolving?"),
    ("Money mules", "How are money mule networks impacting banks?"),
    ("Elder abuse", "How are banks protecting seniors from financial scams?"),
    ("Cyber/ransom", "How are banks addressing cyber fraud and ransomware?"),
    ("AML/sanctions", "How are institutions strengthening AML and sanctions controls?"),
]


def sidebar_controls():
    """Render quick-pick fraud topics as buttons above the search bar."""
    st.markdown("#### Choose a fraud topic")

    focus_label = "None"
    preset_query = ""
    run_now = False

    cols = st.columns(4)
    for idx, (label, query) in enumerate(FOCUS_OPTIONS):
        with cols[idx % 4]:
            if st.button(label, key=f"topic_{idx}"):
                focus_label = label
                preset_query = query
                run_now = True

    st.caption("Click a topic to prefill and run search, or type your own question below.")
    return focus_label, preset_query, run_now
