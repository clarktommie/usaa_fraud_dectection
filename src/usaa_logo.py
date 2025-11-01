import streamlit as st

def display_usaa_logo():
    """Display a fixed USAA logo in the top-left corner with white background."""
    html_code = """
        <div style="
            position: fixed;
            top: 20px;
            left: 30px;
            background-color: white;
            padding: 6px 14px;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            z-index: 9999;
        ">
            <img src="https://pmodncflpdtjfztlrsik.supabase.co/storage/v1/object/public/banners/usaaLogo.png"
                 alt="USAA Logo"
                 style="height:80px; width:auto;">
        </div>
    """

    # Give Streamlit a bigger render shell so the logo isn’t cropped
    st.components.v1.html(html_code, height=120, width=250)
