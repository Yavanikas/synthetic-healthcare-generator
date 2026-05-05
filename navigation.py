import streamlit as st

def back_button(target="app.py"):
    col1, col2 = st.columns([1, 10])

    with col1:
        if st.button("←", key=f"back_{target}"):
            st.switch_page(target)