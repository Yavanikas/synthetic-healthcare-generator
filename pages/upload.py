import streamlit as st
import pandas as pd
from navigation import back_button

back_button("app.py")

st.title("Upload Dataset")

uploaded = st.file_uploader("Upload CSV")

if uploaded:
    df = pd.read_csv(uploaded)
    st.session_state["data"] = df

    st.success("Dataset uploaded successfully!")
    st.write(df.head())

    if st.button("Proceed to Processing"):
        st.switch_page("pages/processing.py")   # match exact filename