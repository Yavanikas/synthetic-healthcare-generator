import streamlit as st
import time
from preprocess import clean_data
from model import train_and_generate
from utils import add_noise
from navigation import back_button

back_button("pages/upload.py")
st.title("Generating Synthetic Data")

if "data" not in st.session_state:
    st.warning("Please upload data first.")
else:
    df = st.session_state["data"]

    progress = st.progress(0)
    status = st.empty()

    # Step 1
    status.write("Analyzing dataset...")
    time.sleep(1)
    progress.progress(25)

    # Step 2
    status.write("Training model...")
    df_clean = clean_data(df)
    synthetic = train_and_generate(df_clean)
    progress.progress(60)

    # Step 3
    status.write("Applying privacy...")
    synthetic = add_noise(synthetic, 0.1)
    progress.progress(85)

    # Save synthetic
    st.session_state["synthetic"] = synthetic
    st.session_state["clean"] = df_clean

    # Step 4
    status.write("Finalizing...")
    time.sleep(1)
    progress.progress(100)

    st.success("Done!")

    if st.button("Go to Results"):
        st.switch_page("pages/result.py")