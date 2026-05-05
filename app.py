import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

from preprocess import clean_data
from model import train_and_generate
from utils import add_noise, evaluate, postprocess_synthetic

st.set_page_config(page_title="Synthetic Healthcare AI", layout="wide")

# ------------------ LOGIN STATE ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False  

if not st.session_state.logged_in:
    st.title("Welcome to Synthetic Healthcare AI")
    st.write("Please log in to continue.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Log In"):
        if username == "admin" and password == "password":
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Invalid credentials. Try 'admin'/'password'.")

    st.stop()  # stops rest of app until login

# ------------------ STATE ------------------
if "step" not in st.session_state:
    st.session_state.step = "landing"

# ------------------ STYLES ------------------
st.markdown("""
<style>
.main {
    text-align: center;
}
.big-title {
    font-size: 48px;
    font-weight: 700;
}
.subtext {
    font-size: 18px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LANDING PAGE ------------------
if st.session_state.step == "landing":

    st.markdown('<div class="big-title">Synthetic Healthcare Data Generator</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="subtext">Generate privacy-preserving synthetic healthcare datasets that maintain statistical integrity while protecting patient confidentiality</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    if st.button("⬆ Upload Dataset"):
        st.session_state.step = "upload"
        st.rerun()

# ------------------ UPLOAD PAGE ------------------
elif st.session_state.step == "upload":

    # 🔥 BACK BUTTON
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("←"):
            st.session_state.step = "landing"
            st.rerun()

    st.header("Upload Healthcare Dataset")

    uploaded = st.file_uploader("Upload CSV file")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.df = df

        st.success("Dataset uploaded successfully")
        st.write(df.head())

        if st.button("Generate Synthetic Data"):
            st.session_state.step = "processing"
            st.rerun()

# ------------------ PROCESSING PAGE ------------------
elif st.session_state.step == "processing":

    # 🔥 BACK BUTTON
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("←"):
            st.session_state.step = "upload"
            st.rerun()

    st.markdown("## Generating Synthetic Data")

    progress = st.progress(0)
    status = st.empty()

    df = st.session_state.df

    df = df.sample(min(500, len(df)))  # speed

    df_clean = clean_data(df)

    # Step 1
    status.markdown("🟢 Analyzing dataset structure...")
    time.sleep(1)
    progress.progress(20)

    # Step 2
    status.markdown("🟢 Training generative model...")
    synthetic = train_and_generate(df_clean)
    progress.progress(60)

    # Step 3
    status.markdown("🔵 Applying privacy-preserving transformations...")
    synthetic = add_noise(synthetic, 0.1)
    synthetic = postprocess_synthetic(synthetic)
    progress.progress(85)

    # Step 4
    status.markdown("⚪ Validating synthetic data quality...")
    real_score, syn_score = evaluate(df_clean, synthetic)
    progress.progress(100)

    st.session_state.synthetic = synthetic
    st.session_state.real_score = real_score
    st.session_state.syn_score = syn_score
    st.session_state.df_clean = df_clean

    st.success("Done!")

    time.sleep(1)
    st.session_state.step = "results"
    st.rerun()

# ------------------ RESULTS PAGE ------------------
elif st.session_state.step == "results":

    # 🔥 BACK BUTTON
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("←"):
            st.session_state.step = "processing"
            st.rerun()

    st.title("Synthetic Data Output")

    synthetic = st.session_state.synthetic
    df_clean = st.session_state.df_clean
    real_score = st.session_state.real_score
    syn_score = st.session_state.syn_score

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Real Score", f"{real_score:.2f}")
    col2.metric("Synthetic Score", f"{syn_score:.2f}")
    col3.metric("Utility", f"{(syn_score/real_score)*100:.1f}%")

    # Data preview
    st.subheader("Synthetic Dataset")
    st.write(synthetic.head())

    # Chart
    if "age" in df_clean.columns:
        plt.figure()
        plt.hist(df_clean["age"], alpha=0.5, label="Real")
        plt.hist(synthetic["age"], alpha=0.5, label="Synthetic")
        plt.legend()
        st.pyplot(plt)

    # Download
    st.download_button(
        "Download Synthetic Data",
        synthetic.to_csv(index=False),
        "synthetic.csv"
    )

    if st.button("⬅ Start Again"):
        st.session_state.step = "landing"
        st.rerun()