import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from utils import evaluate
from navigation import back_button

back_button("pages/processing.py")

st.title("Results")

if "synthetic" not in st.session_state:
    st.warning("Run processing first.")
else:
    synthetic = st.session_state["synthetic"]
    df_clean = st.session_state["clean"]

    st.subheader("Synthetic Data")
    st.write(synthetic.head())

    real_score, syn_score = evaluate(df_clean, synthetic)

    col1, col2, col3 = st.columns(3)

    col1.metric("Real Score", f"{real_score:.2f}")
    col2.metric("Synthetic Score", f"{syn_score:.2f}")
    col3.metric("Utility", f"{(syn_score/real_score)*100:.1f}%")

    st.subheader("📊 Distribution (Quick Insights)")

    # ===============================
    # 🔥 CARD-STYLE MINI GRAPHS
    # ===============================
    numeric_cols = [
        col for col in df_clean.select_dtypes(include=['int64', 'float64']).columns
        if col in synthetic.columns
    ]

    if numeric_cols:
        selected_cols = numeric_cols[:3]

        cols = st.columns(len(selected_cols))

        for i, feature in enumerate(selected_cols):
            with cols[i]:
                st.markdown(f"**{feature.capitalize()}**")

                try:
                    real_binned = pd.cut(df_clean[feature], bins=5)
                    syn_binned = pd.cut(synthetic[feature], bins=5)

                    real_counts = real_binned.value_counts().sort_index()
                    syn_counts = syn_binned.value_counts().sort_index()

                    fig, ax = plt.subplots()
                    x = range(len(real_counts))

                    ax.bar(x, real_counts, width=0.4, label="Real", align='center')
                    ax.bar(x, syn_counts, width=0.4, label="Synthetic", align='edge')

                    ax.set_xticks(x)
                    ax.set_xticklabels(
                        [str(i) for i in real_counts.index],
                        rotation=30,
                        fontsize=8
                    )

                    ax.legend()
                    st.pyplot(fig)
                    plt.clf()

                except Exception:
                    st.warning(f"Could not plot {feature}")

        # ===============================
        # 🧠 INSIGHTS (FIXED LOCATION)
        # ===============================
        st.subheader("Insights")

        for col in selected_cols:
            try:
                real_mean = df_clean[col].mean()
                syn_mean = synthetic[col].mean()

                if real_mean != 0:
                    diff = abs(real_mean - syn_mean) / abs(real_mean)

                    if diff < 0.1:
                        st.success(f"{col}: High similarity between real and synthetic data")
                    else:
                        st.warning(f"{col}: Noticeable deviation detected")
                else:
                    st.info(f"{col}: Cannot evaluate (mean is zero)")

            except Exception:
                st.warning(f"{col}: Could not evaluate")

    else:
        st.warning("No numeric columns available.")

    # ===============================
    # DOWNLOAD
    # ===============================
    st.download_button(
        "Download Synthetic Data",
        synthetic.to_csv(index=False),
        "synthetic.csv"
    )