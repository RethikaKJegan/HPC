import streamlit as st
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="HAR Dashboard",
    layout="wide"
)

# -------------------------------
# TITLE
# -------------------------------
st.title("HAR Analysis Dashboard")
st.write("High-Performance HAR • PCA + XGBoost • Cloud CPU Inference Only")

# -------------------------------
# LOAD MODEL ONCE
# -------------------------------
@st.cache_resource
def load_model():
    with open("har_model_complete.pkl", "rb") as f:
        model_data = pickle.load(f)
    return model_data

model_data = load_model()

xgb = model_data["xgb"]
X_test = model_data["X_test_pca"]
y_test = model_data["y_test"]
labels = model_data["activity_labels"]

# -------------------------------
# CPU INFERENCE BUTTON
# -------------------------------
st.subheader("CPU Inference (Cloud)")

if st.button("Run CPU Inference"):
    start = time.time()
    preds = xgb.predict(X_test)
    end = time.time()

    st.success("Inference Complete!")

    # -------------------------------
    # BASIC METRICS
    # -------------------------------
    st.write(f"**Inference Time:** {end - start:.5f} seconds")
    st.write(f"**Test Accuracy:** {model_data['test_accuracy']*100:.2f}%")

    # -------------------------------
    # PER-CLASS ACCURACY
    # -------------------------------
    st.subheader("Per-Class Accuracy")

    per_class_acc = {}
    for i, lbl in labels.items():
        mask = y_test == i
        acc = np.mean(preds[mask] == y_test[mask])
        per_class_acc[lbl] = acc
        st.write(f"**{lbl}:** {acc*100:.2f}%")

    # ==================================================================
    # 1) RUNTIME BREAKDOWN (CLEAN & COMPACT)
    # ==================================================================
    st.subheader("Runtime Breakdown (CPU Cloud)")

    model_load_time = model_data.get("model_load_time", 0.18)  # default if not saved
    inference_time = end - start
    total_time = model_load_time + inference_time

    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(
        ["Model Load", "Inference", "Total"],
        [model_load_time, inference_time, total_time],
        color=["#4A90E2", "#E74C3C", "#2ECC71"],
        alpha=0.85
    )

    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f"{height:.4f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_ylabel("Time (seconds)")
    ax.set_title("Runtime Breakdown - CPU", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    st.pyplot(fig)

    # ==================================================================
    # 2) PCA FEATURE SCATTER PLOT (SMALL, CLEAN)
    # ==================================================================
    st.subheader("PCA Feature Scatter (Test Set)")

    X_pca = model_data["X_test_pca"][:, :2]   # Use first 2 PCA comps
    unique_labels = np.unique(y_test)

    colors = [
        "#1f77b4", "#2ca02c", "#9467bd",
        "#d62728", "#bcbd22", "#17becf"
    ]

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, lbl in enumerate(unique_labels):
        mask = y_test == lbl
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=10,
            alpha=0.6,
            color=colors[i],
            label=labels[lbl]
        )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("PCA Feature Scatter Plot (Test Set)", fontsize=12, fontweight="bold")
    ax.legend(markerscale=2, fontsize=8)

    st.pyplot(fig)

# -------------------------------
# END OF FILE
# -------------------------------
