import streamlit as st
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="HAR Dashboard", layout="wide")

st.title("HAR Analysis Dashboard")
st.write("High-Performance HAR • PCA + XGBoost • Cloud CPU Inference Only")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    with open("har_model_complete.pkl", "rb") as f:
        return pickle.load(f)

model_data = load_model()
xgb = model_data["xgb"]
X_test = model_data["X_test_pca"]
y_test = model_data["y_test"]
labels = model_data["activity_labels"]

# -------------------------------
# INFERENCE BUTTON
# -------------------------------
st.subheader("CPU Inference (Cloud)")

if st.button("Run CPU Inference"):
    start = time.time()
    preds = xgb.predict(X_test)
    end = time.time()

    st.success("Inference Complete!")

    # BASIC METRICS
    st.write(f"**Inference Time:** {end - start:.5f} seconds")
    st.write(f"**Test Accuracy:** {model_data['test_accuracy']*100:.2f}%")

    # -------------------------------
    # PER-CLASS ACCURACY (TEXT)
    # -------------------------------
    st.subheader("Per-Class Accuracy")

    per_class_acc = {}
    for i, lbl in labels.items():
        mask = y_test == i
        acc = np.mean(preds[mask] == y_test[mask])
        per_class_acc[lbl] = acc
        st.write(f"**{lbl}: {acc*100:.2f}%**")

    # -------------------------------
    #  RUNTIME BREAKDOWN PLOT (SMALL)
    # -------------------------------
    st.subheader("Runtime Breakdown (CPU Cloud)")

    model_load_time = model_data.get("model_load_time", 0.18)
    inference_time = end - start
    total_time = model_load_time + inference_time

    fig, ax = plt.subplots(figsize=(6, 3))  # SMALL FIGURE

    bars = ax.bar(
        ["Model Load", "Inference", "Total"],
        [model_load_time, inference_time, total_time],
        color=["#4A90E2", "#E74C3C", "#2ECC71"],
        alpha=0.85
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.003,
                f"{height:.4f}s",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Seconds")
    ax.set_title("Runtime Breakdown (CPU Cloud)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    st.pyplot(fig)

    # -------------------------------
    #  PCA SCATTER PLOT (SMALL)
    # -------------------------------
    st.subheader("PCA Feature Scatter (Test Set)")

    X_pca = X_test[:, :2]
    unique_labels = np.unique(y_test)

    colors = [
        "#1f77b4", "#2ca02c", "#9467bd",
        "#d62728", "#bcbd22", "#17becf"
    ]

    fig, ax = plt.subplots(figsize=(6, 3))  # SMALL FIGURE

    for i, lbl in enumerate(unique_labels):
        mask = y_test == lbl
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=8, alpha=0.6, color=colors[i],
            label=labels[lbl]
        )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("PCA Scatter (Test Set)", fontsize=11, fontweight="bold")
    ax.legend(markerscale=2, fontsize=7)

    st.pyplot(fig)

    # -------------------------------
    # PER-CLASS ACCURACY BAR PLOT (SMALL)
    # -------------------------------
    st.subheader("Per-Class Accuracy (Graph)")

    classes = list(per_class_acc.keys())
    values = [per_class_acc[c] * 100 for c in classes]

    fig, ax = plt.subplots(figsize=(6, 3))  # SMALL FIGURE

    bars = ax.bar(classes, values, color="#4A90E2", alpha=0.85)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy per Activity Class", fontsize=11, fontweight="bold")

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                f"{height:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.xticks(rotation=25, fontsize=8)

    st.pyplot(fig)
