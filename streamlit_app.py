import streamlit as st
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="HAR Dashboard", layout="wide")

st.title("HAR Analysis Dashboard")
st.write("High-Performance HAR • PCA + XGBoost • Cloud CPU Inference Only")

# Load model
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

st.subheader("CPU Inference (Cloud)")

# Run Button
if st.button("Run CPU Inference"):
    start = time.time()
    preds = xgb.predict(X_test)
    end = time.time()

    st.success("Inference Complete!")

    st.write(f"**Inference Time:** {end - start:.5f} seconds")
    st.write(f"**Test Accuracy:** {model_data['test_accuracy']*100:.2f}%")

    # -----------------------------
    # PER-CLASS ACCURACY CALCULATION
    # -----------------------------
    st.subheader("Per-Class Accuracy")

    class_accuracies = {}

    for i, lbl in labels.items():
        class_mask = (y_test == i)
        class_acc = np.mean(preds[class_mask] == y_test[class_mask])
        class_accuracies[lbl] = class_acc * 100
        st.write(f"**{lbl}:** {class_acc*100:.2f}%")

    # -----------------------------
    #  PLOT: PER-CLASS ACCURACY BAR CHART
    # -----------------------------
    st.subheader("Per-Class Accuracy Chart")

    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=list(class_accuracies.keys()), 
                y=list(class_accuracies.values()), 
                palette="viridis",
                ax=ax)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
    ax.set_title("Per-Class Accuracy (Cloud CPU)", fontsize=13)

    st.pyplot(fig)

    # -----------------------------
    #  PLOT: PCA SCATTER (2 Components)
    # -----------------------------
    st.subheader("PCA Feature Visualization")

    # If PCA already reduced to 50 components, show first 2
    X_vis = X_test[:, :2]

    fig2, ax2 = plt.subplots(figsize=(6,5))
    scatter = ax2.scatter(X_vis[:, 0], X_vis[:, 1], 
                          c=y_test, cmap="tab10", s=6)
    ax2.set_title("PCA Feature Scatter Plot (Test Set)")
    ax2.set_xlabel("PCA Component 1")
    ax2.set_ylabel("PCA Component 2")

    legend_labels = [labels[i] for i in sorted(labels.keys())]
    handles = [plt.Line2D([], [], marker="o", linestyle="", 
                          color=scatter.cmap(scatter.norm(i))) 
               for i in sorted(labels.keys())]

    ax2.legend(handles, legend_labels, fontsize=7, loc="upper right")
    st.pyplot(fig2)

    # -----------------------------
    # 3️ PLOT: INFERENCE TIME BAR
    # -----------------------------
    st.subheader("Inference Time Visualization")

    fig3, ax3 = plt.subplots(figsize=(4,3))
    ax3.bar(["CPU Time"], [end - start], color="#4CAF50")
    ax3.set_ylabel("Seconds")
    ax3.set_title("Inference Time (Cloud CPU)")

    st.pyplot(fig3)

