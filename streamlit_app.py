import streamlit as st
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import time

# =====================================================
# PAGE SETTINGS
# =====================================================
st.set_page_config(page_title="HAR Dashboard", layout="wide")
st.title("HAR Analysis Dashboard")
st.write("High-Performance HAR • PCA + XGBoost • CPU + GPU Comparison (GPU from Docker Export)")

# =====================================================
# LOAD CPU MODEL (CLOUD)
# =====================================================
@st.cache_resource
def load_model():
    with open("har_model_complete.pkl", "rb") as f:
        return pickle.load(f)

model_data = load_model()
xgb = model_data["xgb"]
X_test = model_data["X_test_pca"]
y_test = model_data["y_test"]
labels = model_data["activity_labels"]

# =====================================================
# LOAD GPU RESULTS JSON (user uploaded file)
# =====================================================
st.sidebar.header("GPU Results Upload (From Docker Container)")
uploaded_file = st.sidebar.file_uploader("Upload gpu_results.json", type=["json"])

gpu_results = None
if uploaded_file:
    gpu_results = json.load(uploaded_file)
    st.sidebar.success("GPU Results Loaded!")

# =====================================================
# CPU INFERENCE BUTTON
# =====================================================
st.header("CPU Inference (Cloud Execution)")

if st.button("Run CPU Inference"):
    start = time.time()
    preds = xgb.predict(X_test)
    end = time.time()

    cpu_results = {
        "overall_accuracy": model_data["test_accuracy"],
        "inference_time": end - start,
        "model_load_time": model_data.get("model_load_time", 0.18),
        "per_class_accuracy": {},
    }

    # Calculate per class accuracy
    for i, lbl in labels.items():
        mask = y_test == i
        cpu_results["per_class_accuracy"][lbl] = float(np.mean(preds[mask] == y_test[mask]))

    st.success("CPU Inference Complete!")
    st.write(f"**Accuracy:** {cpu_results['overall_accuracy']*100:.2f}%")
    st.write(f"**Inference Time:** {cpu_results['inference_time']:.5f} sec")

    # Save CPU results into session
    st.session_state["cpu_results"] = cpu_results


# =====================================================
# DISPLAY CPU RESULTS (with charts)
# =====================================================
if "cpu_results" in st.session_state:

    cpu = st.session_state["cpu_results"]

    st.subheader("CPU Results (Cloud)")
    st.write(cpu)

    # -------- Runtime Plot --------
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(["Model Load", "Inference"],
                  [cpu["model_load_time"], cpu["inference_time"]],
                  color=["#4A90E2", "#E74C3C"])

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.002,
                f"{h:.4f}s", ha="center")

    ax.set_title("Runtime Breakdown - CPU")
    st.pyplot(fig)

    # -------- Per Class Accuracy --------
    st.subheader("CPU Per Class Accuracy")

    classes = list(cpu["per_class_accuracy"].keys())
    values = [v * 100 for v in cpu["per_class_accuracy"].values()]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(classes, values, color="#4A90E2")
    ax.set_ylim(0, 100)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+1,
                f"{h:.1f}%", ha="center")

    plt.xticks(rotation=25)
    st.pyplot(fig)

# =====================================================
# DISPLAY GPU RESULTS (when uploaded)
# =====================================================
if gpu_results:
    st.header("GPU Results (From Docker Container)")

    st.write(f"**GPU Accuracy:** {gpu_results['overall_accuracy']*100:.2f}%")
    st.write(f"**Inference Time:** {gpu_results['inference_time']:.5f} sec")

    # -------- Per Class Accuracy Chart --------
    classes = list(gpu_results["per_class_accuracy"].keys())
    values = [v * 100 for v in gpu_results["per_class_accuracy"].values()]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(classes, values, color="#2ECC71")
    ax.set_ylim(0, 100)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+1,
                f"{h:.1f}%", ha="center")

    plt.xticks(rotation=25)
    ax.set_title("GPU Per Class Accuracy")
    st.pyplot(fig)


# =====================================================
# CPU vs GPU COMPARISON
# =====================================================
if "cpu_results" in st.session_state and gpu_results:
    st.header("CPU vs GPU Comparison")

    cpu = st.session_state["cpu_results"]
    gpu = gpu_results

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy Difference",
                  f"{abs(cpu['overall_accuracy'] - gpu['overall_accuracy'])*100:.2f}%")

    with col2:
        speedup = cpu["inference_time"] / gpu["inference_time"]
        st.metric("GPU Speedup", f"{speedup:.2f}x")

    with col3:
        st.metric("Time Saved",
                  f"{cpu['inference_time'] - gpu['inference_time']:.4f}s")

    # -------- Compare Accuracy Side by Side --------
    fig, ax = plt.subplots(figsize=(5, 3))
    devices = ["CPU", "GPU"]
    acc_values = [
        cpu["overall_accuracy"] * 100,
        gpu["overall_accuracy"] * 100,
    ]
    ax.bar(devices, acc_values, color=["#4A90E2", "#2ECC71"])
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy Comparison")
    st.pyplot(fig)

    # -------- Compare Inference Time --------
    fig, ax = plt.subplots(figsize=(5, 3))
    infer = [cpu["inference_time"], gpu["inference_time"]]
    ax.bar(devices, infer, color=["#4A90E2", "#2ECC71"])
    ax.set_title("Inference Time Comparison")
    st.pyplot(fig)
