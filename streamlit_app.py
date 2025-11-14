import streamlit as st
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# =====================================================
# PAGE SETTINGS
# =====================================================
st.set_page_config(page_title="HAR Dashboard", layout="wide")
st.title(" HAR Analysis Dashboard")
st.write("High-Performance HAR • PCA + XGBoost • CPU + GPU Comparison")

# =====================================================
# INITIALIZE SESSION STATE
# =====================================================
if "cpu_results" not in st.session_state:
    st.session_state.cpu_results = None
if "gpu_results" not in st.session_state:
    st.session_state.gpu_results = None
if "cpu_executed" not in st.session_state:
    st.session_state.cpu_executed = False
if "gpu_executed" not in st.session_state:
    st.session_state.gpu_executed = False

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
# LOAD GPU RESULTS JSON (From local file)
# =====================================================
@st.cache_data
def load_gpu_results():
    """Load GPU results from local JSON file"""
    try:
        if os.path.exists("gpu_results.json"):
            with open("gpu_results.json", "r") as f:
                return json.load(f)
        else:
            st.error(" gpu_results.json not found in the same folder!")
            return None
    except Exception as e:
        st.error(f"Error loading GPU results: {e}")
        return None

# =====================================================
# STATUS INDICATORS
# =====================================================
st.markdown("---")
st.subheader(" Analysis Status")

col_status1, col_status2, col_status3 = st.columns(3)

with col_status1:
    if st.session_state.cpu_executed:
        st.success(" CPU Complete")
    else:
        st.warning(" CPU Pending")

with col_status2:
    if st.session_state.gpu_executed:
        st.success(" GPU Complete")
    else:
        st.warning(" GPU Pending")

with col_status3:
    if st.session_state.cpu_executed and st.session_state.gpu_executed:
        st.success(" Comparison Ready")
    else:
        st.info(" Run Both First")

st.markdown("---")

# =====================================================
# ACTION BUTTONS
# =====================================================
st.subheader(" Run Analysis")

col1, col2, col3 = st.columns(3)

# -------- CPU INFERENCE BUTTON --------
with col1:
    if st.button(" Run CPU Inference", use_container_width=True):
        with st.spinner("Running CPU inference..."):
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

            st.session_state.cpu_results = cpu_results
            st.session_state.cpu_executed = True
            
            st.success(" CPU Inference Complete!")
            st.rerun()

# -------- GPU INFERENCE BUTTON --------
with col2:
    if st.button(" Run GPU Inference", use_container_width=True):
        with st.spinner("Loading GPU results..."):
            gpu_data = load_gpu_results()
            
            if gpu_data:
                st.session_state.gpu_results = gpu_data
                st.session_state.gpu_executed = True
                st.success(" GPU Results Loaded!")
                st.rerun()
            else:
                st.error(" Failed to load GPU results!")

# -------- COMPARE BUTTON --------
with col3:
    both_complete = st.session_state.cpu_executed and st.session_state.gpu_executed
    
    if st.button(" Compare Results", 
                 use_container_width=True, 
                 disabled=not both_complete):
        if both_complete:
            st.success(" Showing comparison below!")
            st.rerun()

st.markdown("---")

# =====================================================
# DISPLAY CPU RESULTS
# =====================================================
if st.session_state.cpu_executed and st.session_state.cpu_results:
    
    cpu = st.session_state.cpu_results
    
    st.header(" CPU Results")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{cpu['overall_accuracy']*100:.2f}%")
    with col2:
        st.metric("Inference Time", f"{cpu['inference_time']:.5f}s")
    with col3:
        st.metric("Model Load Time", f"{cpu['model_load_time']:.5f}s")

    # -------- Runtime Plot --------
    st.subheader(" Runtime Breakdown")
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(["Model Load", "Inference"],
                  [cpu["model_load_time"], cpu["inference_time"]],
                  color=["#4A90E2", "#E74C3C"])

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.002,
                f"{h:.4f}s", ha="center", fontweight='bold')

    ax.set_title("Runtime Breakdown - CPU", fontweight='bold')
    ax.set_ylabel("Time (seconds)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # -------- Per Class Accuracy --------
    st.subheader(" Per Class Accuracy")

    classes = list(cpu["per_class_accuracy"].keys())
    values = [v * 100 for v in cpu["per_class_accuracy"].values()]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(classes, values, color="#4A90E2", alpha=0.8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CPU Per Class Accuracy", fontweight='bold')

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+1,
                f"{h:.1f}%", ha="center", fontweight='bold')

    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Download button
    st.download_button(
        label=" Download CPU Results (JSON)",
        data=json.dumps(cpu, indent=2),
        file_name="cpu_results.json",
        mime="application/json"
    )
    
    st.markdown("---")

# =====================================================
# DISPLAY GPU RESULTS
# =====================================================
if st.session_state.gpu_executed and st.session_state.gpu_results:
    
    gpu = st.session_state.gpu_results
    
    st.header(" GPU Results")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{gpu['overall_accuracy']*100:.2f}%")
    with col2:
        st.metric("Inference Time", f"{gpu['inference_time']:.5f}s")
    with col3:
        st.metric("Test Samples", gpu.get('test_samples', 'N/A'))

    # -------- Per Class Accuracy Chart --------
    st.subheader(" Per Class Accuracy")
    
    classes = list(gpu["per_class_accuracy"].keys())
    values = [v * 100 for v in gpu["per_class_accuracy"].values()]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(classes, values, color="#2ECC71", alpha=0.8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("GPU Per Class Accuracy", fontweight='bold')

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+1,
                f"{h:.1f}%", ha="center", fontweight='bold')

    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")

# =====================================================
# CPU vs GPU COMPARISON
# =====================================================
if st.session_state.cpu_executed and st.session_state.gpu_executed:
    
    cpu = st.session_state.cpu_results
    gpu = st.session_state.gpu_results
    
    st.header(" CPU vs GPU Comparison")

    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        acc_diff = abs(cpu['overall_accuracy'] - gpu['overall_accuracy']) * 100
        st.metric("Accuracy Difference", f"{acc_diff:.4f}%")

    with col2:
        speedup = cpu["inference_time"] / gpu["inference_time"]
        st.metric("GPU Speedup", f"{speedup:.2f}x", 
                 delta=f"{speedup-1:.2f}x faster")

    with col3:
        time_saved = cpu['inference_time'] - gpu['inference_time']
        st.metric("Time Saved", f"{time_saved:.4f}s")

    with col4:
        reduction = (time_saved / cpu['inference_time']) * 100
        st.metric("Time Reduction", f"{reduction:.1f}%")

    st.markdown("---")

    # -------- Compare Accuracy Side by Side --------
    st.subheader(" Accuracy Comparison")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    devices = ["CPU", "GPU"]
    acc_values = [
        cpu["overall_accuracy"] * 100,
        gpu["overall_accuracy"] * 100,
    ]
    bars = ax.bar(devices, acc_values, color=["#4A90E2", "#2ECC71"], alpha=0.8)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.5,
                f"{h:.2f}%", ha="center", fontweight='bold', fontsize=12)
    
    ax.set_ylim([min(acc_values)-2, max(acc_values)+3])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall Accuracy Comparison", fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # -------- Compare Inference Time --------
    st.subheader(" Inference Time Comparison")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    infer = [cpu["inference_time"], gpu["inference_time"]]
    bars = ax.bar(devices, infer, color=["#4A90E2", "#2ECC71"], alpha=0.8)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+h*0.05,
                f"{h:.5f}s", ha="center", fontweight='bold', fontsize=12)
    
    ax.set_ylabel("Time (seconds)")
    ax.set_title(f"Inference Time (GPU is {speedup:.2f}x faster)", 
                fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # -------- Per-Class Accuracy Comparison --------
    st.subheader(" Per-Class Accuracy Comparison")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    classes = list(cpu["per_class_accuracy"].keys())
    cpu_acc = [cpu["per_class_accuracy"][c] * 100 for c in classes]
    gpu_acc = [gpu["per_class_accuracy"][c] * 100 for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cpu_acc, width, label='CPU', 
                   color='#4A90E2', alpha=0.8)
    bars2 = ax.bar(x + width/2, gpu_acc, width, label='GPU', 
                   color='#2ECC71', alpha=0.8)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # -------- Comparison Summary Table --------
    st.subheader(" Detailed Comparison Table")
    
    comparison_data = {
        "Metric": [
            "Overall Accuracy",
            "Inference Time",
            "Model Load Time",
            "GPU Speedup",
            "Time Saved"
        ],
        "CPU": [
            f"{cpu['overall_accuracy']*100:.2f}%",
            f"{cpu['inference_time']:.5f}s",
            "1.00x (baseline)",
            "0.0000s (baseline)"
        ],
        "GPU": [
            f"{gpu['overall_accuracy']*100:.2f}%",
            f"{gpu['inference_time']:.5f}s",
            f"{speedup:.2f}x",
            f"{time_saved:.5f}s"
        ]
    }
    
    st.dataframe(comparison_data, use_container_width=True, hide_index=True)
    
    # Download comparison report
    comparison_report = {
        "summary": {
            "accuracy_difference": float(acc_diff),
            "gpu_speedup": float(speedup),
            "time_saved": float(time_saved),
            "reduction_percentage": float(reduction)
        },
        "cpu_results": cpu,
        "gpu_results": gpu
    }
    
    st.download_button(
        label=" Download Comparison Report (JSON)",
        data=json.dumps(comparison_report, indent=2),
        file_name="comparison_report.json",
        mime="application/json"
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p style='font-size: 0.9rem;'>
        <b> Powered by XGBoost • PCA • UCI HAR Dataset</b>
    </p>
    <p style='font-size: 0.8rem;'>
        Human Activity Recognition Dashboard v2.0
    </p>
</div>
""", unsafe_allow_html=True)



