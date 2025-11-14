"""
HAR Analysis - Elegant Streamlit Interface
Refined UI with Smart Stacking and Compact Design
"""

import streamlit as st
import subprocess
import json
import os
import tempfile
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO

# Page config
st.set_page_config(
    page_title="HAR Analysis Tool",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'cpu_results' not in st.session_state:
    st.session_state.cpu_results = None
if 'gpu_results' not in st.session_state:
    st.session_state.gpu_results = None
if 'cpu_images' not in st.session_state:
    st.session_state.cpu_images = {}
if 'gpu_images' not in st.session_state:
    st.session_state.gpu_images = {}
if 'show_comparison' not in st.session_state:
    st.session_state.show_comparison = False

# Custom CSS - Elegant Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
        letter-spacing: 0.5px;
    }
    
    .stButton>button {
        width: 100%;
        height: 45px;
        font-size: 0.9rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        margin: 5px 0;
        letter-spacing: 0.3px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    .stButton>button:disabled {
        background: linear-gradient(135deg, #475569 0%, #64748b 100%) !important;
        cursor: not-allowed !important;
        transform: none !important;
        box-shadow: none !important;
        opacity: 0.6;
    }
    
    .status-compact {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 3px solid #3b82f6;
        text-align: center;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .status-compact.complete {
        border-left-color: #10b981;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
    }
    
    .status-compact.pending {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
    }
    
    .status-compact h2 {
        margin: 0;
        font-size: 0.95rem;
        font-weight: 700;
    }
    
    .status-compact p {
        margin: 0.3rem 0 0 0;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border: 1px solid #334155;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #60a5fa;
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #3b82f6;
    }
    
    .viz-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        margin: 0.8rem 0;
        border: 1px solid #334155;
    }
    
    .viz-container img {
        border-radius: 8px;
        width: 100%;
        height: auto;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.1) 0%, rgba(167, 139, 250, 0.1) 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 3px solid #60a5fa;
        margin: 0.8rem 0;
        color: #e2e8f0;
        font-weight: 500;
        font-size: 0.85rem;
    }
    
    .stExpander {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 10px;
        border: 1px solid #334155;
    }
    
    div[data-testid="stImage"] {
        text-align: center;
    }
    
    div[data-testid="stImage"] > img {
        max-width: 100%;
        height: auto;
        border-radius: 8px;
    }
    
    .result-section {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin: 1rem 0;
    }
    
    .stDownloadButton > button {
        height: 40px !important;
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header"> HAR Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">High-Performance HAR • PCA + XGBoost • CPU vs GPU Inference</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("###  Configuration")
    
    docker_username = st.text_input(
        "Docker Hub Username",
        value="rethika11",
        help="Your Docker Hub username",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### Dataset Info")
    st.markdown("""
    <div class="info-box">
    <b>Dataset:</b> UCI HAR | <b>Samples:</b> 10,299<br>
    <b>Train/Test:</b> 7,352 / 2,947 | <b>Features:</b> 561 → 50<br>
    <b>Classes:</b> 6 | <b>Model:</b> XGBoost
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Pipeline")
    st.markdown("""
    <div class="info-box" style="font-size: 0.8rem;">
    1️ Load Model | 2️ Load Data<br>
    3️ Inference | 4️ Metrics<br>
    5️ Visualize | 6️ Save
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("###  Docker Status")
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success(" Docker Ready")
        else:
            st.error(" Docker Error")
    except:
        st.error(" Docker Not Found")

# Check prerequisites
if not docker_username:
    st.warning(" Enter Docker Hub username")
    st.stop()

try:
    subprocess.run(["docker", "--version"], capture_output=True, check=True)
except:
    st.error(" Docker not running")
    st.stop()

# Status Section - Compact
st.markdown('<p class="section-header"> Analysis Status</p>', unsafe_allow_html=True)

col_status1, col_status2, col_status3 = st.columns(3)

with col_status1:
    if st.session_state.cpu_results is not None:
        st.markdown("""
        <div class="status-compact complete">
            <h2> CPU Complete</h2>
            <p>Ready</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-compact pending">
            <h2> CPU Pending</h2>
            <p>Click to run</p>
        </div>
        """, unsafe_allow_html=True)

with col_status2:
    if st.session_state.gpu_results is not None:
        st.markdown("""
        <div class="status-compact complete">
            <h2> GPU Complete</h2>
            <p>Ready</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-compact pending">
            <h2> GPU Pending</h2>
            <p>Click to run</p>
        </div>
        """, unsafe_allow_html=True)

with col_status3:
    both_complete = (st.session_state.cpu_results is not None and 
                     st.session_state.gpu_results is not None)
    if both_complete:
        st.markdown("""
        <div class="status-compact complete">
            <h2> Compare Ready</h2>
            <p>Both done</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-compact pending">
            <h2> Locked</h2>
            <p>Run both first</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Action Buttons
st.markdown('<p class="section-header"> Run Analysis</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# CPU Analysis Button
with col1:
    if st.button("CPU ANALYSIS", use_container_width=True, key="cpu_btn"):
        with st.spinner(" Running CPU analysis..."):
            image_name = f"{docker_username}/har-cpu:latest"
            
            with tempfile.TemporaryDirectory() as tmpdir:
                results_dir = Path(tmpdir) / "results"
                results_dir.mkdir()
                
                progress = st.progress(0)
                st.info(f" Pulling: {image_name}")
                progress.progress(20)
                subprocess.run(["docker", "pull", image_name], 
                             capture_output=True, timeout=600)
                progress.progress(40)
                
                st.info(" Running inference...")
                progress.progress(60)
                docker_cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{results_dir.absolute()}:/app/results",
                    image_name
                ]
                
                process = subprocess.run(docker_cmd, capture_output=True, text=True)
                progress.progress(80)
                
                if process.returncode != 0:
                    st.error(" Failed")
                    st.code(process.stderr)
                    st.stop()
                
                result_file = results_dir / "results_CPU.json"
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        results = json.load(f)
                    
                    st.session_state.cpu_results = results
                    
                    for img_name, file_name in [
                        ('confusion_matrix', 'confusion_matrix_CPU.png'),
                        ('accuracy', 'accuracy_CPU.png'),
                        ('timing', 'timing_CPU.png')
                    ]:
                        img_file = results_dir / file_name
                        if img_file.exists():
                            with open(img_file, 'rb') as f:
                                st.session_state.cpu_images[img_name] = f.read()
                    
                    progress.progress(100)
                    st.success(" Complete!")
                    time.sleep(1)
                    st.rerun()

# GPU Analysis Button
with col2:
    if st.button(" GPU ANALYSIS", use_container_width=True, key="gpu_btn"):
        with st.spinner(" Running GPU analysis..."):
            image_name = f"{docker_username}/har-gpu:latest"
            
            with tempfile.TemporaryDirectory() as tmpdir:
                results_dir = Path(tmpdir) / "results"
                results_dir.mkdir()
                
                progress = st.progress(0)
                st.info(f" Pulling: {image_name}")
                progress.progress(20)
                subprocess.run(["docker", "pull", image_name], 
                             capture_output=True, timeout=600)
                progress.progress(40)
                
                st.info(" Running inference (GPU)...")
                progress.progress(60)
                docker_cmd = [
                    "docker", "run", "--rm",
                    "--gpus", "all",
                    "-v", f"{results_dir.absolute()}:/app/results",
                    image_name
                ]
                
                process = subprocess.run(docker_cmd, capture_output=True, text=True)
                progress.progress(80)
                
                if process.returncode != 0:
                    st.error(" Failed")
                    st.code(process.stderr)
                    st.stop()
                
                result_file = results_dir / "results_GPU.json"
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        results = json.load(f)
                    
                    st.session_state.gpu_results = results
                    
                    for img_name, file_name in [
                        ('confusion_matrix', 'confusion_matrix_GPU.png'),
                        ('accuracy', 'accuracy_GPU.png'),
                        ('timing', 'timing_GPU.png')
                    ]:
                        img_file = results_dir / file_name
                        if img_file.exists():
                            with open(img_file, 'rb') as f:
                                st.session_state.gpu_images[img_name] = f.read()
                    
                    progress.progress(100)
                    st.success(" Complete!")
                    time.sleep(1)
                    st.rerun()

# Compare Button
with col3:
    both_complete = (st.session_state.cpu_results is not None and 
                     st.session_state.gpu_results is not None)
    
    compare_clicked = st.button(" COMPARE", 
                 use_container_width=True, 
                 disabled=not both_complete,
                 key="compare_btn")
    
    if compare_clicked and both_complete:
        st.session_state.show_comparison = True
        st.rerun()

st.markdown("---")

# Display Results - GPU First if available, then CPU
results_to_display = []

if st.session_state.gpu_results is not None:
    results_to_display.append(('GPU', st.session_state.gpu_results, st.session_state.gpu_images))

if st.session_state.cpu_results is not None:
    results_to_display.append(('CPU', st.session_state.cpu_results, st.session_state.cpu_images))

for device_type, results, images in results_to_display:
    st.markdown(f'<p class="section-header">{"" if device_type == "GPU" else ""} {device_type} Results</p>', unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{results["overall_accuracy"]*100:.2f}%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{results["test_samples"]}</div>
            <div class="metric-label">Samples</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{results["inference_time"]:.4f}s</div>
            <div class="metric-label">Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{device_type}</div>
            <div class="metric-label">Device</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("###  Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'confusion_matrix' in images:
            st.markdown('<div class="viz-container">', unsafe_allow_html=True)
            st.image(images['confusion_matrix'], use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if 'accuracy' in images:
            st.markdown('<div class="viz-container">', unsafe_allow_html=True)
            st.image(images['accuracy'], use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    if 'timing' in images:
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        st.image(images['timing'], use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed metrics
    with st.expander("Full Details"):
        st.json(results)
    
    # Download button
    st.download_button(
        label=f" {device_type} JSON",
        data=json.dumps(results, indent=2),
        file_name=f"{device_type.lower()}_results.json",
        mime="application/json",
        use_container_width=True
    )
    
    st.markdown("---")

# Comparison Section
if st.session_state.show_comparison and st.session_state.cpu_results is not None and st.session_state.gpu_results is not None:
    st.markdown('<p class="section-header"> CPU vs GPU Comparison</p>', unsafe_allow_html=True)
    
    cpu_results = st.session_state.cpu_results
    gpu_results = st.session_state.gpu_results
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    acc_diff = abs(cpu_results['overall_accuracy'] - gpu_results['overall_accuracy']) * 100
    speedup = cpu_results['inference_time'] / gpu_results['inference_time']
    time_saved = cpu_results['inference_time'] - gpu_results['inference_time']
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{acc_diff:.4f}%</div>
            <div class="metric-label">Acc Diff</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{speedup:.2f}x</div>
            <div class="metric-label">Speedup</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{time_saved:.4f}s</div>
            <div class="metric-label">Saved</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{time_saved/cpu_results['inference_time']*100:.1f}%</div>
            <div class="metric-label">Reduction</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Comparison Charts
    st.markdown("###  Visualizations")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.patch.set_facecolor('#0f172a')
    
    # Overall accuracy
    ax1 = axes[0]
    devices = ['CPU', 'GPU']
    accuracies = [cpu_results['overall_accuracy']*100, gpu_results['overall_accuracy']*100]
    colors = ['#3b82f6', '#10b981']
    
    bars = ax1.bar(devices, accuracies, color=colors, alpha=0.85, edgecolor='#60a5fa', linewidth=1.5)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11, color='#e2e8f0')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold', color='#94a3b8')
    ax1.set_title('Accuracy Comparison', fontsize=13, fontweight='bold', color='#e2e8f0', pad=12)
    ax1.set_ylim([min(accuracies) - 2, max(accuracies) + 2])
    ax1.grid(axis='y', alpha=0.2, linestyle='--', color='#334155')
    ax1.set_facecolor('#1e293b')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#334155')
    ax1.spines['bottom'].set_color('#334155')
    ax1.tick_params(colors='#94a3b8', labelsize=10)
    
    # Inference time
    ax2 = axes[1]
    times = [cpu_results['inference_time'], gpu_results['inference_time']]
    
    bars = ax2.bar(devices, times, color=colors, alpha=0.85, edgecolor='#60a5fa', linewidth=1.5)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{height:.4f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=11, color='#e2e8f0')
    
    ax2.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold', color='#94a3b8')
    ax2.set_title(f'Inference Time (GPU {speedup:.2f}x faster)', 
                 fontsize=13, fontweight='bold', color='#e2e8f0', pad=12)
    ax2.grid(axis='y', alpha=0.2, linestyle='--', color='#334155')
    ax2.set_facecolor('#1e293b')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#334155')
    ax2.spines['bottom'].set_color('#334155')
    ax2.tick_params(colors='#94a3b8', labelsize=10)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    # Per-Class Accuracy
    st.markdown("###  Per-Class Accuracy")
    
    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_facecolor('#0f172a')
    
    activities = list(cpu_results['per_class_accuracy'].keys())
    cpu_acc = [cpu_results['per_class_accuracy'][act]*100 for act in activities]
    gpu_acc = [gpu_results['per_class_accuracy'][act]*100 for act in activities]
    
    x = np.arange(len(activities))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cpu_acc, width, label='CPU', 
                   color='#3b82f6', alpha=0.85, edgecolor='#60a5fa', linewidth=1.2)
    bars2 = ax.bar(x + width/2, gpu_acc, width, label='GPU', 
                   color='#10b981', alpha=0.85, edgecolor='#34d399', linewidth=1.2)
    
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold', color='#94a3b8')
    ax.set_title('Per-Class Accuracy', fontsize=13, fontweight='bold', color='#e2e8f0', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(activities, rotation=25, ha='right', fontsize=10, color='#94a3b8')
    ax.legend(fontsize=11, frameon=True, shadow=False, facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
    ax.grid(axis='y', alpha=0.2, linestyle='--', color='#334155')
    ax.set_facecolor('#1e293b')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.tick_params(axis='y', colors='#94a3b8', labelsize=10)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    # Summary Table
    st.markdown("###  Comparison Table")
    
    comparison_data = {
        "Metric": [
            "Overall Accuracy",
            "Inference Time",
            "Test Samples",
            "PCA Components"
        ],
        "CPU": [
            f"{cpu_results['overall_accuracy']*100:.2f}%",
            f"{cpu_results['inference_time']:.4f}s",
            cpu_results['test_samples'],
            cpu_results.get('pca_components', 'N/A')
        ],
        "GPU": [
            f"{gpu_results['overall_accuracy']*100:.2f}%",
            f"{gpu_results['inference_time']:.4f}s",
            gpu_results['test_samples'],
            gpu_results.get('pca_components', 'N/A')
        ]
    }
    
    st.dataframe(comparison_data, use_container_width=True, hide_index=True)
    
    # Download comparison report
    comparison_report = {
        "summary": {
            "accuracy_difference": acc_diff,
            "gpu_speedup": speedup,
            "time_saved": time_saved,
            "reduction_percentage": time_saved/cpu_results['inference_time']*100
        },
        "cpu_results": cpu_results,
        "gpu_results": gpu_results
    }
    
    st.download_button(
        label=" Download Comparison Report",
        data=json.dumps(comparison_report, indent=2),
        file_name="comparison_report.json",
        mime="application/json",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94a3b8; padding: 1.5rem; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 10px; margin-top: 1rem; border: 1px solid #334155;'>
    <p style='font-size: 0.95rem; font-weight: 600; margin-bottom: 0.5rem;'>
        <b> Powered by Docker • PCA + XGBoost • UCI HAR</b>
    </p>
    <p style='font-size: 0.85rem; color: #64748b;'>
        Pre-trained Model • Fast Inference  • CPU & GPU Support
    </p>
    <p style='font-size: 0.8rem; color: #475569; margin-top: 0.5rem;'>
        Human Activity Recognition Dashboard v2.0
    </p>
</div>
""", unsafe_allow_html=True)