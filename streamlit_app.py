import streamlit as st
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="HAR Dashboard", layout="wide")

st.title("HAR Analysis Dashboard")
st.write("High-Performance HAR • PCA + XGBoost • Cloud CPU Inference Only")

# Load model immediately at startup
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

if st.button("Run CPU Inference"):
    start = time.time()
    preds = xgb.predict(X_test)
    end = time.time()

    st.success("Inference Complete!")

    st.write(f"**Inference Time:** {end - start:.5f} seconds")
    st.write(f"**Test Accuracy:** {model_data['test_accuracy']*100:.2f}%")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=labels.values(),
                yticklabels=labels.values(),
                ax=ax)
    plt.title("Confusion Matrix (CPU Cloud)")
    st.pyplot(fig)

    # Per class accuracy
    st.subheader("Per-Class Accuracy:")
    for i, lbl in labels.items():
        class_mask = y_test == i
        class_acc = np.mean(preds[class_mask] == y_test[class_mask])
        st.write(f"**{lbl}:** {class_acc*100:.2f}%")

st.info("Docker-based CPU and GPU inference works only on local machine. Cloud platform does not allow nested Docker, so CPU inference is done directly here.")
