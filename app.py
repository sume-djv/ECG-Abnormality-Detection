from pathlib import Path
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf

from src.utils import signal_quality_score, interpretation

ART = Path("artifacts")

st.set_page_config(page_title="ECG Abnormality Benchmark", page_icon="🫀", layout="wide")
st.title("🫀 ECG Abnormality Detection")
st.caption("AI classification benchmark — NOT a clinical diagnosis system.")

if not (ART / "ecg_cnn.keras").exists():
    st.warning("Model artifacts are not present. Run `python train.py` first.")
    st.stop()

model = tf.keras.models.load_model(ART / "ecg_cnn.keras")

with open(ART / "metrics.json") as f:
    metrics = json.load(f)

data = np.load(ART / "test_samples.npz")
X, y, stored_prob = data["X"], data["y"], data["prob"]

st.sidebar.header("Test sample")
i = st.sidebar.slider("Sample", 0, len(X) - 1, 0)
signal = X[i].squeeze()
prob = float(model.predict(signal[None, :, None], verbose=0)[0, 0])
quality = signal_quality_score(signal)

label = "Abnormal" if prob >= 0.5 else "Normal"
confidence = max(prob, 1 - prob)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Prediction", label)
c2.metric("Abnormal probability", f"{prob:.1%}")
c3.metric("Model confidence", f"{confidence:.1%}")
c4.metric("Signal quality", f"{quality:.1%}")

st.subheader("ECG waveform")
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(signal, linewidth=1.2)
ax.axvline(len(signal)//2, linestyle="--", linewidth=1)
ax.set_xlabel("Sample")
ax.set_ylabel("Normalized amplitude")
ax.grid(alpha=0.2)
st.pyplot(fig, clear_figure=True)

st.info(
    "Novel feature: the dashboard combines model confidence with a heuristic "
    "signal-quality score based on baseline-wander and high-frequency-noise energy. "
    + interpretation(prob, quality)
)

st.subheader("Model performance on held-out test records")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
m2.metric("Precision", f"{metrics['precision']:.3f}")
m3.metric("Recall", f"{metrics['recall']:.3f}")
m4.metric("F1", f"{metrics['f1']:.3f}")
m5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

st.subheader("Confusion matrix")
cm = np.array(metrics["confusion_matrix"])
fig2, ax2 = plt.subplots(figsize=(5, 4))
im = ax2.imshow(cm)
ax2.set_xticks([0, 1], ["Normal", "Abnormal"])
ax2.set_yticks([0, 1], ["Normal", "Abnormal"])
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
for r in range(2):
    for c in range(2):
        ax2.text(c, r, cm[r, c], ha="center", va="center")
st.pyplot(fig2, clear_figure=True)

with st.expander("Classification report"):
    st.code(metrics["classification_report"])

st.caption(
    "Dataset: MIT-BIH Arrhythmia Database via PhysioNet/WFDB. "
    "Results depend on the random seed, records used, preprocessing and hardware. "
    "Do not use this application for medical decisions."
)
