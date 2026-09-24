from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import roc_curve, auc

from src.utils import signal_quality_score, interpretation


# ============================================================
# CONFIG
# ============================================================

ART = Path("artifacts")

st.set_page_config(
    page_title="ECG Abnormality Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-top: 5px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 25px;
    }

    .status-box {
        padding: 14px 18px;
        border-radius: 10px;
        background-color: #f3f4f6;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .disclaimer {
        padding: 14px 18px;
        border-radius: 8px;
        background-color: #fff7ed;
        border-left: 5px solid #f97316;
        margin-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🫀 ECG Abnormality Detection</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-based ECG classification benchmark using the MIT-BIH Arrhythmia Database"
    "</div>",
    unsafe_allow_html=True,
)

st.write("")


# ============================================================
# LOAD MODEL
# ============================================================

if not (ART / "ecg_cnn.keras").exists():
    st.error(
        "Model artifacts are not available. "
        "Please run `python train.py` first."
    )
    st.stop()


@st.cache_resource
def load_ecg_model():
    return tf.keras.models.load_model(
        ART / "ecg_cnn.keras"
    )


model = load_ecg_model()


# ============================================================
# LOAD METRICS
# ============================================================

with open(ART / "metrics.json") as f:
    metrics = json.load(f)


# ============================================================
# LOAD TEST DATA
# ============================================================

data = np.load(
    ART / "test_samples.npz"
)

X = data["X"]
y = data["y"]

if "prob" in data.files:
    stored_prob = data["prob"]
else:
    stored_prob = model.predict(
        X,
        verbose=0
    ).ravel()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧪 ECG Analysis")

mode = st.sidebar.radio(
    "Input mode",
    [
        "Test Dataset Sample",
        "Upload ECG CSV",
    ],
)


# ============================================================
# DATASET SAMPLE MODE
# ============================================================

if mode == "Test Dataset Sample":

    st.sidebar.markdown("---")

    i = st.sidebar.slider(
        "Select test sample",
        min_value=0,
        max_value=len(X) - 1,
        value=0,
    )

    signal = X[i].squeeze()

    sample_source = f"MIT-BIH test sample #{i}"


# ============================================================
# CSV UPLOAD MODE
# ============================================================

else:

    st.sidebar.markdown("---")

    uploaded_file = st.sidebar.file_uploader(
        "Upload ECG signal CSV",
        type=["csv"],
        help="Upload a CSV containing ECG signal values.",
    )

    if uploaded_file is None:

        st.info(
            "👈 Upload an ECG CSV file from the sidebar "
            "to analyze a custom signal."
        )

        st.stop()

    try:

        df = pd.read_csv(
            uploaded_file,
            header=None,
        )

        numeric_values = pd.to_numeric(
            df.iloc[:, 0],
            errors="coerce",
        ).dropna().to_numpy(
            dtype=np.float32
        )

        if len(numeric_values) < 2:
            st.error(
                "The uploaded CSV does not contain enough "
                "numeric ECG samples."
            )
            st.stop()

        original_length = len(numeric_values)

        # Resample to 187 points expected by the CNN
        original_axis = np.linspace(
            0,
            1,
            original_length,
        )

        target_axis = np.linspace(
            0,
            1,
            187,
        )

        signal = np.interp(
            target_axis,
            original_axis,
            numeric_values,
        ).astype(np.float32)

        # Normalize uploaded signal
        signal = (
            signal - signal.mean()
        ) / (
            signal.std() + 1e-7
        )

        sample_source = (
            f"Uploaded ECG "
            f"({original_length} samples → 187)"
        )

    except Exception as e:

        st.error(
            f"Could not read the uploaded CSV: {e}"
        )

        st.stop()


# ============================================================
# SIDEBAR PROJECT INFORMATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader("📚 Project Information")

st.sidebar.write(
    "**Dataset:** MIT-BIH Arrhythmia Database"
)

st.sidebar.write(
    "**Task:** Binary ECG classification"
)

st.sidebar.write(
    "**Classes:** Normal / Abnormal"
)

st.sidebar.write(
    "**Input:** 187 ECG samples"
)

st.sidebar.write(
    "**Model:** 1D CNN"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Research and educational demonstration only."
)


# ============================================================
# PREDICTION
# ============================================================

prob = float(
    model.predict(
        signal[None, :, None],
        verbose=0,
    )[0, 0]
)

quality = signal_quality_score(
    signal
)

label = (
    "Abnormal"
    if prob >= 0.5
    else "Normal"
)

confidence = max(
    prob,
    1 - prob,
)


# ============================================================
# QUALITY STATUS
# ============================================================

if quality >= 0.80:
    quality_status = "Good 🟢"

elif quality >= 0.60:
    quality_status = "Moderate 🟡"

else:
    quality_status = "Low 🔴"


prediction_icon = (
    "🔴"
    if label == "Abnormal"
    else "🟢"
)


# ============================================================
# PREDICTION SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🔍 ECG Prediction</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Prediction",
    f"{prediction_icon} {label}",
)

c2.metric(
    "Abnormal Probability",
    f"{prob:.1%}",
)

c3.metric(
    "Model Confidence",
    f"{confidence:.1%}",
)

c4.metric(
    "Signal Quality",
    f"{quality:.1%}",
)


st.markdown(
    f"""
    <div class="status-box">
    <b>Input:</b> {sample_source}
    &nbsp; | &nbsp;
    <b>Prediction:</b> {label}
    &nbsp; | &nbsp;
    <b>Signal quality:</b> {quality_status}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ECG WAVEFORM
# ============================================================

st.markdown(
    '<div class="section-title">📈 ECG Waveform</div>',
    unsafe_allow_html=True,
)

fig, ax = plt.subplots(
    figsize=(13, 4)
)

ax.plot(
    signal,
    linewidth=1.4,
)

ax.axvline(
    len(signal) // 2,
    linestyle="--",
    linewidth=1,
    label="Reference position",
)

ax.set_xlabel(
    "Sample"
)

ax.set_ylabel(
    "Normalized amplitude"
)

ax.set_title(
    "ECG Signal"
)

ax.grid(
    alpha=0.2
)

ax.legend()

st.pyplot(
    fig,
    clear_figure=True,
)


# ============================================================
# INTERPRETATION
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Model Interpretation</div>',
    unsafe_allow_html=True,
)

st.info(
    interpretation(
        prob,
        quality,
    )
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "Accuracy",
    f"{metrics['accuracy']:.3f}",
)

m2.metric(
    "Precision",
    f"{metrics['precision']:.3f}",
)

m3.metric(
    "Recall",
    f"{metrics['recall']:.3f}",
)

m4.metric(
    "F1 Score",
    f"{metrics['f1']:.3f}",
)

m5.metric(
    "ROC-AUC",
    f"{metrics['roc_auc']:.3f}",
)


# ============================================================
# ANALYTICS TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Confusion Matrix",
        "📈 ROC Curve",
        "📋 Classification Report",
        "📚 Dataset & Model",
    ]
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

with tab1:

    cm = np.array(
        metrics["confusion_matrix"]
    )

    fig2, ax2 = plt.subplots(
        figsize=(6, 5)
    )

    ax2.imshow(cm)

    ax2.set_xticks(
        [0, 1],
        ["Normal", "Abnormal"],
    )

    ax2.set_yticks(
        [0, 1],
        ["Normal", "Abnormal"],
    )

    ax2.set_xlabel(
        "Predicted"
    )

    ax2.set_ylabel(
        "Actual"
    )

    ax2.set_title(
        "Confusion Matrix"
    )

    for r in range(2):

        for c in range(2):

            ax2.text(
                c,
                r,
                cm[r, c],
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

    st.pyplot(
        fig2,
        clear_figure=True,
    )


# ============================================================
# ROC CURVE
# ============================================================

with tab2:

    fpr, tpr, _ = roc_curve(
        y,
        stored_prob,
    )

    roc_value = auc(
        fpr,
        tpr,
    )

    fig3, ax3 = plt.subplots(
        figsize=(7, 5)
    )

    ax3.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC-AUC = {roc_value:.3f}",
    )

    ax3.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Random classifier",
    )

    ax3.set_xlabel(
        "False Positive Rate"
    )

    ax3.set_ylabel(
        "True Positive Rate"
    )

    ax3.set_title(
        "Receiver Operating Characteristic"
    )

    ax3.grid(
        alpha=0.2
    )

    ax3.legend()

    st.pyplot(
        fig3,
        clear_figure=True,
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

with tab3:

    st.code(
        metrics["classification_report"],
        language="text",
    )


# ============================================================
# DATASET + MODEL INFORMATION
# ============================================================

with tab4:

    left, right = st.columns(2)

    with left:

        st.subheader(
            "📚 Dataset"
        )

        st.write(
            "**MIT-BIH Arrhythmia Database**"
        )

        st.write(
            "Public ECG dataset accessed using WFDB."
        )

        st.write(
            f"Test samples: **{len(X):,}**"
        )

        normal_count = int(
            np.sum(y == 0)
        )

        abnormal_count = int(
            np.sum(y == 1)
        )

        st.write(
            f"Normal samples: **{normal_count:,}**"
        )

        st.write(
            f"Abnormal samples: **{abnormal_count:,}**"
        )

        # Class distribution
        labels = [
            "Normal",
            "Abnormal",
        ]

        counts = [
            normal_count,
            abnormal_count,
        ]

        fig4, ax4 = plt.subplots(
            figsize=(6, 4)
        )

        ax4.bar(
            labels,
            counts,
        )

        ax4.set_ylabel(
            "Number of samples"
        )

        ax4.set_title(
            "Test Set Class Distribution"
        )

        for idx, value in enumerate(
            counts
        ):

            ax4.text(
                idx,
                value,
                str(value),
                ha="center",
                va="bottom",
            )

        st.pyplot(
            fig4,
            clear_figure=True,
        )

    with right:

        st.subheader(
            "🧠 Model"
        )

        st.write(
            "**1D Convolutional Neural Network (CNN)**"
        )

        st.write(
            "Input: 187-sample ECG beat"
        )

        st.write(
            "Output: Normal / Abnormal"
        )

        st.write(
            "Binary classification using sigmoid output."
        )

        st.write(
            "Preprocessing includes per-beat normalization."
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">
    ⚠️ <b>Research / Educational Use Only</b><br><br>
    This application is an AI classification benchmark and
    <b>not a clinical diagnosis system</b>.
    Predictions should not be used for medical decisions.
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Dataset: MIT-BIH Arrhythmia Database via PhysioNet/WFDB. "
    "Results depend on records, preprocessing, random seed "
    "and hardware used during training."
)