from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.utils import signal_quality_score, interpretation


# ============================================================
# CONFIG
# ============================================================

ART = Path("artifacts")

st.set_page_config(
    page_title="ECG AI Analyzer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM THEME
# ============================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Light"


theme = st.sidebar.radio(
    "🎨 Appearance",
    ["Light", "Dark"],
    index=0 if st.session_state.theme == "Light" else 1,
)

st.session_state.theme = theme


if theme == "Dark":

    st.markdown(
        """
        <style>

        .stApp {
            background-color: #0b1220;
            color: #e5e7eb;
        }

        [data-testid="stSidebar"] {
            background-color: #111827;
        }

        [data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }

        h1, h2, h3, h4, h5, p, label {
            color: #e5e7eb !important;
        }

        .metric-card {
            background: #111827;
            border: 1px solid #263244;
            color: #e5e7eb;
        }

        .feature-card {
            background: #111827;
            border: 1px solid #263244;
            color: #e5e7eb;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
        <style>

        .stApp {
            background-color: #f7f9fc;
        }

        .metric-card {
            background: white;
            border: 1px solid #e3e8f0;
            color: #172033;
        }

        .feature-card {
            background: white;
            border: 1px solid #e3e8f0;
            color: #172033;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD MODEL + ARTIFACTS
# ============================================================

MODEL_PATH = ART / "ecg_cnn.keras"
METRICS_PATH = ART / "metrics.json"
TEST_PATH = ART / "test_samples.npz"


if not MODEL_PATH.exists():
    st.error(
        "⚠️ Model file is missing. Please run `python train.py` first."
    )
    st.stop()


if not METRICS_PATH.exists():
    st.error(
        "⚠️ Metrics file is missing. Please run `python train.py` first."
    )
    st.stop()


if not TEST_PATH.exists():
    st.error(
        "⚠️ Test sample file is missing. Please run `python train.py` first."
    )
    st.stop()


import tensorflow as tf


@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


@st.cache_data
def load_project_data():

    with open(
        METRICS_PATH,
        encoding="utf-8"
    ) as f:

        metrics = json.load(f)

    data = np.load(TEST_PATH)

    X = data["X"]
    y = data["y"]

    return metrics, X, y


model = load_model()

metrics, X, y = load_project_data()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def prepare_signal(signal):

    signal = np.asarray(
        signal,
        dtype=np.float32
    ).reshape(-1)

    if len(signal) != 187:

        old_axis = np.linspace(
            0,
            1,
            len(signal)
        )

        new_axis = np.linspace(
            0,
            1,
            187
        )

        signal = np.interp(
            new_axis,
            old_axis,
            signal
        ).astype(np.float32)

    signal = signal - signal.mean()

    signal = signal / (
        signal.std() + 1e-7
    )

    return signal


def predict_signal(signal):

    signal = prepare_signal(signal)

    probability = float(
        model.predict(
            signal[None, :, None],
            verbose=0
        )[0, 0]
    )

    quality = float(
        signal_quality_score(signal)
    )

    label = (
        "Abnormal"
        if probability >= 0.5
        else "Normal"
    )

    confidence = max(
        probability,
        1 - probability
    )

    return (
        signal,
        probability,
        quality,
        label,
        confidence
    )


def quality_status(score):

    if score >= 0.85:

        return "Good", "🟢"

    elif score >= 0.65:

        return "Fair", "🟡"

    else:

        return "Low", "🔴"


def metric_card(
    title,
    value,
    subtitle=""
):

    st.markdown(
        f"""
        <div class="metric-card"
             style="
             padding:18px;
             border-radius:16px;
             text-align:center;
             margin-bottom:10px;
             box-shadow:0 4px 15px rgba(0,0,0,0.05);
             ">

            <div style="
                font-size:12px;
                font-weight:700;
                letter-spacing:0.05em;
            ">
                {title}
            </div>

            <div style="
                font-size:30px;
                font-weight:800;
                margin:7px 0;
            ">
                {value}
            </div>

            <div style="
                font-size:13px;
                opacity:0.7;
            ">
                {subtitle}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def plot_ecg(
    signal,
    title="ECG Signal",
    noisy_signal=None
):

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    ax.plot(
        signal,
        linewidth=1.7,
        label="ECG signal"
    )

    if noisy_signal is not None:

        ax.plot(
            noisy_signal,
            linewidth=1,
            alpha=0.55,
            label="Noisy signal"
        )

    ax.axvline(
        len(signal) // 2,
        linestyle="--",
        linewidth=1.2,
        label="Reference position"
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Sample"
    )

    ax.set_ylabel(
        "Normalized amplitude"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend()

    fig.tight_layout()

    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🫀 ECG AI Analyzer"
)

st.sidebar.caption(
    "Interactive ECG abnormality classification benchmark"
)

st.sidebar.divider()


input_mode = st.sidebar.radio(
    "📥 Input Mode",
    [
        "Test Dataset Sample",
        "Upload ECG CSV",
        "Noise Stress Test"
    ]
)


st.sidebar.divider()


st.sidebar.markdown(
    "### 📌 Project Information"
)

st.sidebar.write(
    "**Dataset:** MIT-BIH Arrhythmia Database"
)

st.sidebar.write(
    "**Task:** Binary ECG Classification"
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


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🫀 ECG AI Analyzer"
)

st.caption(
    "AI-powered ECG waveform classification with "
    "signal-quality awareness."
)

st.warning(
    "⚠️ Research/educational benchmark only — "
    "not a clinical diagnosis system."
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔎 ECG Analyzer",
        "📊 Model Performance",
        "ℹ️ About Project"
    ]
)


# ============================================================
# TAB 1 — ECG ANALYZER
# ============================================================

with tab1:

    # --------------------------------------------------------
    # TEST DATASET SAMPLE
    # --------------------------------------------------------

    if input_mode == "Test Dataset Sample":

        st.sidebar.subheader(
            "🧪 Test Sample"
        )

        sample_index = st.sidebar.slider(
            "Select ECG sample",
            0,
            len(X) - 1,
            0
        )


        signal = X[
            sample_index
        ].squeeze().astype(
            np.float32
        )


        (
            signal,
            probability,
            quality,
            prediction,
            confidence
        ) = predict_signal(signal)


        ground_truth = (
            "Abnormal"
            if int(y[sample_index]) == 1
            else "Normal"
        )


        quality_text, quality_icon = (
            quality_status(quality)
        )


        is_match = (
            prediction == ground_truth
        )


        # ----------------------------------------------------
        # RESULT HEADER
        # ----------------------------------------------------

        st.markdown(
            "### 🔎 ECG Prediction"
        )


        c1, c2, c3, c4 = st.columns(4)


        with c1:

            icon = (
                "🟢"
                if prediction == "Normal"
                else "🔴"
            )

            metric_card(
                "PREDICTION",
                f"{icon} {prediction}",
                f"Ground truth: {ground_truth}"
            )


        with c2:

            metric_card(
                "ABNORMAL PROBABILITY",
                f"{probability:.1%}",
                "Model output"
            )


        with c3:

            metric_card(
                "MODEL CONFIDENCE",
                f"{confidence:.1%}",
                "Prediction certainty"
            )


        with c4:

            metric_card(
                "SIGNAL QUALITY",
                f"{quality:.1%}",
                f"{quality_icon} {quality_text}"
            )


        # ----------------------------------------------------
        # SAMPLE STATUS
        # ----------------------------------------------------

        status_text = (
            "✅ Prediction matches ground truth"
            if is_match
            else "⚠️ Prediction differs from ground truth"
        )


        st.markdown(
            f"""
            <div class="feature-card"
                 style="
                 padding:15px 20px;
                 border-radius:14px;
                 margin:15px 0;
                 ">

                <b>📋 Test Sample #{sample_index}</b>

                &nbsp; | &nbsp;

                Prediction:
                <b>{prediction}</b>

                &nbsp; | &nbsp;

                Ground Truth:
                <b>{ground_truth}</b>

                &nbsp; | &nbsp;

                {status_text}

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # WAVEFORM
        # ----------------------------------------------------

        st.markdown(
            "### 📈 ECG Waveform"
        )


        st.pyplot(
            plot_ecg(
                signal,
                f"ECG Sample #{sample_index}"
            ),
            clear_figure=True
        )


        # ----------------------------------------------------
        # NOVEL FEATURE 1
        # ----------------------------------------------------

        st.markdown(
            "### ✨ Novel Feature 1 — Signal Quality–Aware Prediction"
        )


        st.info(
            "The dashboard presents model confidence together "
            "with a heuristic ECG signal-quality score based "
            "on baseline-wander and high-frequency-noise energy."
        )


        # ----------------------------------------------------
        # NOVEL FEATURE 2
        # ----------------------------------------------------

        st.markdown(
            "### ✨ Novel Feature 2 — Confidence + Quality Interpretation"
        )


        interpretation_text = interpretation(
            probability,
            quality
        )


        st.success(
            interpretation_text
        )


    # ========================================================
    # CSV UPLOAD
    # ========================================================

    elif input_mode == "Upload ECG CSV":

        st.markdown(
            "### 📤 Upload ECG CSV"
        )


        st.write(
            "Upload a CSV containing numeric ECG samples. "
            "The first numeric column will be used."
        )


        uploaded_file = st.file_uploader(
            "Choose ECG CSV",
            type=["csv"]
        )


        if uploaded_file is None:

            st.info(
                "📄 No CSV selected yet."
            )


        else:

            try:

                dataframe = pd.read_csv(
                    uploaded_file
                )


                numeric_columns = dataframe.select_dtypes(
                    include=np.number
                )


                if numeric_columns.empty:

                    st.error(
                        "❌ No numeric ECG column was found."
                    )


                else:

                    raw_signal = (
                        numeric_columns
                        .iloc[:, 0]
                        .dropna()
                        .to_numpy(
                            dtype=np.float32
                        )
                    )


                    (
                        signal,
                        probability,
                        quality,
                        prediction,
                        confidence
                    ) = predict_signal(
                        raw_signal
                    )


                    quality_text, quality_icon = (
                        quality_status(
                            quality
                        )
                    )


                    c1, c2, c3, c4 = (
                        st.columns(4)
                    )


                    with c1:

                        icon = (
                            "🟢"
                            if prediction == "Normal"
                            else "🔴"
                        )

                        metric_card(
                            "PREDICTION",
                            f"{icon} {prediction}",
                            "Uploaded ECG"
                        )


                    with c2:

                        metric_card(
                            "ABNORMAL PROBABILITY",
                            f"{probability:.1%}",
                            "Model output"
                        )


                    with c3:

                        metric_card(
                            "MODEL CONFIDENCE",
                            f"{confidence:.1%}",
                            "Prediction certainty"
                        )


                    with c4:

                        metric_card(
                            "SIGNAL QUALITY",
                            f"{quality:.1%}",
                            f"{quality_icon} {quality_text}"
                        )


                    st.markdown(
                        "### 📈 Uploaded ECG"
                    )


                    st.pyplot(
                        plot_ecg(
                            signal,
                            "Uploaded ECG — normalized to 187 samples"
                        ),
                        clear_figure=True
                    )


                    st.markdown(
                        "### ✨ Novel Feature 3 — User ECG Exploration"
                    )


                    st.info(
                        "The same benchmark preprocessing and "
                        "classification pipeline can be explored "
                        "with a user-provided numeric ECG signal."
                    )


            except Exception as error:

                st.error(
                    f"❌ Could not process this CSV: {error}"
                )


    # ========================================================
    # NOISE STRESS TEST
    # ========================================================

    else:

        st.markdown(
            "### 🧪 Noise Stress Test"
        )


        st.write(
            "Add controlled Gaussian noise to a test heartbeat "
            "and compare signal quality and model output."
        )


        sample_index = st.sidebar.slider(
            "Base ECG sample",
            0,
            len(X) - 1,
            0,
            key="noise_sample"
        )


        noise_level = st.sidebar.slider(
            "Noise level",
            0.00,
            1.00,
            0.15,
            0.01
        )


        clean_signal, clean_probability, clean_quality, clean_prediction, clean_confidence = (
            predict_signal(
                X[sample_index].squeeze()
            )
        )


        rng = np.random.default_rng(
            42 + sample_index +
            int(noise_level * 1000)
        )


        noisy_raw = (
            clean_signal
            +
            rng.normal(
                0,
                noise_level,
                clean_signal.shape
            ).astype(np.float32)
        )


        noisy_signal, noisy_probability, noisy_quality, noisy_prediction, noisy_confidence = (
            predict_signal(
                noisy_raw
            )
        )


        st.markdown(
            "### ✨ Novel Feature 4 — Noise Stress Test"
        )


        a, b = st.columns(2)


        with a:

            metric_card(
                "CLEAN SIGNAL",
                clean_prediction,
                f"Quality {clean_quality:.1%} • Confidence {clean_confidence:.1%}"
            )


        with b:

            metric_card(
                "NOISY SIGNAL",
                noisy_prediction,
                f"Quality {noisy_quality:.1%} • Confidence {noisy_confidence:.1%}"
            )


        st.pyplot(
            plot_ecg(
                clean_signal,
                "Clean vs Noisy ECG",
                noisy_signal=noisy_signal
            ),
            clear_figure=True
        )


        q1, q2, q3 = st.columns(3)


        with q1:

            metric_card(
                "NOISE LEVEL",
                f"{noise_level:.2f}",
                "Controlled Gaussian noise"
            )


        with q2:

            metric_card(
                "QUALITY CHANGE",
                f"{noisy_quality - clean_quality:+.1%}"
            )


        with q3:

            metric_card(
                "CONFIDENCE CHANGE",
                f"{noisy_confidence - clean_confidence:+.1%}"
            )


        st.info(
            "This is a robustness experiment. "
            "It does not represent clinical performance "
            "under real-world noise."
        )


# ============================================================
# TAB 2 — MODEL PERFORMANCE
# ============================================================

with tab2:

    st.markdown(
        "### 📊 Model Performance Center"
    )


    st.caption(
        "Metrics calculated on the held-out test records "
        "used by the benchmark."
    )


    m1, m2, m3, m4, m5 = (
        st.columns(5)
    )


    with m1:

        metric_card(
            "ACCURACY",
            f"{metrics['accuracy']:.3f}"
        )


    with m2:

        metric_card(
            "PRECISION",
            f"{metrics['precision']:.3f}"
        )


    with m3:

        metric_card(
            "RECALL",
            f"{metrics['recall']:.3f}"
        )


    with m4:

        metric_card(
            "F1 SCORE",
            f"{metrics['f1']:.3f}"
        )


    with m5:

        metric_card(
            "ROC-AUC",
            f"{metrics['roc_auc']:.3f}"
        )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.markdown(
        "### 🧩 Confusion Matrix"
    )


    confusion_matrix = np.array(
        metrics["confusion_matrix"]
    )


    fig, ax = plt.subplots(
        figsize=(6, 5)
    )


    image = ax.imshow(
        confusion_matrix
    )


    ax.set_xticks(
        [0, 1],
        ["Normal", "Abnormal"]
    )


    ax.set_yticks(
        [0, 1],
        ["Normal", "Abnormal"]
    )


    ax.set_xlabel(
        "Predicted"
    )


    ax.set_ylabel(
        "Actual"
    )


    ax.set_title(
        "Confusion Matrix",
        fontweight="bold"
    )


    for row in range(2):

        for col in range(2):

            ax.text(
                col,
                row,
                int(
                    confusion_matrix[
                        row,
                        col
                    ]
                ),
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold"
            )


    fig.colorbar(
        image,
        ax=ax
    )


    fig.tight_layout()


    st.pyplot(
        fig,
        clear_figure=True
    )


    # --------------------------------------------------------
    # REPORT + EXPLANATION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            "### 📋 Classification Report"
        )


        st.code(
            metrics[
                "classification_report"
            ]
        )


    with col2:

        st.markdown(
            "### 🧠 Metric Guide"
        )


        st.markdown(
            """
            **Accuracy**  
            Overall fraction of correct predictions.

            **Precision**  
            How often predicted abnormal beats were actually abnormal.

            **Recall**  
            How many abnormal beats were detected.

            **F1 Score**  
            Balance between precision and recall.

            **ROC-AUC**  
            Measures ranking performance across classification thresholds.
            """
        )


# ============================================================
# TAB 3 — ABOUT
# ============================================================

with tab3:

    st.markdown(
        "### ℹ️ About This Project"
    )


    st.markdown(
        """
        ## 🫀 ECG AI Analyzer

        This project is an AI-based research/educational benchmark
        for binary ECG abnormality classification.

        ### Dataset

        **MIT-BIH Arrhythmia Database**

        ### AI Model

        **1D Convolutional Neural Network (1D CNN)**

        ### Input

        **187 ECG samples per heartbeat**

        ### Classes

        🟢 Normal  
        🔴 Abnormal

        ### Implemented Dashboard Features

        - 🔎 ECG test-sample explorer
        - 📈 ECG waveform visualization
        - ✨ Signal-quality-aware prediction
        - ✨ Confidence + quality interpretation
        - 📤 ECG CSV exploration
        - 🧪 Noise stress testing
        - 📊 Model performance center
        - 🧩 Confusion matrix
        - 📋 Classification report
        - 🌗 Light / Dark theme
        - 🎨 User-friendly dashboard design
        """
    )


    st.warning(
        "⚠️ This application is not a medical device "
        "and must not be used for diagnosis or medical decisions."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "🫀 ECG AI Analyzer • MIT-BIH benchmark • "
    "Results depend on dataset split, preprocessing, "
    "random seed, model and hardware."
)