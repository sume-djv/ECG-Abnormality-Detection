from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tensorflow as tf

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
# THEME
# ============================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

theme = st.sidebar.radio(
    "🎨 Theme",
    ["Light", "Dark"],
    horizontal=True,
    index=0 if st.session_state.theme == "Light" else 1,
)

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
        [data-testid="stMetric"] {
            background: #111827;
            border: 1px solid #263244;
            border-radius: 14px;
            padding: 12px;
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
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e3e8f0;
            border-radius: 14px;
            padding: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ARTIFACTS
# ============================================================

MODEL_PATH = ART / "ecg_cnn.keras"
METRICS_PATH = ART / "metrics.json"
TEST_PATH = ART / "test_samples.npz"

for path, message in [
    (MODEL_PATH, "Model artifact is missing. Run `python train.py` first."),
    (METRICS_PATH, "Metrics artifact is missing. Run `python train.py` first."),
    (TEST_PATH, "Test-sample artifact is missing. Run `python train.py` first."),
]:
    if not path.exists():
        st.error(f"⚠️ {message}")
        st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_project_data():
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)

    data = np.load(TEST_PATH)
    return metrics, data["X"], data["y"]


model = load_model()
metrics, X, y = load_project_data()

# Stored probabilities are useful for ROC/probability plots.
test_data = np.load(TEST_PATH)
stored_prob = (
    np.asarray(test_data["prob"]).reshape(-1)
    if "prob" in test_data.files
    else None
)


# ============================================================
# HELPERS
# ============================================================

def prepare_signal(signal):
    """Convert an ECG vector to the 187-sample format used by the CNN."""
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)

    if signal.size == 0:
        raise ValueError("ECG signal is empty.")

    if signal.size != 187:
        old_axis = np.linspace(0, 1, signal.size)
        new_axis = np.linspace(0, 1, 187)
        signal = np.interp(
            new_axis,
            old_axis,
            signal,
        ).astype(np.float32)

    signal = signal - signal.mean()
    signal = signal / (signal.std() + 1e-7)

    return signal


def predict_signal(signal, threshold=0.50):
    """Run the trained CNN and return probability, quality and timing."""
    signal = prepare_signal(signal)

    probability = float(
        model.predict(
            signal[None, :, None],
            verbose=0,
        )[0, 0]
    )

    quality = float(signal_quality_score(signal))
    label = "Abnormal" if probability >= threshold else "Normal"
    confidence = probability if label == "Abnormal" else 1 - probability

    return signal, probability, quality, label, confidence


def quality_status(score):
    if score >= 0.85:
        return "Good", "🟢"
    if score >= 0.65:
        return "Fair", "🟡"
    return "Low", "🔴"


def probability_status(probability):
    if probability < 0.20:
        return "Very low abnormal probability", "🟢"
    if probability < 0.50:
        return "Below decision threshold", "🟡"
    if probability < 0.80:
        return "Elevated abnormal probability", "🟠"
    return "High abnormal probability", "🔴"


def get_indices(selection):
    labels = np.asarray(y).reshape(-1)

    if selection == "Healthy / Normal Beat":
        return np.where(labels == 0)[0]

    if selection == "Abnormal Beat":
        return np.where(labels == 1)[0]

    return np.arange(len(labels))


def make_ecg_plotly(
    signal,
    start=0,
    end=187,
    show_smooth=True,
    show_reference=True,
):
    start = max(0, min(int(start), 186))
    end = max(start + 1, min(int(end), 187))

    x = np.arange(start, end)
    view = signal[start:end]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=view,
            mode="lines",
            name="ECG waveform",
            line=dict(width=2),
        )
    )

    if show_smooth:
        window = 9
        kernel = np.ones(window) / window
        smoothed = np.convolve(signal, kernel, mode="same")

        fig.add_trace(
            go.Scatter(
                x=x,
                y=smoothed[start:end],
                mode="lines",
                name="Smoothed trend",
                line=dict(width=2),
            )
        )

    if show_reference:
        reference = len(signal) // 2

        if start <= reference <= end:
            fig.add_vline(
                x=reference,
                line_dash="dash",
                annotation_text="Reference",
                annotation_position="top",
            )

    if theme == "Dark":
        plot_bg = "#111827"
        paper_bg = "#111827"
        text_color = "#e5e7eb"
        grid_color = "#263244"
    else:
        plot_bg = "#ffffff"
        paper_bg = "#ffffff"
        text_color = "#172033"
        grid_color = "#e5e7eb"

    fig.update_layout(
        title="Interactive Waveform Telemetry",
        xaxis_title="Sample",
        yaxis_title="Normalized amplitude",
        hovermode="x unified",
        dragmode="pan",
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font=dict(color=text_color),
        xaxis=dict(
            gridcolor=grid_color,
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(gridcolor=grid_color),
        legend=dict(orientation="h"),
        margin=dict(l=30, r=20, t=55, b=30),
        height=470,
    )

    return fig


def make_confusion_matrix():
    return np.asarray(
        metrics["confusion_matrix"],
        dtype=np.int64,
    )


def make_roc_plot():
    from sklearn.metrics import roc_curve, auc

    if stored_prob is None or len(stored_prob) != len(y):
        return None

    fpr, tpr, _ = roc_curve(y, stored_prob)
    score = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {score:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def make_probability_plot():
    if stored_prob is None or len(stored_prob) != len(y):
        return None

    normal = stored_prob[np.asarray(y) == 0]
    abnormal = stored_prob[np.asarray(y) == 1]

    fig, ax = plt.subplots(figsize=(6.2, 4.8))

    ax.hist(normal, bins=25, alpha=0.65, label="Normal")
    ax.hist(abnormal, bins=25, alpha=0.65, label="Abnormal")
    ax.axvline(0.5, linestyle="--", linewidth=1, label="Default threshold")

    ax.set_xlabel("Predicted abnormal probability")
    ax.set_ylabel("Number of beats")
    ax.set_title("Prediction Probability Distribution", fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def show_heartbeat_guide():
    st.caption(
        "Educational reference only. The trained model currently predicts "
        "two classes: Normal and Abnormal."
    )

    c1, c2, c3 = st.columns(3)

    c1.info("**P wave**\n\nTypical atrial depolarization.")
    c2.info("**QRS complex**\n\nTypical ventricular depolarization.")
    c3.info("**T wave**\n\nTypical ventricular repolarization.")


# ============================================================
# SIDEBAR — HEARTBEAT SELECTION
# ============================================================

st.sidebar.title("🫀 ECG AI Analyzer")
st.sidebar.caption("User-friendly ECG exploration dashboard")

st.sidebar.divider()

st.sidebar.markdown("### 🫀 Step 1 — Select Heartbeat")

heartbeat_type = st.sidebar.radio(
    "Choose beat type",
    [
        "All Test Beats",
        "Healthy / Normal Beat",
        "Abnormal Beat",
    ],
)

available = get_indices(heartbeat_type)

if len(available) == 0:
    st.sidebar.error("No samples available for this selection.")
    st.stop()

selected_position = st.sidebar.slider(
    "Heartbeat",
    0,
    len(available) - 1,
    0,
)

selected_index = int(
    available[selected_position]
)

ground_truth = (
    "Abnormal"
    if int(y[selected_index]) == 1
    else "Normal"
)

st.sidebar.caption(
    f"Beat {selected_position + 1:,} of {len(available):,}"
)

st.sidebar.divider()

st.sidebar.markdown("### ⚙️ Controls")

analysis_mode = st.sidebar.radio(
    "Mode",
    [
        "Heartbeat Checker",
        "CSV Upload",
        "Noise Stress Test",
    ],
)

threshold = st.sidebar.slider(
    "Decision threshold",
    0.10,
    0.90,
    0.50,
    0.05,
    help="Probability at or above this value is classified as Abnormal.",
)

st.sidebar.divider()

with st.sidebar.expander("🫀 Heartbeat reference"):
    st.write("**Normal Beat:** model's Normal class.")
    st.write("**Abnormal Beat:** all non-Normal beats in this binary benchmark.")
    st.caption(
        "PVC/PAC/Fusion/Paced are not separate model classes in the current trained artifact."
    )


# ============================================================
# HEADER
# ============================================================

st.title("🫀 ECG AI Analyzer")

st.caption(
    "Interactive ECG abnormality classification benchmark"
)

st.warning(
    "⚠️ Research / educational use only — not a clinical diagnosis system."
)

top = st.columns([1, 1, 1, 1.25])

top[0].metric("Dataset", "MIT-BIH")
top[1].metric("AI Model", "1D CNN")
top[2].metric("Input", "187 samples")
top[3].metric("Task", "Binary ECG", "Normal / Abnormal")


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔎 Heartbeat Checker",
        "📈 Interactive Waveform Telemetry",
        "📊 Accuracy & Scorecard",
        "ℹ️ Project Guide",
    ]
)


# ============================================================
# TAB 1 — HEARTBEAT CHECKER
# ============================================================

with tab1:

    if analysis_mode == "Heartbeat Checker":

        signal, probability, quality, prediction, confidence = predict_signal(
            X[selected_index].squeeze(),
            threshold,
        )

        st.markdown("### 🫀 Selected Heartbeat")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Prediction",
            f"{'🟢' if prediction == 'Normal' else '🔴'} {prediction}",
            f"Ground truth: {ground_truth}",
        )

        c2.metric(
            "Abnormal Probability",
            f"{probability:.1%}",
        )

        c3.metric(
            "Model Confidence",
            f"{confidence:.1%}",
        )

        qtext, qicon = quality_status(quality)

        c4.metric(
            "Signal Quality",
            f"{quality:.1%}",
            f"{qicon} {qtext}",
        )

        if prediction == ground_truth:
            st.success(
                f"✅ Sample #{selected_index}: prediction matches the ground truth."
            )
        else:
            st.warning(
                f"⚠️ Sample #{selected_index}: prediction differs from the ground truth."
            )

        st.markdown("### 📈 ECG Waveform")

        st.plotly_chart(
            make_ecg_plotly(signal),
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
            },
        )

        probability_text, probability_icon = probability_status(probability)

        st.progress(
            int(np.clip(probability, 0, 1) * 100),
        )

        st.caption(
            f"{probability_icon} {probability_text} • "
            f"Threshold: {threshold:.2f}"
        )

        st.markdown("### ✨ Novel Features")

        n1, n2 = st.columns(2)

        with n1:
            st.info(
                "**✨ Signal Quality–Aware Prediction**\n\n"
                "The model output is shown together with a heuristic "
                "signal-quality score."
            )

        with n2:
            st.info(
                "**✨ Heartbeat-Aware Exploration**\n\n"
                "Filter the held-out test set into Normal or Abnormal beats "
                "and inspect them one by one."
            )

        with st.expander("🫀 ECG waveform guide"):
            show_heartbeat_guide()

        st.info(
            interpretation(
                probability,
                quality,
            )
        )

    elif analysis_mode == "CSV Upload":

        st.markdown("### 📤 Upload ECG CSV")

        uploaded = st.file_uploader(
            "Choose an ECG CSV file",
            type=["csv"],
        )

        if uploaded is None:

            st.info(
                "Upload a CSV containing numeric ECG samples to explore it."
            )

        else:

            try:

                dataframe = pd.read_csv(uploaded)
                numeric = dataframe.select_dtypes(include=np.number)

                if numeric.empty:

                    st.error(
                        "No numeric ECG column was found in the uploaded file."
                    )

                else:

                    raw_signal = (
                        numeric.iloc[:, 0]
                        .dropna()
                        .to_numpy(dtype=np.float32)
                    )

                    (
                        signal,
                        probability,
                        quality,
                        prediction,
                        confidence,
                    ) = predict_signal(
                        raw_signal,
                        threshold,
                    )

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric(
                        "Prediction",
                        f"{'🟢' if prediction == 'Normal' else '🔴'} {prediction}",
                    )

                    c2.metric(
                        "Abnormal Probability",
                        f"{probability:.1%}",
                    )

                    c3.metric(
                        "Model Confidence",
                        f"{confidence:.1%}",
                    )

                    qtext, qicon = quality_status(quality)

                    c4.metric(
                        "Signal Quality",
                        f"{quality:.1%}",
                        f"{qicon} {qtext}",
                    )

                    st.markdown("### 📈 Uploaded ECG")

                    st.plotly_chart(
                        make_ecg_plotly(signal),
                        use_container_width=True,
                        config={
                            "displaylogo": False,
                            "scrollZoom": True,
                        },
                    )

                    st.info(
                        "Uploaded signals are normalized and converted to the "
                        "187-sample format expected by the trained CNN."
                    )

            except Exception as error:

                st.error(
                    f"Could not process this CSV: {error}"
                )

    else:

        st.markdown("### 🌪️ Noise Stress Test")

        noise_level = st.slider(
            "Noise strength",
            0.00,
            1.00,
            0.15,
            0.01,
        )

        clean_signal, clean_prob, clean_quality, clean_label, clean_conf = (
            predict_signal(
                X[selected_index].squeeze(),
                threshold,
            )
        )

        rng = np.random.default_rng(
            1000 + selected_index + int(noise_level * 100)
        )

        noisy_raw = (
            clean_signal
            + rng.normal(
                0,
                noise_level,
                clean_signal.shape,
            ).astype(np.float32)
        )

        noisy_signal, noisy_prob, noisy_quality, noisy_label, noisy_conf = (
            predict_signal(
                noisy_raw,
                threshold,
            )
        )

        st.markdown("### ✨ Novel Feature — Noise Robustness Test")

        a, b = st.columns(2)

        a.metric(
            "Clean",
            clean_label,
            f"Quality {clean_quality:.1%} • Confidence {clean_conf:.1%}",
        )

        b.metric(
            "Noisy",
            noisy_label,
            f"Quality {noisy_quality:.1%} • Confidence {noisy_conf:.1%}",
        )

        st.plotly_chart(
            make_ecg_plotly(
                clean_signal,
                0,
                187,
                noisy_signal=None,
            ),
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
            },
        )

        n1, n2, n3 = st.columns(3)

        n1.metric(
            "Noise level",
            f"{noise_level:.2f}",
        )

        n2.metric(
            "Quality change",
            f"{noisy_quality - clean_quality:+.1%}",
        )

        n3.metric(
            "Confidence change",
            f"{noisy_conf - clean_conf:+.1%}",
        )

        st.info(
            "This is a robustness experiment, not a clinical noise-performance test."
        )


# ============================================================
# TAB 2 — INTERACTIVE WAVEFORM TELEMETRY
# ============================================================

with tab2:

    st.markdown("## 📈 Interactive Waveform Telemetry")

    st.caption(
        "Click and drag to pan, use the mouse wheel to zoom, or use the "
        "Plotly toolbar for box zoom and reset."
    )

    telemetry_index = st.selectbox(
        "🫀 Select heartbeat",
        options=available.tolist(),
        index=selected_position,
        format_func=lambda idx: (
            f"Sample #{idx} — "
            f"{'Abnormal' if int(y[idx]) else 'Normal'}"
        ),
    )

    telemetry_signal, telemetry_prob, telemetry_quality, telemetry_label, telemetry_conf = (
        predict_signal(
            X[telemetry_index].squeeze(),
            threshold,
        )
    )

    preset = st.selectbox(
        "🔍 View",
        [
            "Full heartbeat",
            "Center reference",
            "Middle section",
        ],
    )

    if preset == "Full heartbeat":
        view_start, view_end = 0, 187
    elif preset == "Center reference":
        view_start, view_end = 50, 137
    else:
        view_start, view_end = 47, 140

    custom = st.checkbox(
        "Custom view",
        False,
    )

    if custom:
        s1, s2 = st.columns(2)

        view_start = s1.slider(
            "Start",
            0,
            186,
            view_start,
        )

        view_end = s2.slider(
            "End",
            view_start + 1,
            187,
            max(view_start + 1, view_end),
        )

    show_smooth = st.checkbox(
        "Show smoothed trend",
        True,
    )

    show_reference = st.checkbox(
        "Show reference position",
        True,
    )

    st.plotly_chart(
        make_ecg_plotly(
            telemetry_signal,
            view_start,
            view_end,
            show_smooth=show_smooth,
            show_reference=show_reference,
        ),
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
        },
    )

    t1, t2, t3, t4 = st.columns(4)

    t1.metric(
        "Prediction",
        telemetry_label,
    )

    t2.metric(
        "Abnormal probability",
        f"{telemetry_prob:.1%}",
    )

    t3.metric(
        "Signal quality",
        f"{telemetry_quality:.1%}",
    )

    t4.metric(
        "Confidence",
        f"{telemetry_conf:.1%}",
    )

    st.markdown("### 🫀 Heartbeat Navigation")

    positions = available.tolist()
    current_position = positions.index(telemetry_index)

    n1, n2, n3 = st.columns(3)

    with n1:
        if st.button(
            "⬅️ Previous Beat",
            disabled=current_position == 0,
            use_container_width=True,
        ):
            st.session_state.telemetry_index = int(
                positions[current_position - 1]
            )
            st.rerun()

    with n2:
        st.metric(
            "Current beat",
            f"#{telemetry_index}",
        )

    with n3:
        if st.button(
            "Next Beat ➡️",
            disabled=current_position == len(positions) - 1,
            use_container_width=True,
        ):
            st.session_state.telemetry_index = int(
                positions[current_position + 1]
            )
            st.rerun()

    with st.expander("🫀 What can be explored here?"):
        st.write(
            "Use the sample selector and navigation buttons to move through the "
            "selected Normal/Abnormal heartbeat group. The waveform can be "
            "zoomed and panned directly."
        )


# ============================================================
# TAB 3 — PERFORMANCE
# ============================================================

with tab3:

    st.markdown("## 📊 Accuracy & Scorecard")

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['precision']:.3f}")
    m3.metric("Recall", f"{metrics['recall']:.3f}")
    m4.metric("F1 Score", f"{metrics['f1']:.3f}")
    m5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    if stored_prob is not None and len(stored_prob) == len(y):

        c1, c2 = st.columns(2)

        with c1:

            roc_fig = make_roc_plot()

            if roc_fig is not None:
                st.pyplot(
                    roc_fig,
                    clear_figure=True,
                )

        with c2:

            prob_fig = make_probability_plot()

            if prob_fig is not None:
                st.pyplot(
                    prob_fig,
                    clear_figure=True,
                )

    st.markdown("### 🧩 Confusion Matrix")

    cm = make_confusion_matrix()

    matrix_mode = st.radio(
        "Display",
        [
            "Absolute counts",
            "Normalized by actual class",
        ],
        horizontal=True,
    )

    display_cm = cm.astype(float)

    if matrix_mode == "Normalized by actual class":

        row_sums = display_cm.sum(
            axis=1,
            keepdims=True,
        )

        display_cm = np.divide(
            display_cm,
            row_sums,
            out=np.zeros_like(display_cm),
            where=row_sums != 0,
        ) * 100

    fig, ax = plt.subplots(
        figsize=(6.5, 5),
    )

    image = ax.imshow(
        display_cm,
    )

    ax.set_xticks(
        [0, 1],
        ["Normal", "Abnormal"],
    )

    ax.set_yticks(
        [0, 1],
        ["Normal", "Abnormal"],
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        "Confusion Matrix",
        fontweight="bold",
    )

    for r in range(2):
        for c in range(2):

            value = display_cm[r, c]

            text = (
                f"{int(value)}"
                if matrix_mode == "Absolute counts"
                else f"{value:.1f}%"
            )

            ax.text(
                c,
                r,
                text,
                ha="center",
                va="center",
                fontweight="bold",
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    with st.expander("📋 Classification Report"):
        st.code(
            metrics["classification_report"]
        )


# ============================================================
# TAB 4 — PROJECT GUIDE
# ============================================================

with tab4:

    st.markdown("## ℹ️ Project Guide")

    p1, p2, p3, p4, p5, p6 = st.columns(6)

    p1.info("**1. Dataset**\n\nMIT-BIH ECG records")
    p2.info("**2. Beat Extraction**\n\n187-sample heartbeat")
    p3.info("**3. Normalize**\n\nPer-beat normalization")
    p4.info("**4. CNN**\n\n1D convolutional model")
    p5.info("**5. Predict**\n\nNormal / Abnormal")
    p6.info("**6. Analyze**\n\nQuality + telemetry")

    st.markdown("### ✨ Dashboard Features")

    f1, f2 = st.columns(2)

    f1.markdown(
        """
        **🫀 Heartbeat Explorer**

        Select Normal or Abnormal test beats and move through them one by one.

        **📈 Interactive Waveform Telemetry**

        Zoom, pan, change the view window and inspect the selected heartbeat.
        """
    )

    f2.markdown(
        """
        **🔬 Signal Quality Awareness**

        Display the model prediction together with a heuristic signal-quality score.

        **🌪️ Robustness Test**

        Add controlled noise and compare model behavior.
        """
    )

    with st.expander("🫀 ECG basics"):
        show_heartbeat_guide()

    st.warning(
        "⚠️ This application is a research/educational benchmark, "
        "not a clinical diagnosis system."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🫀 ECG AI Analyzer • MIT-BIH benchmark • Research/educational use only"
)
