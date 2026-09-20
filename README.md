# ECG Abnormality Detection — AI Classification Benchmark

> **Important:** This project is an AI research/benchmark demo, not a clinical diagnostic system.

## What this project does
- Uses the public **MIT-BIH Arrhythmia Database** through PhysioNet/WFDB.
- Extracts heartbeat-centered ECG segments from expert beat annotations.
- Converts the original beat symbols into a simple benchmark:
  - `0 = Normal`
  - `1 = Abnormal`
- Trains a lightweight 1D CNN.
- Reports accuracy, precision, recall, F1, ROC-AUC, confusion matrix and classification report.
- Visualizes the ECG waveform for a selected sample.
- Includes a **novel signal-quality + confidence feature**:
  - estimates signal quality from baseline wander and high-frequency noise,
  - combines it with model confidence,
  - warns when a prediction is low-confidence or the segment quality is poor.

## Dataset
MIT-BIH Arrhythmia Database:
https://physionet.org/content/mitdb/1.0.0/

The database contains 48 half-hour, two-channel ambulatory ECG excerpts sampled at 360 Hz with roughly 110,000 expert beat annotations.

## Project structure
```text
ecg_abnormality_detection/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
└── src/
    ├── __init__.py
    ├── data.py
    ├── model.py
    └── utils.py
```

## 1. Install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Train
```bash
python train.py
```

This automatically downloads the MIT-BIH records with `wfdb`, extracts beat-centered windows, creates a patient/record-aware split, trains the CNN, and saves:
```text
artifacts/ecg_cnn.keras
artifacts/metrics.json
artifacts/test_samples.npz
```

## 3. Launch the dashboard
```bash
streamlit run app.py
```

The dashboard shows:
- ECG waveform
- predicted class
- probability
- signal-quality score
- confidence/quality interpretation
- test-set metrics
- confusion matrix

## Benchmark design
The default binary mapping is:
- Normal: `N`
- Abnormal: all retained non-normal beat symbols

This is intentionally a **benchmark classification task**, not a clinical diagnosis.

For a stricter research benchmark, the code can be extended to AAMI 5-class grouping.

## Novel feature
### Signal Quality + Confidence Gate
The model prediction is paired with a simple quality estimator:
- low-frequency energy → baseline-wander indicator
- high-frequency energy → noise indicator
- normalized quality score in `[0, 1]`
- model confidence is reported separately

If the quality score or confidence is low, the dashboard displays an **"uncertain / inspect signal"** state rather than presenting the prediction as a diagnosis.

## Citation
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine, 2001.
