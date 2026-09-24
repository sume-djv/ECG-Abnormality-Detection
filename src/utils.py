import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

def signal_quality_score(x, fs=360):
    """
    Heuristic benchmark feature, not a medical quality standard.
    Returns 0..1. Higher means cleaner-looking signal.
    """
    x = np.asarray(x).astype(float).ravel()
    x = x - np.mean(x)
    if len(x) < 16:
        return 0.0

    f, pxx = welch(x, fs=fs, nperseg=min(128, len(x)))
    total = trapezoid(pxx, f) + 1e-12
    baseline = trapezoid(pxx[f < 0.5], f[f < 0.5])
    high = trapezoid(pxx[f > 40], f[f > 40])

    baseline_ratio = baseline / total
    high_ratio = high / total

    penalty = 1.5 * baseline_ratio + 1.5 * high_ratio
    return float(np.clip(1.0 - penalty, 0.0, 1.0))

def interpretation(prob_abnormal, quality):
    confidence = max(prob_abnormal, 1 - prob_abnormal)
    if quality < 0.35 or confidence < 0.60:
        return "Uncertain / inspect signal"
    if prob_abnormal >= 0.5:
        return "Benchmark class: Abnormal"
    return "Benchmark class: Normal"
