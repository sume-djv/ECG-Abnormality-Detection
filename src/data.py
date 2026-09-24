import os
from pathlib import Path
import numpy as np
import wfdb

DATA_DIR = Path("data/mitdb")
ARTIFACT_DIR = Path("artifacts")

# MIT-BIH beat symbols retained for the binary benchmark.
NORMAL_SYMBOLS = {"N"}
# Symbols commonly appearing in MIT-BIH annotations.
VALID_SYMBOLS = {
    "N","L","R","B","A","a","J","S","V","r","F","e","j","n","E","/","f","Q","?",
    "!","[","]","x","(",")","t","u","m"
}

def ensure_records(record_ids=None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if record_ids is None:
        record_ids = [
            "100","101","102","103","104","105","106","107","108","109",
            "111","112","113","114","115","116","117","118","119","121",
            "122","123","124","200","201","202","203","205","207","208",
            "209","210","212","213","214","215","217","219","220","221",
            "222","223","228","230","231","232","233","234"
        ]
    for rid in record_ids:
        dat = DATA_DIR / f"{rid}.dat"
        hea = DATA_DIR / f"{rid}.hea"
        atr = DATA_DIR / f"{rid}.atr"
        if not (dat.exists() and hea.exists() and atr.exists()):
            wfdb.dl_database("mitdb", dl_dir=str(DATA_DIR), records=[rid])
    return record_ids

def extract_beats(record_ids=None, window=187, max_per_class=15000, seed=42):
    rng = np.random.default_rng(seed)
    record_ids = ensure_records(record_ids)

    X, y, groups = [], [], []
    half = window // 2

    for rid in record_ids:
        record = wfdb.rdrecord(str(DATA_DIR / rid), channels=[0])
        ann = wfdb.rdann(str(DATA_DIR / rid), "atr")
        signal = record.p_signal[:, 0].astype(np.float32)

        for sample, symbol in zip(ann.sample, ann.symbol):
            if symbol not in VALID_SYMBOLS:
                continue
            left, right = sample - half, sample + half
            if left < 0 or right > len(signal):
                continue

            beat = signal[left:right]
            if len(beat) != window:
                continue

            label = 0 if symbol in NORMAL_SYMBOLS else 1
            X.append(beat)
            y.append(label)
            groups.append(rid)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)

    # Cap each class to keep demo training practical.
    selected = []
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.append(idx)
    selected = np.concatenate(selected)
    rng.shuffle(selected)

    X, y, groups = X[selected], y[selected], groups[selected]
   
    # Per-beat normalization keeps morphology while reducing amplitude offsets.
    print("DEBUG X shape:", X.shape)
    print("DEBUG X dtype:", X.dtype)
    print("DEBUG y shape:", y.shape)
    print("DEBUG number of beats:", len(X))
    X = X - X.mean(axis=1, keepdims=True)
    X = X / (X.std(axis=1, keepdims=True) + 1e-7)
    return X[..., None], y, groups
