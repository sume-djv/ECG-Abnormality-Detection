from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from src.data import extract_beats, ARTIFACT_DIR
from src.model import build_model

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading/loading MIT-BIH and extracting beats...")
X, y, groups = extract_beats(seed=SEED)

# Record-aware split prevents beats from the same ECG record leaking across splits.
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
train_idx, test_idx = next(gss1.split(X, y, groups=groups))

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
tr2, val2 = next(gss2.split(X_train, y_train, groups=groups[train_idx]))
X_val, y_val = X_train[val2], y_train[val2]
X_train, y_train = X_train[tr2], y_train[tr2]

print("Shapes:", X_train.shape, X_val.shape, X_test.shape)
print("Class counts:", np.bincount(y_train))

model = build_model(X_train.shape[1:])
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
]

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=128,
    callbacks=callbacks,
    verbose=1,
)

prob = model.predict(X_test, verbose=0).ravel()
pred = (prob >= 0.5).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, prob)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "classification_report": classification_report(
        y_test, pred, target_names=["Normal", "Abnormal"], zero_division=0
    )
}

print("\n=== TEST METRICS ===")
for k, v in metrics.items():
    print(k, ":", v)

model.save(ARTIFACT_DIR / "ecg_cnn.keras")
with open(ARTIFACT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Keep a small set for the dashboard without storing the full dataset.
rng = np.random.default_rng(SEED)
n = min(500, len(X_test))
idx = rng.choice(len(X_test), size=n, replace=False)
np.savez_compressed(
    ARTIFACT_DIR / "test_samples.npz",
    X=X_test[idx],
    y=y_test[idx],
    prob=prob[idx],
)

print("\nSaved artifacts to:", ARTIFACT_DIR.resolve())
