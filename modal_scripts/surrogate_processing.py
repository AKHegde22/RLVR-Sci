"""Train surrogate MLP on volume data using precomputed fingerprints."""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

FP_RADIUS = 2
FP_N_BITS = 2048
META_SUFFIX = "_meta.json"


def _meta_path(model_path: str) -> str:
    return model_path.replace(".joblib", META_SUFFIX)


def train_surrogate_on_volume(
    data_mount: str,
    min_r2: float = 0.70,
    random_state: int = 42,
    force: bool = False,
) -> dict:
    """
    Train surrogate from qm9_processed.csv + qm9_fingerprints.npy on the volume.
    Skips training if a model exists and passes the R2 gate (unless force=True).
    """
    csv_path = os.path.join(data_mount, "qm9_processed.csv")
    fp_path = os.path.join(data_mount, "qm9_fingerprints.npy")
    model_path = os.path.join(data_mount, "surrogate_gap_model.joblib")
    meta_path = _meta_path(model_path)

    if not force and os.path.exists(model_path) and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("r2", 0) >= min_r2:
            meta["status"] = "cached"
            return meta

    df = pd.read_csv(csv_path)
    X = np.load(fp_path)
    y = df["gap"].to_numpy(dtype=np.float64)

    if X.shape[0] != len(y):
        raise ValueError(
            f"Fingerprint rows ({X.shape[0]}) != CSV rows ({len(y)}). Re-run dataset prep."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=random_state
    )

    model = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        max_iter=200,
        early_stopping=True,
        random_state=random_state,
        verbose=True,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if r2 < min_r2:
        raise ValueError(f"Surrogate R2 {r2:.4f} below minimum {min_r2:.2f}")

    joblib.dump(model, model_path)
    meta = {
        "status": "trained",
        "r2": float(r2),
        "mse": float(mse),
        "gap_mean": float(y.mean()),
        "gap_std": float(y.std()),
        "gap_min": float(y.min()),
        "gap_max": float(y.max()),
        "min_r2_threshold": float(min_r2),
        "fp_radius": FP_RADIUS,
        "fp_n_bits": FP_N_BITS,
        "random_state": random_state,
        "train_rows": int(len(y)),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta
