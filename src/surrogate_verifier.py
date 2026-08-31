import json
import os
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

from src.chemistry import batch_morgan_fingerprints, FP_N_BITS, FP_RADIUS, morgan_fingerprint

DEFAULT_MIN_R2 = 0.70
META_SUFFIX = "_meta.json"


def _meta_path(model_path: str) -> str:
    return model_path.replace(".joblib", META_SUFFIX)


def compute_fingerprint_matrix(smiles_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Compute fingerprint matrix and boolean mask for valid rows."""
    fps: list[np.ndarray] = []
    valid_mask: list[bool] = []

    for smiles in tqdm(smiles_series, desc="Fingerprints"):
        fp = morgan_fingerprint(str(smiles))
        if fp is not None:
            fps.append(fp)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    if not fps:
        return np.empty((0, FP_N_BITS), dtype=np.uint8), np.array(valid_mask, dtype=bool)

    return np.stack(fps, axis=0), np.array(valid_mask, dtype=bool)


def train_surrogate(
    data_path: str = "data/qm9_processed.csv",
    model_path: str = "data/surrogate_gap_model.joblib",
    min_r2: float = DEFAULT_MIN_R2,
    random_state: int = 42,
) -> dict[str, float]:
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print("Computing fingerprints...")
    smiles = df["smiles"].astype(str)
    fp_matrix, valid_mask = compute_fingerprint_matrix(smiles)
    gaps = df.loc[valid_mask, "gap"].to_numpy()

    print(f"Dataset size: {fp_matrix.shape[0]} valid molecules")

    X_train, X_test, y_train, y_test = train_test_split(
        fp_matrix, gaps, test_size=0.1, random_state=random_state
    )

    print("Training MLP Regressor...")
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

    print(f"Test MSE: {mse:.6f}")
    print(f"Test R2 Score: {r2:.4f}")

    if r2 < min_r2:
        raise ValueError(
            f"Surrogate R2 {r2:.4f} below minimum {min_r2:.2f}. "
            "Do not run RLVR until the surrogate is accurate enough."
        )

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(model, model_path)

    meta = {
        "r2": float(r2),
        "mse": float(mse),
        "gap_mean": float(gaps.mean()),
        "gap_std": float(gaps.std()),
        "gap_min": float(gaps.min()),
        "gap_max": float(gaps.max()),
        "min_r2_threshold": float(min_r2),
        "fp_radius": FP_RADIUS,
        "fp_n_bits": FP_N_BITS,
        "random_state": random_state,
    }
    meta_path = _meta_path(model_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {meta_path}")
    return meta


def load_surrogate_meta(model_path: str = "data/surrogate_gap_model.joblib") -> dict[str, Any]:
    meta_path = _meta_path(model_path)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Surrogate metadata not found at {meta_path}. Retrain with train_surrogate()."
        )
    with open(meta_path) as f:
        return json.load(f)


def check_surrogate_quality(
    model_path: str = "data/surrogate_gap_model.joblib",
    min_r2: float = DEFAULT_MIN_R2,
) -> dict[str, Any]:
    meta = load_surrogate_meta(model_path)
    if meta["r2"] < min_r2:
        raise ValueError(
            f"Surrogate R2 {meta['r2']:.4f} below minimum {min_r2:.2f}. Retrain the surrogate."
        )
    return meta


class SurrogateVerifier:
    def __init__(self, model_path: str = "data/surrogate_gap_model.joblib"):
        self.model = joblib.load(model_path)
        self.meta = load_surrogate_meta(model_path)

    def predict(self, smiles_list: list[str]) -> np.ndarray:
        fps = batch_morgan_fingerprints(smiles_list)
        valid_idx = [i for i, fp in enumerate(fps) if fp is not None]

        results = np.zeros(len(smiles_list), dtype=np.float32)
        if valid_idx:
            valid_fps = np.stack([fps[i] for i in valid_idx], axis=0)
            preds = self.model.predict(valid_fps)
            for i, pred in zip(valid_idx, preds):
                results[i] = float(pred)
        return results

    def predict_normalized(self, smiles_list: list[str]) -> np.ndarray:
        preds = self.predict(smiles_list)
        std = max(float(self.meta["gap_std"]), 1e-6)
        mean = float(self.meta["gap_mean"])
        return (preds - mean) / std


if __name__ == "__main__":
    train_surrogate()
