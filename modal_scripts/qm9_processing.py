"""QM9 download and processing logic for Modal jobs (no imports from src/)."""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from modal_scripts.config import QM9_SOURCE_URL

FP_RADIUS = 2
FP_N_BITS = 2048


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def morgan_fingerprint(smiles: str) -> np.ndarray | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_N_BITS)
        return np.array(fp, dtype=np.uint8)
    except (ValueError, RuntimeError, Chem.KekulizeException):
        return None


def compute_fingerprint_matrix(smiles_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
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


def prepare_qm9_files(
    data_dir: str,
    force: bool = False,
) -> dict:
    """
    Download QM9, canonicalize SMILES, save processed CSV, fingerprint bank, and manifest.
    Returns a summary dict written to dataset_manifest.json.
    """
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "qm9_processed.csv")
    fp_path = os.path.join(data_dir, "qm9_fingerprints.npy")
    manifest_path = os.path.join(data_dir, "dataset_manifest.json")

    if os.path.exists(csv_path) and os.path.exists(fp_path) and not force:
        df = pd.read_csv(csv_path)
        fp_matrix = np.load(fp_path)
        summary = {
            "status": "cached",
            "source_url": QM9_SOURCE_URL,
            "num_molecules": int(len(df)),
            "fingerprint_shape": list(fp_matrix.shape),
            "csv_path": csv_path,
            "fingerprints_path": fp_path,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(manifest_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    print(f"Downloading QM9 from {QM9_SOURCE_URL}...")
    raw_df = pd.read_csv(QM9_SOURCE_URL)
    raw_df = raw_df[["smiles", "gap"]].dropna()

    canonical_smiles: list[str] = []
    gaps: list[float] = []
    for smiles, gap in tqdm(
        zip(raw_df["smiles"], raw_df["gap"]),
        total=len(raw_df),
        desc="Canonicalizing",
    ):
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            canonical_smiles.append(canonical)
            gaps.append(float(gap))

    df = pd.DataFrame({"smiles": canonical_smiles, "gap": gaps})
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} molecules to {csv_path}")

    print("Computing fingerprint bank for novelty rewards...")
    fp_matrix, valid_mask = compute_fingerprint_matrix(df["smiles"])
    np.save(fp_path, fp_matrix)
    print(f"Saved fingerprint matrix {fp_matrix.shape} to {fp_path}")

    invalid_count = int(valid_mask.size - valid_mask.sum())
    summary = {
        "status": "created",
        "source_url": QM9_SOURCE_URL,
        "source_rows": int(len(raw_df)),
        "num_molecules": int(len(df)),
        "fingerprint_rows": int(fp_matrix.shape[0]),
        "fingerprint_shape": list(fp_matrix.shape),
        "invalid_fingerprint_rows": invalid_count,
        "gap_min": float(df["gap"].min()),
        "gap_max": float(df["gap"].max()),
        "gap_mean": float(df["gap"].mean()),
        "gap_std": float(df["gap"].std()),
        "csv_path": csv_path,
        "fingerprints_path": fp_path,
        "manifest_path": manifest_path,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary
