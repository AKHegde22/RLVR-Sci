import os

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from src.chemistry import morgan_fingerprint
from src.surrogate_verifier import compute_fingerprint_matrix

QM9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"


def _canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def download_and_process_qm9(
    output_path: str = "data/qm9_processed.csv",
    fingerprints_path: str = "data/qm9_fingerprints.npy",
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Download QM9, canonicalize SMILES, save gap labels, and precompute fingerprints for novelty checks.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if os.path.exists(output_path) and not force_download:
        print(f"Using cached dataset at {output_path}")
        df = pd.read_csv(output_path)
    else:
        print(f"Downloading QM9 from {QM9_URL}...")
        df = pd.read_csv(QM9_URL)
        df = df[["smiles", "gap"]].dropna()

        canonical_smiles: list[str | None] = []
        gaps: list[float] = []
        for smiles, gap in tqdm(zip(df["smiles"], df["gap"]), total=len(df), desc="Canonicalizing"):
            canonical = _canonicalize_smiles(smiles)
            if canonical is not None:
                canonical_smiles.append(canonical)
                gaps.append(float(gap))

        df = pd.DataFrame({"smiles": canonical_smiles, "gap": gaps})
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} molecules to {output_path}")

    if os.path.exists(fingerprints_path) and not force_download:
        print(f"Using cached fingerprints at {fingerprints_path}")
        return df

    print("Computing QM9 fingerprint bank for novelty rewards...")
    fp_matrix, valid_mask = compute_fingerprint_matrix(df["smiles"])
    np.save(fingerprints_path, fp_matrix)
    print(f"Saved fingerprint matrix {fp_matrix.shape} to {fingerprints_path}")
    if not valid_mask.all():
        print(
            f"Warning: {valid_mask.size - valid_mask.sum()} rows lacked fingerprints after canonicalization."
        )

    return df


if __name__ == "__main__":
    download_and_process_qm9()
