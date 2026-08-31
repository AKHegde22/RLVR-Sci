"""Chemistry helpers: fingerprints and similarity."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

FP_RADIUS = 2
FP_N_BITS = 2048


def morgan_fingerprint(smiles: str) -> np.ndarray | None:
    """Return Morgan fingerprint bit vector as numpy array, or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_N_BITS)
        return np.array(fp, dtype=np.uint8)
    except (ValueError, RuntimeError, Chem.KekulizeException):
        return None


def batch_morgan_fingerprints(smiles_list: list[str]) -> list[np.ndarray | None]:
    return [morgan_fingerprint(s) for s in smiles_list]


def tanimoto_similarity(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Tanimoto similarity between two bit vectors."""
    intersection = float(np.dot(fp_a, fp_b))
    union = float(fp_a.sum() + fp_b.sum() - intersection)
    if union <= 0:
        return 0.0
    return intersection / union


def max_tanimoto_similarity(query_fp: np.ndarray, fp_matrix: np.ndarray) -> float:
    """Max Tanimoto similarity between query_fp and rows of fp_matrix."""
    if fp_matrix.size == 0:
        return 0.0
    query = query_fp.astype(np.float32)
    matrix = fp_matrix.astype(np.float32)
    dots = matrix @ query
    sums = matrix.sum(axis=1) + query.sum() - dots
    similarities = dots / np.maximum(sums, 1e-8)
    return float(similarities.max())


def mean_pairwise_tanimoto_distance(fingerprints: list[np.ndarray]) -> float:
    """Mean Tanimoto distance (1 - similarity) across unique pairs."""
    if len(fingerprints) < 2:
        return 1.0
    distances = []
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            distances.append(1.0 - tanimoto_similarity(fingerprints[i], fingerprints[j]))
    return float(np.mean(distances))
