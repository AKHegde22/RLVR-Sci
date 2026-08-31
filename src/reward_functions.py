"""GRPO reward functions for molecule generation RLVR."""

from collections import defaultdict
from typing import Any, Callable

import numpy as np

from src.chemistry import (
    max_tanimoto_similarity,
    morgan_fingerprint,
    tanimoto_similarity,
)
from src.smiles_utils import extract_smiles_from_completion, prompt_to_key
from src.surrogate_verifier import SurrogateVerifier, check_surrogate_quality


RewardFunc = Callable[..., list[float]]


def validity_reward_func(completions, **kwargs) -> list[float]:
    """Reward chemically valid SMILES extracted from model output."""
    rewards = []
    for completion in completions:
        smiles = extract_smiles_from_completion(completion)
        rewards.append(1.0 if smiles is not None else -1.0)
    return rewards


def create_surrogate_reward_func(
    model_path: str = "data/surrogate_gap_model.joblib",
) -> RewardFunc:
    verifier = SurrogateVerifier(model_path)

    def surrogate_reward_func(completions, **kwargs) -> list[float]:
        smiles_list = [extract_smiles_from_completion(c) for c in completions]
        normalized = verifier.predict_normalized(
            [s if s is not None else "C" for s in smiles_list]
        )
        rewards = []
        for smiles, score in zip(smiles_list, normalized):
            rewards.append(float(score) if smiles is not None else 0.0)
        return rewards

    return surrogate_reward_func


def create_novelty_reward_func(
    fingerprints_path: str = "data/qm9_fingerprints.npy",
    similarity_threshold: float = 0.85,
) -> RewardFunc:
    qm9_fps = np.load(fingerprints_path)

    def novelty_reward_func(completions, **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            smiles = extract_smiles_from_completion(completion)
            if smiles is None:
                rewards.append(0.0)
                continue
            fp = morgan_fingerprint(smiles)
            if fp is None:
                rewards.append(0.0)
                continue
            max_sim = max_tanimoto_similarity(fp, qm9_fps)
            if max_sim >= similarity_threshold:
                rewards.append(-1.0)
            else:
                rewards.append(float(1.0 - max_sim))
        return rewards

    return novelty_reward_func


def diversity_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Within each prompt group, reward completions that are structurally diverse.
    GRPO passes the same prompt for each generation in a group.
    """
    groups: dict[str, list[tuple[int, str | None]]] = defaultdict(list)
    for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
        groups[prompt_to_key(prompt)].append((idx, extract_smiles_from_completion(completion)))

    rewards = [0.0] * len(completions)
    for members in groups.values():
        indexed_fps: list[tuple[int, np.ndarray]] = []
        for idx, smiles in members:
            if smiles is None:
                rewards[idx] = 0.0
                continue
            fp = morgan_fingerprint(smiles)
            if fp is not None:
                indexed_fps.append((idx, fp))

        if len(indexed_fps) <= 1:
            for idx, _ in indexed_fps:
                rewards[idx] = 1.0
            continue

        for idx, fp in indexed_fps:
            other_fps = [other_fp for other_idx, other_fp in indexed_fps if other_idx != idx]
            distances = [1.0 - tanimoto_similarity(fp, other_fp) for other_fp in other_fps]
            rewards[idx] = float(np.mean(distances))

    return rewards


def create_conditional_gap_reward_func(
    model_path: str = "data/surrogate_gap_model.joblib",
) -> RewardFunc:
    verifier = SurrogateVerifier(model_path)

    def conditional_gap_reward_func(completions, target_gap=None, **kwargs) -> list[float | None]:
        if target_gap is None:
            return [None for _ in completions]

        rewards: list[float | None] = []
        for completion, target in zip(completions, target_gap):
            target_value = float(target)
            if target_value < 0:
                rewards.append(None)
                continue
            smiles = extract_smiles_from_completion(completion)
            if smiles is None:
                rewards.append(0.0)
                continue
            pred = float(verifier.predict([smiles])[0])
            if pred >= target_value:
                rewards.append(1.0)
            else:
                gap_error = abs(pred - target_value)
                rewards.append(max(0.0, 1.0 - gap_error / max(target_value, 0.05)))
        return rewards

    return conditional_gap_reward_func


def create_reward_functions(
    surrogate_model_path: str = "data/surrogate_gap_model.joblib",
    qm9_fingerprints_path: str = "data/qm9_fingerprints.npy",
    min_r2: float = 0.70,
    novelty_similarity_threshold: float = 0.85,
) -> tuple[list[RewardFunc], list[float]]:
    """Build reward function list and weights after surrogate quality gate."""
    check_surrogate_quality(surrogate_model_path, min_r2=min_r2)

    reward_funcs: list[RewardFunc] = [
        validity_reward_func,
        create_surrogate_reward_func(surrogate_model_path),
        create_novelty_reward_func(qm9_fingerprints_path, novelty_similarity_threshold),
        diversity_reward_func,
        create_conditional_gap_reward_func(surrogate_model_path),
    ]
    reward_weights = [1.0, 1.0, 0.5, 0.3, 0.5]
    return reward_funcs, reward_weights
