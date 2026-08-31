"""
End-to-end Modal smoke test for RLVR-Sci.

Verifies volume dataset, trains surrogate, tests rewards, runs short GRPO on GPU.

Run from repo root:
  modal run modal_scripts/smoke_test.py
  modal run modal_scripts/smoke_test.py --force-surrogate
"""

import modal

from modal_scripts.config import (
    DATA_MOUNT,
    HF_CACHE_MOUNT,
    HF_VOLUME_NAME,
    QM9_CSV_REL,
    QM9_FINGERPRINTS_REL,
    QM9_MANIFEST_REL,
    SMOKE_APP_NAME,
    SMOKE_MAX_STEPS,
    SMOKE_MIN_R2,
    SMOKE_NUM_PROMPTS,
    SMOKE_OUTPUT_REL,
    SMOKE_TEST_MODEL,
    SURROGATE_MODEL_REL,
    VOLUME_NAME,
)

data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_volume = modal.Volume.from_name(HF_VOLUME_NAME, create_if_missing=True)

volume_mounts = {
    DATA_MOUNT: data_volume,
    HF_CACHE_MOUNT: hf_volume,
}

hf_env = {
    "HF_HOME": HF_CACHE_MOUNT,
    "HF_HUB_CACHE": f"{HF_CACHE_MOUNT}/hub",
    "TRANSFORMERS_CACHE": HF_CACHE_MOUNT,
}

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas==2.2.3",
        "numpy==2.1.3",
        "scikit-learn==1.5.2",
        "joblib==1.4.2",
        "rdkit==2024.9.5",
        "tqdm==4.67.1",
    )
    .add_local_python_source("modal_scripts")
    .add_local_python_source("src")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.49.0",
        "trl==0.16.1",
        "peft==0.14.0",
        "accelerate==1.2.1",
        "datasets==3.2.0",
        "bitsandbytes==0.45.0",
        "pandas==2.2.3",
        "numpy==2.1.3",
        "scikit-learn==1.5.2",
        "joblib==1.4.2",
        "rdkit==2024.9.5",
        "sentencepiece==0.2.0",
        "protobuf==5.29.2",
    )
    .add_local_python_source("modal_scripts")
    .add_local_python_source("src")
)

app = modal.App(SMOKE_APP_NAME)


def _paths() -> dict[str, str]:
    return {
        "csv": f"{DATA_MOUNT}/{QM9_CSV_REL}",
        "fingerprints": f"{DATA_MOUNT}/{QM9_FINGERPRINTS_REL}",
        "manifest": f"{DATA_MOUNT}/{QM9_MANIFEST_REL}",
        "surrogate": f"{DATA_MOUNT}/{SURROGATE_MODEL_REL}",
        "grpo_output": f"{DATA_MOUNT}/{SMOKE_OUTPUT_REL}",
    }


@app.function(
    image=cpu_image,
    volumes=volume_mounts,
    timeout=600,
    cpu=1,
)
def verify_dataset() -> dict:
    """Step 1: Confirm volume dataset files and row counts."""
    import json
    import os

    import numpy as np
    import pandas as pd

    paths = _paths()
    required = ("csv", "fingerprints", "manifest")
    missing = [name for name in required if not os.path.exists(paths[name])]
    if missing:
        raise FileNotFoundError(f"Missing volume files: {missing}")

    with open(paths["manifest"]) as f:
        manifest = json.load(f)

    df = pd.read_csv(paths["csv"])
    fp = np.load(paths["fingerprints"])

    result = {
        "step": "verify_dataset",
        "passed": True,
        "csv_rows": int(len(df)),
        "fingerprint_rows": int(fp.shape[0]),
        "fingerprint_dims": list(fp.shape),
        "manifest_num_molecules": manifest.get("num_molecules"),
        "rows_match": len(df) == fp.shape[0],
    }
    if not result["rows_match"]:
        result["passed"] = False
        raise ValueError("CSV row count does not match fingerprint matrix rows.")
    return result


@app.function(
    image=cpu_image,
    volumes=volume_mounts,
    timeout=60 * 60 * 2,
    cpu=4,
    memory=16384,
)
def train_surrogate(force: bool = False) -> dict:
    """Step 2: Train surrogate MLP on full volume dataset (uses precomputed fingerprints)."""
    from modal_scripts.surrogate_processing import train_surrogate_on_volume

    meta = train_surrogate_on_volume(
        data_mount=DATA_MOUNT,
        min_r2=SMOKE_MIN_R2,
        force=force,
    )
    data_volume.commit()
    return {
        "step": "train_surrogate",
        "passed": True,
        **meta,
    }


@app.function(
    image=cpu_image,
    volumes=volume_mounts,
    timeout=600,
    cpu=1,
)
def ensure_surrogate_cached() -> dict:
    """Use existing surrogate on volume; fail if missing or below R2 gate."""
    from modal_scripts.surrogate_processing import train_surrogate_on_volume

    meta = train_surrogate_on_volume(
        data_mount=DATA_MOUNT,
        min_r2=SMOKE_MIN_R2,
        force=False,
    )
    return {
        "step": "ensure_surrogate_cached",
        "passed": True,
        **meta,
    }


@app.function(
    image=cpu_image,
    volumes=volume_mounts,
    timeout=600,
    cpu=1,
)
def verify_rewards() -> dict:
    """Step 3: Smoke test reward functions against volume artifacts."""
    from src.reward_functions import (
        create_reward_functions,
        validity_reward_func,
    )

    paths = _paths()
    reward_funcs, weights = create_reward_functions(
        surrogate_model_path=paths["surrogate"],
        qm9_fingerprints_path=paths["fingerprints"],
        min_r2=SMOKE_MIN_R2,
    )

    completions = [
        "<smiles>CCO</smiles>",
        "invalid molecule text",
        "The molecule is <smiles>c1ccccc1</smiles>",
    ]
    validity = validity_reward_func(completions=completions)
    expected_validity = [1.0, -1.0, 1.0]

    if validity != expected_validity:
        raise ValueError(f"Validity rewards mismatch: {validity} != {expected_validity}")

    surrogate_fn = reward_funcs[1]
    surrogate_scores = surrogate_fn(completions=completions)
    if surrogate_scores[0] <= surrogate_scores[1]:
        raise ValueError(f"Surrogate should prefer valid SMILES: {surrogate_scores}")

    return {
        "step": "verify_rewards",
        "passed": True,
        "validity_rewards": validity,
        "surrogate_rewards": surrogate_scores,
        "num_reward_functions": len(reward_funcs),
        "reward_weights": weights,
    }


@app.function(
    image=gpu_image,
    volumes=volume_mounts,
    gpu="L4",
    timeout=60 * 60,
    memory=16384,
    env=hf_env,
)
def run_grpo_smoke(seed: int = 42) -> dict:
    """Step 4: Short GRPO run on GPU with a small instruct model."""
    import os
    import sys

    sys.path.insert(0, "/root")

    from src.grpo_rlvr import run_grpo_training

    paths = _paths()
    os.makedirs(paths["grpo_output"], exist_ok=True)

    run_grpo_training(
        model_name=SMOKE_TEST_MODEL,
        output_dir=paths["grpo_output"],
        surrogate_model_path=paths["surrogate"],
        qm9_fingerprints_path=paths["fingerprints"],
        qm9_path=paths["csv"],
        max_steps=SMOKE_MAX_STEPS,
        num_prompts=SMOKE_NUM_PROMPTS,
        min_r2=SMOKE_MIN_R2,
        seed=seed,
        use_4bit=True,
    )

    data_volume.commit()
    checkpoint_files = [
        f
        for f in os.listdir(paths["grpo_output"])
        if f.endswith(".safetensors") or f == "adapter_config.json"
    ]
    return {
        "step": "run_grpo_smoke",
        "passed": True,
        "model": SMOKE_TEST_MODEL,
        "max_steps": SMOKE_MAX_STEPS,
        "output_dir": paths["grpo_output"],
        "checkpoint_files": checkpoint_files,
    }


@app.local_entrypoint()
def main(force_surrogate: bool = False, skip_surrogate: bool = False, seed: int = 42):
    print("=" * 60)
    print("RLVR-Sci Modal smoke test")
    print(f"Volume: {VOLUME_NAME} | Profile: check with `modal profile current`")
    print("=" * 60)

    results = []

    print("\n[1/4] Verifying dataset on volume...")
    r1 = verify_dataset.remote()
    results.append(r1)
    print(f"  PASSED — {r1['csv_rows']} molecules, fingerprints {r1['fingerprint_dims']}")

    if skip_surrogate:
        print("\n[2/4] Checking surrogate (skip training, use cache)...")
        r2 = ensure_surrogate_cached.remote()
    else:
        print("\n[2/4] Training surrogate on volume data...")
        r2 = train_surrogate.remote(force=force_surrogate)
    results.append(r2)
    print(f"  PASSED — R2={r2['r2']:.4f}, status={r2.get('status', 'trained')}")

    print("\n[3/4] Verifying reward functions...")
    r3 = verify_rewards.remote()
    results.append(r3)
    print(f"  PASSED — validity={r3['validity_rewards']}")

    print("\n[4/4] Running GRPO smoke test on GPU (this may take several minutes)...")
    r4 = run_grpo_smoke.remote(seed=seed)
    results.append(r4)
    print(f"  PASSED — checkpoints: {r4['checkpoint_files']}")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
    for r in results:
        print(f"  {r['step']}: OK")
