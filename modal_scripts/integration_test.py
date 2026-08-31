"""
Scaled Modal integration test for RLVR-Sci (~$3–5 on L4).

Exercises dataset, surrogate, all reward paths (including conditional gap),
medium Qwen GRPO, and short Mistral-7B 4-bit GRPO before a full $30 run.

Run from repo root:
  modal run --detach modal_scripts/integration_test.py --skip-surrogate
  modal run modal_scripts/integration_test.py --wait --skip-surrogate  # block locally
"""

import modal

from modal_scripts.config import (
    DATA_MOUNT,
    HF_CACHE_MOUNT,
    HF_VOLUME_NAME,
    INTEGRATION_APP_NAME,
    INTEGRATION_MISTRAL_OUTPUT_REL,
    INTEGRATION_MISTRAL_PROMPTS,
    INTEGRATION_MISTRAL_STEPS,
    INTEGRATION_QWEN_OUTPUT_REL,
    INTEGRATION_QWEN_PROMPTS,
    INTEGRATION_QWEN_STEPS,
    PRODUCTION_MODEL,
    QM9_CSV_REL,
    QM9_FINGERPRINTS_REL,
    QM9_MANIFEST_REL,
    SMOKE_MIN_R2,
    SMOKE_TEST_MODEL,
    SURROGATE_MODEL_REL,
    SURROGATE_META_REL,
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
        "datasets==3.2.0",
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

app = modal.App(INTEGRATION_APP_NAME)


def _paths() -> dict[str, str]:
    return {
        "csv": f"{DATA_MOUNT}/{QM9_CSV_REL}",
        "fingerprints": f"{DATA_MOUNT}/{QM9_FINGERPRINTS_REL}",
        "manifest": f"{DATA_MOUNT}/{QM9_MANIFEST_REL}",
        "surrogate": f"{DATA_MOUNT}/{SURROGATE_MODEL_REL}",
        "surrogate_meta": f"{DATA_MOUNT}/{SURROGATE_META_REL}",
        "qwen_output": f"{DATA_MOUNT}/{INTEGRATION_QWEN_OUTPUT_REL}",
        "mistral_output": f"{DATA_MOUNT}/{INTEGRATION_MISTRAL_OUTPUT_REL}",
    }


def _has_lora_checkpoints(output_dir: str) -> bool:
    import os

    if not os.path.isdir(output_dir):
        return False
    names = os.listdir(output_dir)
    return any(name.endswith(".safetensors") for name in names) and "adapter_config.json" in names


@app.function(image=cpu_image, volumes=volume_mounts, timeout=600, cpu=1)
def verify_dataset() -> dict:
    import json
    import os

    import numpy as np
    import pandas as pd

    paths = _paths()
    for name in ("csv", "fingerprints", "manifest"):
        if not os.path.exists(paths[name]):
            raise FileNotFoundError(f"Missing volume file: {paths[name]}")

    with open(paths["manifest"]) as f:
        manifest = json.load(f)

    df = pd.read_csv(paths["csv"])
    fp = np.load(paths["fingerprints"])
    if len(df) != fp.shape[0]:
        raise ValueError("CSV row count does not match fingerprint matrix rows.")

    gap = df["gap"]
    return {
        "step": "verify_dataset",
        "passed": True,
        "csv_rows": int(len(df)),
        "fingerprint_dims": list(fp.shape),
        "gap_min": float(gap.min()),
        "gap_max": float(gap.max()),
        "gap_mean": float(gap.mean()),
        "manifest_num_molecules": manifest.get("num_molecules"),
    }


@app.function(
    image=cpu_image,
    volumes=volume_mounts,
    timeout=60 * 60 * 2,
    cpu=4,
    memory=16384,
)
def ensure_surrogate(force: bool = False) -> dict:
    from modal_scripts.surrogate_processing import train_surrogate_on_volume

    meta = train_surrogate_on_volume(
        data_mount=DATA_MOUNT,
        min_r2=SMOKE_MIN_R2,
        force=force,
    )
    data_volume.commit()
    return {"step": "ensure_surrogate", "passed": True, **meta}


@app.function(image=cpu_image, volumes=volume_mounts, timeout=600, cpu=1)
def verify_rewards_extended() -> dict:
    """All five reward functions + conditional gap with dataset-style target_gap batches."""
    from src.prompt_dataset import prepare_prompts_dataset
    from src.reward_functions import (
        create_conditional_gap_reward_func,
        create_novelty_reward_func,
        create_reward_functions,
        create_surrogate_reward_func,
        diversity_reward_func,
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
        "<smiles>C</smiles>",
        "",
    ]
    validity = validity_reward_func(completions=completions)
    if validity != [1.0, -1.0, 1.0, 1.0, -1.0]:
        raise ValueError(f"Validity mismatch: {validity}")

    surrogate_fn = create_surrogate_reward_func(paths["surrogate"])
    surrogate_scores = surrogate_fn(completions=completions)
    if surrogate_scores[0] <= surrogate_scores[1]:
        raise ValueError(f"Surrogate should prefer valid SMILES: {surrogate_scores}")

    novelty_fn = create_novelty_reward_func(paths["fingerprints"])
    novelty_scores = novelty_fn(completions=completions)
    if len(novelty_scores) != len(completions):
        raise ValueError("Novelty reward length mismatch")

    prompt = [{"role": "user", "content": "test prompt"}]
    diversity_scores = diversity_reward_func(
        prompts=[prompt] * 3,
        completions=["<smiles>C</smiles>", "<smiles>CCO</smiles>", "<smiles>CCC</smiles>"],
    )
    if not all(s > 0 for s in diversity_scores):
        raise ValueError(f"Diversity rewards should be positive: {diversity_scores}")

    cond_fn = create_conditional_gap_reward_func(paths["surrogate"])
    cond_positive = cond_fn(
        completions=["<smiles>CCO</smiles>"],
        target_gap=[0.05],
    )
    if cond_positive[0] is None:
        raise ValueError("Conditional reward should apply when target_gap >= 0")
    cond_skipped = cond_fn(
        completions=["<smiles>CCO</smiles>"],
        target_gap=[-1.0],
    )
    if cond_skipped[0] is not None:
        raise ValueError("Conditional reward should be None for unconditional prompts")

    train_ds, eval_ds = prepare_prompts_dataset(
        qm9_path=paths["csv"],
        num_prompts=80,
        seed=42,
    )
    if "target_gap" not in train_ds.column_names:
        raise ValueError("train_dataset missing target_gap column")

    sample = train_ds.select(range(12))
    target_gaps = list(sample["target_gap"])
    batch_completions = ["<smiles>CCO</smiles>"] * len(target_gaps)
    batch_cond = cond_fn(completions=batch_completions, target_gap=target_gaps)
    if len(batch_cond) != len(target_gaps):
        raise ValueError(
            f"Conditional reward length {len(batch_cond)} != batch size {len(target_gaps)}"
        )

    conditional_count = sum(1 for t in target_gaps if float(t) >= 0)
    applied_count = sum(1 for r in batch_cond if r is not None)
    for idx, (target, reward) in enumerate(zip(target_gaps, batch_cond)):
        if float(target) >= 0 and reward is None:
            raise ValueError(
                f"Conditional reward None for conditional prompt at index {idx}, "
                f"target_gap={target}"
            )
    if applied_count != conditional_count:
        raise ValueError(
            f"Conditional reward applied {applied_count} times, expected {conditional_count}"
        )

    eval_sample = eval_ds.select(range(min(8, len(eval_ds))))
    eval_targets = list(eval_sample["target_gap"])
    eval_cond = cond_fn(
        completions=["<smiles>C</smiles>"] * len(eval_targets),
        target_gap=eval_targets,
    )
    eval_conditional = sum(1 for t in eval_targets if float(t) >= 0)
    eval_applied = sum(1 for r in eval_cond if r is not None)
    if eval_applied != eval_conditional:
        raise ValueError("Eval batch conditional reward count mismatch")

    return {
        "step": "verify_rewards_extended",
        "passed": True,
        "num_reward_functions": len(reward_funcs),
        "reward_weights": weights,
        "train_prompts": len(train_ds),
        "eval_prompts": len(eval_ds),
        "conditional_in_batch": conditional_count,
        "conditional_rewards_applied": applied_count,
        "eval_conditional_applied": eval_applied,
    }


@app.function(image=cpu_image, volumes=volume_mounts, timeout=600, cpu=2)
def verify_surrogate_inference() -> dict:
    """Batch surrogate predictions and fingerprint bank sanity."""
    import numpy as np

    from src.surrogate_verifier import SurrogateVerifier

    paths = _paths()
    fp = np.load(paths["fingerprints"])
    verifier = SurrogateVerifier(paths["surrogate"])

    smiles = ["C", "CCO", "c1ccccc1", "CC(=O)O", "invalid"]
    preds = verifier.predict(smiles[:4])
    normed = verifier.predict_normalized(smiles[:4])

    if len(preds) != 4 or len(normed) != 4:
        raise ValueError("Surrogate batch predict length mismatch")
    if not np.all(np.isfinite(preds)) or not np.all(np.isfinite(normed)):
        raise ValueError(f"Non-finite surrogate outputs: preds={preds}, normed={normed}")

    gap_min = float(verifier.meta["gap_min"])
    gap_max = float(verifier.meta["gap_max"])
    margin = 0.5
    if not all(gap_min - margin <= p <= gap_max + margin for p in preds):
        raise ValueError(f"Raw predictions outside expected gap range: {preds}")
    if fp.shape[1] != 2048:
        raise ValueError(f"Expected 2048-bit fingerprints, got {fp.shape[1]}")

    return {
        "step": "verify_surrogate_inference",
        "passed": True,
        "sample_preds": [float(p) for p in preds],
        "sample_normed": [float(n) for n in normed],
        "fingerprint_shape": list(fp.shape),
    }


def _run_grpo(
    model_name: str,
    output_dir: str,
    max_steps: int,
    num_prompts: int,
    seed: int,
    use_4bit: bool,
    step_name: str,
) -> dict:
    import os
    import sys

    sys.path.insert(0, "/root")

    from src.grpo_rlvr import run_grpo_training

    paths = _paths()
    os.makedirs(output_dir, exist_ok=True)

    run_grpo_training(
        model_name=model_name,
        output_dir=output_dir,
        surrogate_model_path=paths["surrogate"],
        qm9_fingerprints_path=paths["fingerprints"],
        qm9_path=paths["csv"],
        max_steps=max_steps,
        num_prompts=num_prompts,
        min_r2=SMOKE_MIN_R2,
        seed=seed,
        use_4bit=use_4bit,
    )

    checkpoint_files = [
        f
        for f in os.listdir(output_dir)
        if f.endswith(".safetensors") or f == "adapter_config.json"
    ]
    if not checkpoint_files:
        raise FileNotFoundError(f"No LoRA checkpoints in {output_dir}")

    return {
        "step": step_name,
        "passed": True,
        "model": model_name,
        "max_steps": max_steps,
        "num_prompts": num_prompts,
        "use_4bit": use_4bit,
        "output_dir": output_dir,
        "checkpoint_files": checkpoint_files,
    }


@app.function(
    image=gpu_image,
    volumes=volume_mounts,
    gpu="L4",
    timeout=60 * 120,
    memory=16384,
    env=hf_env,
)
def run_grpo_qwen_scaled(seed: int = 42) -> dict:
    paths = _paths()
    result = _run_grpo(
        model_name=SMOKE_TEST_MODEL,
        output_dir=paths["qwen_output"],
        max_steps=INTEGRATION_QWEN_STEPS,
        num_prompts=INTEGRATION_QWEN_PROMPTS,
        seed=seed,
        use_4bit=True,
        step_name="run_grpo_qwen_scaled",
    )
    data_volume.commit()
    return result


@app.function(
    image=gpu_image,
    volumes=volume_mounts,
    gpu="L4",
    timeout=60 * 120,
    memory=24576,
    env=hf_env,
)
def run_grpo_mistral_smoke(seed: int = 42) -> dict:
    paths = _paths()
    result = _run_grpo(
        model_name=PRODUCTION_MODEL,
        output_dir=paths["mistral_output"],
        max_steps=INTEGRATION_MISTRAL_STEPS,
        num_prompts=INTEGRATION_MISTRAL_PROMPTS,
        seed=seed,
        use_4bit=True,
        step_name="run_grpo_mistral_smoke",
    )
    data_volume.commit()
    return result


@app.function(
    image=cpu_image,
    volumes=volume_mounts,
    timeout=60 * 240,
    cpu=1,
    memory=4096,
)
def run_integration_pipeline(
    skip_surrogate: bool = False,
    force_surrogate: bool = False,
    skip_mistral: bool = False,
    seed: int = 42,
) -> dict:
    """
    Run the full integration test on Modal (orchestrator stays in the cloud).
    Safe to disconnect local machine once spawned.
    """
    paths = _paths()
    results: list[dict] = []

    print("=" * 60)
    print("RLVR-Sci integration pipeline (cloud orchestrator)")
    print("=" * 60)

    print("\n[1] Verifying dataset...")
    r1 = verify_dataset.local()
    results.append(r1)
    print(f"  OK — {r1['csv_rows']} molecules")

    print("\n[2] Ensuring surrogate...")
    r2 = ensure_surrogate.local(force=force_surrogate)
    results.append(r2)
    print(f"  OK — R2={r2['r2']:.4f}, status={r2.get('status', 'trained')}")

    print("\n[3] Extended reward checks...")
    r3 = verify_rewards_extended.local()
    results.append(r3)
    print("  OK")

    print("\n[4] Surrogate inference...")
    r4 = verify_surrogate_inference.local()
    results.append(r4)
    print("  OK")

    if _has_lora_checkpoints(paths["qwen_output"]):
        print("\n[5] Qwen GRPO — skipped (checkpoints already on volume)")
        results.append(
            {
                "step": "run_grpo_qwen_scaled",
                "passed": True,
                "status": "cached",
                "output_dir": paths["qwen_output"],
            }
        )
    else:
        print(f"\n[5] Qwen GRPO ({INTEGRATION_QWEN_STEPS} steps)...")
        r5 = run_grpo_qwen_scaled.remote(seed=seed)
        results.append(r5)
        print(f"  OK — {r5['checkpoint_files']}")

    if not skip_mistral:
        if _has_lora_checkpoints(paths["mistral_output"]):
            print("\n[6] Mistral GRPO — skipped (checkpoints already on volume)")
            results.append(
                {
                    "step": "run_grpo_mistral_smoke",
                    "passed": True,
                    "status": "cached",
                    "output_dir": paths["mistral_output"],
                }
            )
        else:
            print(f"\n[6] Mistral-7B GRPO ({INTEGRATION_MISTRAL_STEPS} steps)...")
            r6 = run_grpo_mistral_smoke.remote(seed=seed)
            results.append(r6)
            print(f"  OK — {r6['checkpoint_files']}")

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 60)
    for r in results:
        print(f"  {r['step']}: OK")

    return {"passed": True, "results": results}


@app.local_entrypoint()
def main(
    skip_surrogate: bool = False,
    force_surrogate: bool = False,
    skip_mistral: bool = False,
    seed: int = 42,
    wait: bool = False,
):
    print("=" * 60)
    print("RLVR-Sci Modal integration test (scaled)")
    print(f"Volume: {VOLUME_NAME}")
    print(
        f"Plan: Qwen {INTEGRATION_QWEN_STEPS} steps / {INTEGRATION_QWEN_PROMPTS} prompts"
        + (
            f"; Mistral-7B {INTEGRATION_MISTRAL_STEPS} steps / "
            f"{INTEGRATION_MISTRAL_PROMPTS} prompts"
            if not skip_mistral
            else "; Mistral skipped"
        )
    )
    print("Estimated cost: ~$3–5 on L4")
    print("=" * 60)

    pipeline_kwargs = {
        "skip_surrogate": skip_surrogate,
        "force_surrogate": force_surrogate,
        "skip_mistral": skip_mistral,
        "seed": seed,
    }

    if wait:
        print("\nWaiting for cloud pipeline (local process must stay connected)...")
        result = run_integration_pipeline.remote(**pipeline_kwargs)
        print("\nPipeline finished:", result["passed"])
        return

    function_call = run_integration_pipeline.spawn(**pipeline_kwargs)
    print("\nStarted integration pipeline on Modal (cloud orchestrator).")
    print("You can close your laptop or disconnect — the job runs on Modal only.")
    print("Tip: also pass `modal run --detach` so the CLI survives disconnects.")
    print(f"\nFunction call id: {function_call.object_id}")
    print("Track: modal app list")
    print("Logs: modal app logs rlvr-sci-integration-test")
