import os
import random
from typing import Optional

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.reward_functions import create_reward_functions

MISTRAL_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

CONDITIONAL_PROMPT_TEMPLATE = (
    "Respond with exactly one novel molecule using this format: "
    "<smiles>YOUR_SMILES_HERE</smiles>. "
    "Target HOMO-LUMO gap: at least {target_gap:.3f} Hartree. "
    "Do not include any other text."
)

UNCONDITIONAL_PROMPTS = [
    (
        "Respond with exactly one novel small organic molecule using this format: "
        "<smiles>YOUR_SMILES_HERE</smiles>. Maximize HOMO-LUMO gap. "
        "Do not include any other text."
    ),
    (
        "Output a single valid SMILES for a novel molecule with a large HOMO-LUMO gap. "
        "Format: <smiles>YOUR_SMILES_HERE</smiles> only."
    ),
    (
        "Generate one chemically valid, novel SMILES string wrapped in <smiles></smiles> tags. "
        "Prefer structures with high HOMO-LUMO gap."
    ),
]


def prepare_prompts_dataset(
    qm9_path: str = "data/qm9_processed.csv",
    num_prompts: int = 500,
    eval_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Optional[Dataset]]:
    """
    Build diverse property-conditioned and unconditional prompts from QM9 gaps.
    Returns train and eval datasets with optional target_gap for conditional rewards.
    """
    rng = random.Random(seed)
    df = pd.read_csv(qm9_path)

    if len(df) < num_prompts:
        num_prompts = len(df)

    sampled = df.sample(n=num_prompts, random_state=seed).reset_index(drop=True)

    prompts: list[list[dict[str, str]]] = []
    target_gaps: list[float] = []

    for _, row in sampled.iterrows():
        gap = float(row["gap"])
        if rng.random() < 0.6:
            text = CONDITIONAL_PROMPT_TEMPLATE.format(target_gap=gap)
            target_gaps.append(gap)
        else:
            text = rng.choice(UNCONDITIONAL_PROMPTS)
            target_gaps.append(-1.0)

        prompts.append([{"role": "user", "content": text}])

    dataset = Dataset.from_dict(
        {
            "prompt": prompts,
            "target_gap": target_gaps,
        }
    )

    eval_size = max(1, int(len(dataset) * eval_fraction))
    split = dataset.train_test_split(test_size=eval_size, seed=seed)
    return split["train"], split["test"]


def _build_peft_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=MISTRAL_LORA_TARGET_MODULES,
    )


def run_grpo_training(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    output_dir: str = "outputs/grpo_model",
    surrogate_model_path: str = "data/surrogate_gap_model.joblib",
    qm9_fingerprints_path: str = "data/qm9_fingerprints.npy",
    qm9_path: str = "data/qm9_processed.csv",
    max_steps: int = 500,
    num_prompts: int = 500,
    min_r2: float = 0.70,
    seed: int = 42,
    use_4bit: bool = False,
) -> None:
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    reward_funcs, reward_weights = create_reward_functions(
        surrogate_model_path=surrogate_model_path,
        qm9_fingerprints_path=qm9_fingerprints_path,
        min_r2=min_r2,
    )

    train_dataset, eval_dataset = prepare_prompts_dataset(
        qm9_path=qm9_path,
        num_prompts=num_prompts,
        seed=seed,
    )

    model_kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if use_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    peft_config = _build_peft_config()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        max_prompt_length=256,
        max_completion_length=128,
        num_generations=4,
        max_steps=max_steps,
        save_steps=min(100, max_steps),
        logging_steps=min(10, max_steps),
        eval_strategy="steps",
        eval_steps=min(50, max_steps),
        bf16=True,
        beta=0.0,
        seed=seed,
        data_seed=seed,
        reward_weights=reward_weights,
        log_completions=True,
        num_completions_to_print=2,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    run_grpo_training()
