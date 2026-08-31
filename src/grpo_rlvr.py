import os
import inspect
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompt_dataset import prepare_prompts_dataset
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

NUM_GENERATIONS = 4


def _per_device_train_batch_size(num_generations: int = NUM_GENERATIONS) -> int:
    """TRL requires (per_device_batch * world_size) % num_generations == 0."""
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    per_device = 1
    while (per_device * world_size) % num_generations != 0:
        per_device += 1
    return per_device


def _supported_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs that the installed TRL class does not accept."""
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def _build_grpo_config(
    output_dir: str,
    max_steps: int,
    seed: int,
    reward_weights: list[float],
    per_device_train_batch_size: int,
) -> GRPOConfig:
    return GRPOConfig(
        **_supported_kwargs(
            GRPOConfig,
            {
                "output_dir": output_dir,
                "learning_rate": 1e-5,
                "per_device_train_batch_size": per_device_train_batch_size,
                "per_device_eval_batch_size": per_device_train_batch_size,
                "gradient_accumulation_steps": 4,
                "max_prompt_length": 256,
                "max_completion_length": 128,
                "num_generations": NUM_GENERATIONS,
                "max_steps": max_steps,
                "save_steps": min(100, max_steps),
                "logging_steps": min(10, max_steps),
                "eval_strategy": "steps",
                "eval_steps": min(50, max_steps),
                "bf16": True,
                "beta": 0.0,
                "seed": seed,
                "data_seed": seed,
                "reward_weights": reward_weights,
                "log_completions": True,
                "num_completions_to_print": 2,
                "report_to": "none",
                "gradient_checkpointing": True,
                # Keep target_gap (and other reward kwargs) from the dataset.
                "remove_unused_columns": False,
            },
        )
    )


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

    per_device_batch = _per_device_train_batch_size(NUM_GENERATIONS)
    print(f"GRPO per_device_train_batch_size={per_device_batch} (num_generations={NUM_GENERATIONS})")

    training_args = _build_grpo_config(
        output_dir=output_dir,
        max_steps=max_steps,
        seed=seed,
        reward_weights=reward_weights,
        per_device_train_batch_size=per_device_batch,
    )

    trainer = GRPOTrainer(
        **_supported_kwargs(
            GRPOTrainer,
            {
                "model": model,
                "reward_funcs": reward_funcs,
                "args": training_args,
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset,
                "processing_class": tokenizer,
                "tokenizer": tokenizer,
                "reward_weights": reward_weights,
            },
        )
    )

    print("Starting GRPO training...")
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    run_grpo_training()
