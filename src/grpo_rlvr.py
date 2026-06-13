import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import sys
import os

# Ensure the root directory is in python path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.reward_functions import validity_reward_func, create_surrogate_reward_func

def prepare_prompts_dataset(num_prompts=1000):
    # GRPOTrainer uses conversational templates if chat model
    prompts = ["Generate a novel molecule SMILES string:"] * num_prompts
    dataset = Dataset.from_dict({
        "prompt": [{"role": "user", "content": p} for p in prompts]
    })
    return dataset

def run_grpo_training(model_name="mistralai/Mistral-7B-v0.1", output_dir="outputs/grpo_model"):
    print(f"Loading model and tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with bfloat16 for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Setup LoRA for 7B model fine-tuning
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Create reward functions
    surrogate_reward = create_surrogate_reward_func("data/surrogate_gap_model.joblib")
    reward_funcs = [validity_reward_func, surrogate_reward]

    # Dataset
    train_dataset = prepare_prompts_dataset()

    # GRPO Config
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=1, # Keep small for 7B model
        gradient_accumulation_steps=4,
        max_prompt_length=64,
        max_completion_length=128,
        num_generations=4, # Group size G=4 for GRPO
        max_steps=500,
        save_steps=100,
        logging_steps=10,
        bf16=True,
        beta=0.1, # KL penalty
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    print("Starting GRPO training...")
    trainer.train()
    
    print(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    run_grpo_training()
