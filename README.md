# RLVR for Scientific Hypothesis Generation

This project implements Reinforcement Learning from Verifiable Reward (RLVR) for scientific hypothesis generation. Specifically, it focuses on generating novel chemical molecules (represented as SMILES strings) with targeted physical properties.

Because true physics simulations (like Density Functional Theory - DFT) are computationally expensive and too slow to be run directly inside an RL loop, this project introduces a **fast surrogate verifier**. The surrogate verifier approximates the computationally expensive physical simulation, making RL training tractable.

The project uses Group Relative Policy Optimization (GRPO), leveraging Hugging Face's `trl` library with a 7B parameter LLM via LoRA for efficient fine-tuning.

## Architecture

1.  **Dataset:** The pipeline uses the **QM9 dataset**, a standard benchmark in computational chemistry containing ~134k small organic molecules and their quantum mechanical properties.
2.  **Surrogate Verifier:** A Multi-Layer Perceptron (MLP) trained on the QM9 dataset. It takes a molecule's Morgan Fingerprint (via RDKit) as input and predicts the HOMO-LUMO gap.
3.  **Generator (Policy):** A 7B parameter LLM (e.g., `mistralai/Mistral-7B-Instruct-v0.2`) loaded via PEFT/LoRA.
4.  **RLVR Loop (GRPO):**
    *   The LLM generates a group of $G$ candidate SMILES strings for a single prompt.
    *   The **Reward Function** checks for chemical validity (using RDKit). Valid molecules are then scored by the **Surrogate Verifier** to estimate their HOMO-LUMO gap.
    *   GRPO normalizes these scores within the generated group to compute advantages and updates the model to maximize the surrogate score (HOMO-LUMO gap).

## Installation

1. Create a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can run the entire pipeline or individual components using the `main.py` script.

**Run the full pipeline:**
```bash
python main.py
```

**Run individual steps:**

1.  **Download and process the QM9 dataset:**
    ```bash
    python main.py --download
    ```
2.  **Train the surrogate verifier:**
    ```bash
    python main.py --train-surrogate
    ```
3.  **Run the GRPO RLVR loop:**
    *(Note: This requires a GPU with sufficient VRAM to load a 7B model using LoRA in bfloat16, approx 16GB VRAM)*
    ```bash
    python main.py --rlvr
    ```

## Project Structure
- `data/`: Directory where the QM9 dataset and the trained surrogate model (`surrogate_gap_model.joblib`) are saved.
- `src/data_prep.py`: Script to download and parse the QM9 dataset.
- `src/surrogate_verifier.py`: Script defining the surrogate verifier model and the training loop.
- `src/reward_functions.py`: Defines the chemical validity reward and the surrogate property reward used by `trl`.
- `src/grpo_rlvr.py`: Defines the GRPO training setup using PEFT/LoRA and `trl`.
- `main.py`: Orchestrator script.
