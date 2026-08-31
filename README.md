# RLVR for Scientific Hypothesis Generation

This project implements Reinforcement Learning from Verifiable Reward (RLVR) for scientific hypothesis generation. Specifically, it focuses on generating novel chemical molecules (represented as SMILES strings) with targeted physical properties.

Because true physics simulations (like Density Functional Theory - DFT) are computationally expensive and too slow to be run directly inside an RL loop, this project introduces a **fast surrogate verifier**. The surrogate verifier approximates the computationally expensive physical simulation, making RL training tractable.

The project uses Group Relative Policy Optimization (GRPO), leveraging Hugging Face's `trl` library with a 7B parameter LLM via LoRA for efficient fine-tuning.

## Architecture

1. **Dataset:** The pipeline uses the **QM9 dataset**, a standard benchmark in computational chemistry containing ~134k small organic molecules and their quantum mechanical properties.
2. **Surrogate Verifier:** A Multi-Layer Perceptron (MLP) trained on the QM9 dataset. It takes a molecule's Morgan Fingerprint (via RDKit) as input and predicts the HOMO-LUMO gap. Training is gated on a minimum test R² before RLVR runs.
3. **Generator (Policy):** A 7B parameter LLM (e.g., `mistralai/Mistral-7B-Instruct-v0.2`) loaded via PEFT/LoRA with attention and MLP adapters.
4. **RLVR Loop (GRPO):**
   - The LLM generates a group of $G$ candidate SMILES strings for each prompt.
   - **Reward functions** (summed with weights):
     - **Validity:** RDKit parse check on extracted SMILES (supports `<smiles>` tags and free-text extraction).
     - **Surrogate gap:** Normalized predicted HOMO-LUMO gap from the MLP verifier.
     - **Novelty:** Penalizes high Tanimoto similarity to QM9; rewards distance from the training bank.
     - **Diversity:** Within each prompt group, rewards structurally diverse completions.
     - **Conditional target:** For property-conditioned prompts, rewards meeting the requested gap threshold.
   - GRPO normalizes rewards within each generation group and updates the policy.

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

1. **Download and process the QM9 dataset** (canonical SMILES + fingerprint bank for novelty):
   ```bash
   python main.py --download
   ```
2. **Train the surrogate verifier** (requires R² ≥ 0.70 by default):
   ```bash
   python main.py --train-surrogate
   ```
3. **Run the GRPO RLVR loop:**
   *(Requires a GPU with sufficient VRAM; ~16GB+ with 4-bit QLoRA for 7B)*
   ```bash
   python main.py --rlvr
   ```

**Smoke test** (small model, 10 steps, 4-bit — useful before a full run):
```bash
python main.py --smoke-test
```

**Common options:**
```bash
python main.py --rlvr \
  --model-name mistralai/Mistral-7B-Instruct-v0.2 \
  --max-steps 500 \
  --num-prompts 500 \
  --min-r2 0.70 \
  --use-4bit \
  --seed 42
```

## Testing

```bash
python -m pytest tests/ -v
```

## Project Structure

- `data/`: QM9 CSV, surrogate model (`surrogate_gap_model.joblib`), metadata, and QM9 fingerprint bank (`qm9_fingerprints.npy`).
- `src/data_prep.py`: Download QM9, canonicalize SMILES, build fingerprint bank.
- `src/chemistry.py`: Fingerprints and Tanimoto similarity helpers.
- `src/smiles_utils.py`: SMILES extraction from model completions.
- `src/surrogate_verifier.py`: Surrogate MLP training, metadata, and prediction.
- `src/reward_functions.py`: GRPO reward functions (validity, surrogate, novelty, diversity, conditional).
- `src/grpo_rlvr.py`: GRPO training setup with eval split and logging.
- `tests/`: Unit tests for parsing, chemistry, rewards, and surrogate training.
- `modal_scripts/`: Modal cloud jobs (dataset volume prep, etc.) — separate from `src/`.
- `main.py`: Orchestrator script.

## Design Notes

- **Output format:** Prompts instruct the model to wrap SMILES in `<smiles>...</smiles>`; the parser also handles plain SMILES and prose.
- **Memory:** `beta=0.0` avoids loading a separate reference model. Use `--use-4bit` on smaller GPUs.
- **Surrogate gate:** RLVR refuses to start if surrogate test R² is below `--min-r2`.
