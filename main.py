import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_prep import download_and_process_qm9
from src.grpo_rlvr import run_grpo_training
from src.surrogate_verifier import DEFAULT_MIN_R2, train_surrogate

SMOKE_TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def main() -> None:
    parser = argparse.ArgumentParser(description="RLVR for Scientific Hypothesis Generation")
    parser.add_argument("--download", action="store_true", help="Download and process QM9 dataset")
    parser.add_argument("--train-surrogate", action="store_true", help="Train the surrogate MLP model on QM9")
    parser.add_argument("--rlvr", action="store_true", help="Run GRPO RLVR training loop")
    parser.add_argument("--force-download", action="store_true", help="Re-download QM9 and rebuild fingerprint bank")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL, help="Hugging Face model id for GRPO")
    parser.add_argument("--max-steps", type=int, default=500, help="GRPO training steps")
    parser.add_argument("--num-prompts", type=int, default=500, help="Number of RL prompts sampled from QM9")
    parser.add_argument("--min-r2", type=float, default=DEFAULT_MIN_R2, help="Minimum surrogate R2 before RLVR")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-4bit", action="store_true", help="Load policy model in 4-bit for lower VRAM")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick GRPO smoke test on a small model with few steps",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/grpo_model", help="GRPO output directory")

    args = parser.parse_args()

    if not any([args.download, args.train_surrogate, args.rlvr, args.smoke_test]):
        print("No specific steps selected. Running full pipeline...")
        args.download = True
        args.train_surrogate = True
        args.rlvr = True

    data_path = "data/qm9_processed.csv"
    model_path = "data/surrogate_gap_model.joblib"
    fingerprints_path = "data/qm9_fingerprints.npy"

    if args.download or args.force_download:
        download_and_process_qm9(
            output_path=data_path,
            fingerprints_path=fingerprints_path,
            force_download=args.force_download,
        )

    if args.train_surrogate:
        if not os.path.exists(data_path):
            print("Dataset not found. Downloading...")
            download_and_process_qm9(data_path, fingerprints_path)
        train_surrogate(data_path, model_path, min_r2=args.min_r2, random_state=args.seed)

    if args.rlvr or args.smoke_test:
        if not os.path.exists(model_path):
            print("Surrogate model not found! Train it first with --train-surrogate.")
            return
        if not os.path.exists(fingerprints_path):
            print("QM9 fingerprint bank not found! Run --download first.")
            return

        model_name = SMOKE_TEST_MODEL if args.smoke_test else args.model_name
        max_steps = 10 if args.smoke_test else args.max_steps
        num_prompts = 50 if args.smoke_test else args.num_prompts
        output_dir = "outputs/grpo_smoke_test" if args.smoke_test else args.output_dir

        if args.smoke_test:
            print(f"Running smoke test: model={model_name}, max_steps={max_steps}")

        run_grpo_training(
            model_name=model_name,
            output_dir=output_dir,
            surrogate_model_path=model_path,
            qm9_fingerprints_path=fingerprints_path,
            qm9_path=data_path,
            max_steps=max_steps,
            num_prompts=num_prompts,
            min_r2=args.min_r2,
            seed=args.seed,
            use_4bit=args.use_4bit or args.smoke_test,
        )


if __name__ == "__main__":
    main()
