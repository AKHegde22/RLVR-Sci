import argparse
import sys
import os

# Ensure the root directory is in python path to import src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.data_prep import download_and_process_qm9
from src.surrogate_verifier import train_surrogate
from src.grpo_rlvr import run_grpo_training

def main():
    parser = argparse.ArgumentParser(description="RLVR for Scientific Hypothesis Generation")
    parser.add_argument("--download", action="store_true", help="Download and process QM9 dataset")
    parser.add_argument("--train-surrogate", action="store_true", help="Train the surrogate MLP model on QM9")
    parser.add_argument("--rlvr", action="store_true", help="Run GRPO RLVR training loop")
    
    args = parser.parse_args()
    
    # If no flags passed, default to doing everything
    if not any([args.download, args.train_surrogate, args.rlvr]):
        print("No specific steps selected. Running full pipeline...")
        args.download = True
        args.train_surrogate = True
        args.rlvr = True

    if args.download:
        download_and_process_qm9("data/qm9_processed.csv")
        
    if args.train_surrogate:
        # Check if dataset exists, if not maybe download
        if not os.path.exists("data/qm9_processed.csv"):
            print("Dataset not found. Downloading...")
            download_and_process_qm9("data/qm9_processed.csv")
        train_surrogate("data/qm9_processed.csv", "data/surrogate_gap_model.joblib")
        
    if args.rlvr:
        if not os.path.exists("data/surrogate_gap_model.joblib"):
            print("Surrogate model not found! Please train it first or run the full pipeline.")
            return
        
        # Using a solid 7B chat model (or base model) for RLVR
        run_grpo_training(model_name="mistralai/Mistral-7B-Instruct-v0.2")

if __name__ == "__main__":
    main()
