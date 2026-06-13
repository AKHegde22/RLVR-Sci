import pandas as pd
import os

def download_and_process_qm9(output_path="data/qm9_processed.csv"):
    """
    Downloads the QM9 dataset, extracts SMILES and the HOMO-LUMO gap property,
    and saves it to a CSV for surrogate model training.
    """
    print("Downloading QM9 dataset...")
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Downloading from {url}...")
    df = pd.read_csv(url)
    
    # We want SMILES and HOMO-LUMO gap
    df_filtered = df[['smiles', 'gap']].dropna()
    
    df_filtered.to_csv(output_path, index=False)
    print(f"Saved {len(df_filtered)} molecules to {output_path}")

if __name__ == "__main__":
    download_and_process_qm9()
