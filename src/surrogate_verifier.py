import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return None

def train_surrogate(data_path="data/qm9_processed.csv", model_path="data/surrogate_gap_model.joblib"):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Computing fingerprints...")
    fps = []
    gaps = []
    
    for idx, row in df.iterrows():
        fp = get_morgan_fingerprint(row['smiles'])
        if fp is not None:
            fps.append(fp)
            gaps.append(row['gap'])
            
    X = np.array(fps)
    y = np.array(gaps)
    
    print(f"Dataset size: {X.shape[0]} valid molecules")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    print("Training MLP Regressor...")
    model = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=200, early_stopping=True, verbose=True)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R2 Score: {r2:.4f}")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

class SurrogateVerifier:
    def __init__(self, model_path="data/surrogate_gap_model.joblib"):
        self.model = joblib.load(model_path)
        
    def predict(self, smiles_list):
        fps = [get_morgan_fingerprint(s) for s in smiles_list]
        valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
        
        results = np.zeros(len(smiles_list)) # Default score
        if len(valid_idx) > 0:
            valid_fps = np.array([fps[i] for i in valid_idx])
            preds = self.model.predict(valid_fps)
            for i, pred in zip(valid_idx, preds):
                results[i] = pred
        return results

if __name__ == "__main__":
    train_surrogate()
