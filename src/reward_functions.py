import numpy as np
from rdkit import Chem
import sys
import os

# Ensure the root directory is in python path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.surrogate_verifier import SurrogateVerifier

def validity_reward_func(completions, **kwargs):
    """
    Reward function that checks if the generated SMILES is valid.
    """
    rewards = []
    for completion in completions:
        # GRPOTrainer passes completion text. 
        # In a chat setup, completion might be a list of dicts.
        # Assuming simple text completions.
        if isinstance(completion, list):
             # Extract content if it's a chat message
             text = completion[-1]["content"] if len(completion) > 0 else ""
        else:
             text = str(completion)
        
        smiles = text.strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            rewards.append(1.0)
        else:
            rewards.append(-1.0) # Penalize invalid strings
    return rewards

def create_surrogate_reward_func(model_path="data/surrogate_gap_model.joblib"):
    verifier = SurrogateVerifier(model_path)
    
    def surrogate_reward_func(completions, **kwargs):
        """
        Reward function that scores the SMILES based on the surrogate model.
        Maximizing HOMO-LUMO gap.
        """
        smiles_list = []
        for completion in completions:
            if isinstance(completion, list):
                 text = completion[-1]["content"] if len(completion) > 0 else ""
            else:
                 text = str(completion)
            smiles_list.append(text.strip())
            
        preds = verifier.predict(smiles_list)
        
        rewards = []
        for smiles, pred in zip(smiles_list, preds):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Maximize gap -> positive reward
                rewards.append(float(pred))
            else:
                rewards.append(0.0) # Handled by validity_reward
                
        return rewards
        
    return surrogate_reward_func
