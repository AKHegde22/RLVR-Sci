import json
import os
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor

from src.chemistry import morgan_fingerprint, tanimoto_similarity
from src.reward_functions import (
    create_conditional_gap_reward_func,
    create_novelty_reward_func,
    create_surrogate_reward_func,
    diversity_reward_func,
    validity_reward_func,
)
from src.smiles_utils import extract_completion_text, extract_smiles, extract_smiles_from_completion
from src.surrogate_verifier import _meta_path, train_surrogate


class SmilesUtilsTests(unittest.TestCase):
    def test_extract_from_tags(self):
        text = "Here is the molecule: <smiles>CCO</smiles>"
        self.assertEqual(extract_smiles(text), "CCO")

    def test_extract_from_plain(self):
        self.assertEqual(extract_smiles("c1ccccc1"), "c1ccccc1")

    def test_extract_from_prose(self):
        text = "The answer is CCO because it is simple."
        self.assertEqual(extract_smiles(text), "CCO")

    def test_extract_completion_chat(self):
        completion = [{"role": "assistant", "content": "<smiles>CCO</smiles>"}]
        self.assertEqual(extract_smiles_from_completion(completion), "CCO")
        self.assertEqual(extract_completion_text(completion), "<smiles>CCO</smiles>")

    def test_invalid_smiles(self):
        self.assertIsNone(extract_smiles("not a molecule at all"))

    def test_invalid_prose_words(self):
        self.assertIsNone(extract_smiles("invalid molecule text"))


class ChemistryTests(unittest.TestCase):
    def test_tanimoto_identical(self):
        fp = morgan_fingerprint("CCO")
        self.assertIsNotNone(fp)
        self.assertAlmostEqual(tanimoto_similarity(fp, fp), 1.0)


class RewardFunctionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "surrogate.joblib")
        self.fp_path = os.path.join(self.temp_dir.name, "fps.npy")

        smiles = ["C", "CC", "CCO", "c1ccccc1"]
        fps = [morgan_fingerprint(s) for s in smiles]
        self.fp_matrix = np.stack(fps, axis=0)
        np.save(self.fp_path, self.fp_matrix)

        X = self.fp_matrix
        y = np.array([0.1, 0.15, 0.2, 0.25])
        model = MLPRegressor(hidden_layer_sizes=(16,), max_iter=500, random_state=0)
        model.fit(X, y)
        joblib.dump(model, self.model_path)

        meta = {
            "r2": 0.9,
            "mse": 0.001,
            "gap_mean": float(y.mean()),
            "gap_std": float(y.std()),
            "gap_min": float(y.min()),
            "gap_max": float(y.max()),
            "min_r2_threshold": 0.7,
            "fp_radius": 2,
            "fp_n_bits": 2048,
            "random_state": 0,
        }
        with open(_meta_path(self.model_path), "w") as f:
            json.dump(meta, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_validity_reward(self):
        completions = [
            "<smiles>CCO</smiles>",
            "invalid text",
        ]
        rewards = validity_reward_func(completions=completions)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], -1.0)

    def test_surrogate_reward(self):
        reward_fn = create_surrogate_reward_func(self.model_path)
        rewards = reward_fn(completions=["<smiles>CCO</smiles>", "bad"])
        self.assertGreater(rewards[0], rewards[1])

    def test_novelty_reward(self):
        reward_fn = create_novelty_reward_func(self.fp_path, similarity_threshold=0.99)
        rewards = reward_fn(completions=["<smiles>CCO</smiles>"])
        self.assertLessEqual(rewards[0], 0.0)

    def test_diversity_reward(self):
        prompt = [{"role": "user", "content": "test"}]
        completions = ["<smiles>C</smiles>", "<smiles>CCO</smiles>"]
        rewards = diversity_reward_func(
            prompts=[prompt, prompt],
            completions=completions,
        )
        self.assertEqual(len(rewards), 2)
        self.assertGreater(rewards[0], 0.0)

    def test_conditional_gap_reward(self):
        reward_fn = create_conditional_gap_reward_func(self.model_path)
        rewards = reward_fn(
            completions=["<smiles>CCO</smiles>"],
            target_gap=[0.05],
        )
        self.assertEqual(rewards[0], 1.0)
        skipped = reward_fn(completions=["<smiles>CCO</smiles>"], target_gap=[-1.0])
        self.assertIsNone(skipped[0])


class SurrogateTrainingTests(unittest.TestCase):
    def test_train_surrogate_on_tiny_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = os.path.join(tmp, "tiny.csv")
            model_path = os.path.join(tmp, "model.joblib")

            import pandas as pd

            df = pd.DataFrame(
                {
                    "smiles": ["C", "CC", "CCO", "CCC", "c1ccccc1", "CCN", "CCCl", "COC"],
                    "gap": [0.1, 0.12, 0.15, 0.18, 0.22, 0.14, 0.16, 0.17],
                }
            )
            df.to_csv(data_path, index=False)

            meta = train_surrogate(
                data_path=data_path,
                model_path=model_path,
                min_r2=0.0,
                random_state=0,
            )
            self.assertGreater(meta["r2"], 0.0)
            self.assertTrue(os.path.exists(_meta_path(model_path)))


if __name__ == "__main__":
    unittest.main()
