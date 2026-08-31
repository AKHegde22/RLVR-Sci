"""GRPO prompt dataset construction (no torch dependency)."""

import random
from typing import Optional

import pandas as pd
from datasets import Dataset

CONDITIONAL_PROMPT_TEMPLATE = (
    "Respond with exactly one novel molecule using this format: "
    "<smiles>YOUR_SMILES_HERE</smiles>. "
    "Target HOMO-LUMO gap: at least {target_gap:.3f} Hartree. "
    "Do not include any other text."
)

UNCONDITIONAL_PROMPTS = [
    (
        "Respond with exactly one novel small organic molecule using this format: "
        "<smiles>YOUR_SMILES_HERE</smiles>. Maximize HOMO-LUMO gap. "
        "Do not include any other text."
    ),
    (
        "Output a single valid SMILES for a novel molecule with a large HOMO-LUMO gap. "
        "Format: <smiles>YOUR_SMILES_HERE</smiles> only."
    ),
    (
        "Generate one chemically valid, novel SMILES string wrapped in <smiles></smiles> tags. "
        "Prefer structures with high HOMO-LUMO gap."
    ),
]


def prepare_prompts_dataset(
    qm9_path: str = "data/qm9_processed.csv",
    num_prompts: int = 500,
    eval_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Optional[Dataset]]:
    """
    Build diverse property-conditioned and unconditional prompts from QM9 gaps.
    Returns train and eval datasets with target_gap for conditional rewards.
    """
    rng = random.Random(seed)
    df = pd.read_csv(qm9_path)

    if len(df) < num_prompts:
        num_prompts = len(df)

    sampled = df.sample(n=num_prompts, random_state=seed).reset_index(drop=True)

    prompts: list[list[dict[str, str]]] = []
    target_gaps: list[float] = []

    for _, row in sampled.iterrows():
        gap = float(row["gap"])
        if rng.random() < 0.6:
            text = CONDITIONAL_PROMPT_TEMPLATE.format(target_gap=gap)
            target_gaps.append(gap)
        else:
            text = rng.choice(UNCONDITIONAL_PROMPTS)
            target_gaps.append(-1.0)

        prompts.append([{"role": "user", "content": text}])

    dataset = Dataset.from_dict(
        {
            "prompt": prompts,
            "target_gap": target_gaps,
        }
    )

    eval_size = max(1, int(len(dataset) * eval_fraction))
    split = dataset.train_test_split(test_size=eval_size, seed=seed)
    return split["train"], split["test"]
