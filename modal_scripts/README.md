# Modal scripts (RLVR-Sci)

Modal deployment code lives here **separately** from `src/` and `main.py` to avoid mixing local and cloud workflows.

**Active profile:** use `modal profile current` — dataset volumes are created in that workspace (currently `akntemp30`).

## Volume

| Resource | Name | Mount |
|----------|------|-------|
| Dataset volume | `rlvr-sci-data` | `/vol/data` |

Files after preparation:

- `qm9_processed.csv` — canonical SMILES + HOMO-LUMO `gap` (Hartree)
- `qm9_fingerprints.npy` — Morgan fingerprint bank for novelty rewards
- `dataset_manifest.json` — download metadata and stats

Source: [DeepChem QM9 CSV](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv)

## Prepare QM9 dataset on Modal

From the repository root:

```bash
modal profile current
modal run modal_scripts/prepare_qm9_dataset.py
```

Force re-download and rebuild:

```bash
modal run modal_scripts/prepare_qm9_dataset.py --force
```

Inspect the volume:

```bash
modal volume ls rlvr-sci-data
```

Expected files at the volume root:

- `qm9_processed.csv`
- `qm9_fingerprints.npy`
- `dataset_manifest.json`

Inside Modal containers these are mounted at `/vol/data/` (see `config.py`).

## Layout

```
modal_scripts/
  config.py                 # Volume names and paths
  qm9_processing.py         # Dataset logic (no src/ imports)
  prepare_qm9_dataset.py    # Modal app entrypoint
  README.md
```

Future Modal jobs (surrogate training, GRPO) should be added here, not in `src/`.

## Integration test (scaled, ~$3–5 on L4)

Before a full Mistral-7B run, exercise medium Qwen GRPO + short Mistral 4-bit GRPO:

```bash
# Detached — safe to close laptop / disconnect (runs entirely on Modal)
modal run --detach modal_scripts/integration_test.py --skip-surrogate

# Block locally until finished (for debugging)
modal run modal_scripts/integration_test.py --skip-surrogate --wait
```

Track progress: `modal app list` or the Modal dashboard. Logs: `modal app logs rlvr-sci-integration-test`

Runs: dataset verify → surrogate (cached if possible) → extended reward checks
(including conditional gap with `target_gap`) → surrogate batch inference →
Qwen GRPO (60 steps) → Mistral-7B 4-bit GRPO (25 steps).

## Smoke test (full pipeline on Modal)

```bash
modal run modal_scripts/smoke_test.py
```

Runs: dataset verify → surrogate training → reward checks → GRPO (Qwen2.5-0.5B, 10 steps, L4 GPU).
