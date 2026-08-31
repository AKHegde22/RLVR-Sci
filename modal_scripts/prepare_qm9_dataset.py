"""
Modal job: download and process QM9 into workspace Volume storage.

Run from repo root:
  modal run modal_scripts/prepare_qm9_dataset.py
  modal run modal_scripts/prepare_qm9_dataset.py --force
"""

import modal

from modal_scripts.config import (
    APP_NAME,
    DATA_MOUNT,
    QM9_CSV_REL,
    QM9_FINGERPRINTS_REL,
    QM9_MANIFEST_REL,
    VOLUME_NAME,
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas==2.2.3",
        "numpy==2.1.3",
        "tqdm==4.67.1",
        "rdkit==2024.9.5",
    )
    .add_local_python_source("modal_scripts")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={DATA_MOUNT: volume},
    timeout=60 * 60 * 3,
    cpu=2,
    memory=8192,
)
def prepare_qm9_on_volume(force: bool = False) -> dict:
    """Download QM9, process, and persist artifacts under the Modal Volume."""
    from modal_scripts.qm9_processing import prepare_qm9_files

    summary = prepare_qm9_files(data_dir=DATA_MOUNT, force=force)
    volume.commit()
    return summary


@app.local_entrypoint()
def main(force: bool = False):
    print(f"Modal profile workspace volume: {VOLUME_NAME}")
    print(f"Data mount: {DATA_MOUNT}")
    summary = prepare_qm9_on_volume.remote(force=force)
    print("Dataset preparation complete.")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("\nVolume files:")
    print(f"  {DATA_MOUNT}/{QM9_CSV_REL}")
    print(f"  {DATA_MOUNT}/{QM9_FINGERPRINTS_REL}")
    print(f"  {DATA_MOUNT}/{QM9_MANIFEST_REL}")
