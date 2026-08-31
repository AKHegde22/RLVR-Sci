"""Shared Modal resource names and mount paths for RLVR-Sci."""

# Persistent dataset volume in the active Modal workspace (profile: akntemp30).
VOLUME_NAME = "rlvr-sci-data"

# Mount point inside Modal containers.
DATA_MOUNT = "/vol/data"

# Relative paths inside the volume (under DATA_MOUNT).
QM9_CSV_REL = "qm9_processed.csv"
QM9_FINGERPRINTS_REL = "qm9_fingerprints.npy"
QM9_MANIFEST_REL = "dataset_manifest.json"

# Public DeepChem QM9 release (SMILES + quantum properties including gap).
QM9_SOURCE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"

APP_NAME = "rlvr-sci-prepare-qm9"
SMOKE_APP_NAME = "rlvr-sci-smoke-test"

SURROGATE_MODEL_REL = "surrogate_gap_model.joblib"
SURROGATE_META_REL = "surrogate_gap_model_meta.json"

HF_VOLUME_NAME = "huggingface-cache"
HF_CACHE_MOUNT = "/hf-cache"

SMOKE_TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SMOKE_OUTPUT_REL = "smoke_outputs/grpo"
SMOKE_MAX_STEPS = 10
SMOKE_NUM_PROMPTS = 50
SMOKE_MIN_R2 = 0.70

# Scaled integration test (~$3–5 on L4; validates paths closer to full $30 run).
INTEGRATION_APP_NAME = "rlvr-sci-integration-test"
PRODUCTION_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
INTEGRATION_QWEN_STEPS = 60
INTEGRATION_QWEN_PROMPTS = 180
INTEGRATION_MISTRAL_STEPS = 25
INTEGRATION_MISTRAL_PROMPTS = 100
INTEGRATION_QWEN_OUTPUT_REL = "integration_outputs/grpo_qwen"
INTEGRATION_MISTRAL_OUTPUT_REL = "integration_outputs/grpo_mistral"
