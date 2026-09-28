"""Configuration for Experiment 10: Direct LLM Text Modality."""

from exp4_baseline.config import CLINICAL_DIM as _EXP4_CLINICAL_DIM  # single source of truth
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "asm_data"
CSV_PATH = DATA_DIR / "alfred_1st_regimen.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "exp10_results"

# Clinical feature configuration (from exp4)
CLINICAL_CONFIG = {
    "numeric_features": ["age_init"],
    "binary_features": [
        "sex",
        "pretrt_sz_5",
        "focal",
        "fam_hx",
        "febrile",
        "ci",
        "birth_t",
        "head",
        "drug",
        "alcohol",
        "cvd",
        "psy",
        "ld",
    ],
    "categorical_features": ["lesion", "eeg_cat"],
    "input_dim": _EXP4_CLINICAL_DIM,  # see exp4_baseline/config.py
}

CLINICAL_DIM = _EXP4_CLINICAL_DIM

# LLM model configurations
LLM_MODELS = {
    "pubmedbert": {
        "model_name": "NeuML/pubmedbert-base-embeddings",
        "embed_dim": 768,
        "pooling": "cls",
    },
    "clinicalbert": {
        "model_name": "medicalai/ClinicalBERT",
        "embed_dim": 768,
        "pooling": "cls",
    },
    "qwen": {
        "model_name": "Qwen/Qwen2.5-0.5B",
        "embed_dim": 896,
        "pooling": "last_token",  # Decoder-only model: use last non-pad token
    },
}

# Text preprocessing
MIN_REPORT_LENGTH = 20
ERROR_PATTERNS = ["Err:", "Exceed time window", "#N/A", "No EEG data"]
MAX_TOKEN_LENGTH = 512

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 8,  # Smaller than exp5 due to LLM memory usage
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "dropout": 0.3,
    "num_classes": 2,
}

# Fine-tuning configuration (unfreezing last N layers)
FINETUNE_CONFIG = {
    "batch_size": 4,  # Even smaller for backprop through LLM
    "learning_rate": 2e-5,  # Lower LR for fine-tuning
    "encoder_lr": 2e-5,  # LLM layers
    "head_lr": 1e-3,  # Classification head
    "weight_decay": 1e-4,
    "epochs": 50,
    "patience": 15,
    "dropout": 0.3,
    "num_classes": 2,
    "unfreeze_layers": 2,  # Unfreeze last N transformer layers
}

# Cross-validation config
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# Outcome mapping: raw 1 = success -> 1, raw 2 = failure -> 0
from shared.cohort import OUTCOME_MAPPING  # single source of truth  # noqa: E402,F401

# Experiment definitions
EXPERIMENTS = [
    # Frozen encoder experiments
    {"name": "exp10_pubmedbert_frozen", "llm_model": "pubmedbert", "freeze": True},
    {"name": "exp10_clinicalbert_frozen", "llm_model": "clinicalbert", "freeze": True},
    {"name": "exp10_qwen_frozen", "llm_model": "qwen", "freeze": True},
]

# Fine-tuning experiments (run after identifying best frozen model)
FINETUNE_EXPERIMENTS = [
    {"name": "exp10_pubmedbert_finetune", "llm_model": "pubmedbert", "freeze": False},
    {"name": "exp10_clinicalbert_finetune", "llm_model": "clinicalbert", "freeze": False},
]
