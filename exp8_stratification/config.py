"""Configuration for Experiment 8: Stratification Analysis."""

from exp4_baseline.config import CLINICAL_DIM as _EXP4_CLINICAL_DIM  # single source of truth
from pathlib import Path

# Paths (same as exp7)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "asm_data"
CSV_PATH = DATA_DIR / "alfred_1st_regimen.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "exp8_results"
EEG_CACHE_PATH = OUTPUTS_DIR / "eeg_cache" / "processed_eeg.pkl"

# Create results directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Clinical feature configuration
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

# Features with severe imbalance (>95% majority)
SEVERELY_IMBALANCED_FEATURES = ["ld", "birth_t", "febrile", "ci"]

# Features to use for multi-label stratification
# (outcome is always included, these are additional)
STRATIFICATION_FEATURES = ["focal", "sex", "age_group"]

# Text embeddings
TEXT_EMBEDDINGS = {
    "clinicalbert": OUTPUTS_DIR / "bert_alfred_1stregimen_eeg_embeddings.npy",
    "pubmedbert": OUTPUTS_DIR / "pubmedBert_alfred_1stregimen_eeg_embeddings.npy",
}

# SMILES embeddings
SMILES_EMBEDDINGS = {
    "chemberta": OUTPUTS_DIR / "chemberta_asm_embeddings.npy",
}

ASM_NAMES_FILE = OUTPUTS_DIR / "asm_drug_names.txt"

# Embedding dimensions
CLINICAL_DIM = _EXP4_CLINICAL_DIM
TEXT_DIM = 768
EEG_DIM = 256
SMILES_DIM = 768

# EEG processing parameters
EEG_CONFIG = {
    "target_sr": 200,
    "min_duration_sec": 600,
    "skip_start_sec": 300,
    "use_duration_sec": 1200,
    "window_sec": 10,
}

MAX_WINDOWS = int(EEG_CONFIG["use_duration_sec"] / EEG_CONFIG["window_sec"])  # 120

# EEG encoder config
EEG_ENCODER_CONFIG = {
    "encoder_type": "simplecnn",
    "n_channels": 27,
    "n_times": 2000,
    "embed_dim": 256,
    "num_heads": 4,
    "num_layers": 2,
    "max_windows": MAX_WINDOWS,
    "window_chunk_size": 32,
}

# Training configuration (same as exp7a MLP - best performing)
TRAINING_CONFIG = {
    "batch_size": 8,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "dropout": 0.3,
    "hidden_dim": 64,
    "num_classes": 2,
}

# Cross-validation config
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# ASM name mapping
from shared.cohort import ASM_NAME_MAPPING  # single source of truth  # noqa: E402,F401

# Outcome mapping: raw 1 = success -> 1, raw 2 = failure -> 0
from shared.cohort import OUTCOME_MAPPING  # single source of truth  # noqa: E402,F401

# Experiment configurations
EXPERIMENTS = [
    {
        "name": "exp8_baseline",
        "stratification": "outcome_only",
        "description": "Baseline: StratifiedKFold on outcome only",
    },
    {
        "name": "exp8_multilabel",
        "stratification": "multilabel",
        "description": "Multi-label stratification on outcome + focal + sex",
    },
]
