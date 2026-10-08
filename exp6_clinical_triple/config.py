"""Configuration for Experiment 6: Clinical + SMILES + Third Modality Fusion."""

from exp4_baseline.config import CLINICAL_DIM as _EXP4_CLINICAL_DIM  # single source of truth
from pathlib import Path

from exp2_fusion.config import N_CHANNELS  # the 19-channel cache montage
from shared.eeg_cache import CACHE_PATHS as _EEG_CACHE_PATHS

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "asm_data"
EEG_DIR = DATA_DIR / "Alfred" / "EEG"
CSV_PATH = DATA_DIR / "alfred_1st_regimen.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "exp6_results"
EEG_CACHE_PATH = _EEG_CACHE_PATHS["alfred"]   # version-2 cache (shared/eeg_cache.py)

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

# Text embeddings (from exp1/exp3)
TEXT_EMBEDDINGS = {
    "clinicalbert": OUTPUTS_DIR / "bert_alfred_1stregimen_eeg_embeddings.npy",
    "pubmedbert": OUTPUTS_DIR / "pubmedBert_alfred_1stregimen_eeg_embeddings.npy",
}

# SMILES embeddings (from exp1/exp3)
SMILES_EMBEDDINGS = {
    "chemberta": OUTPUTS_DIR / "chemberta_asm_embeddings.npy",
    "smilestrf": OUTPUTS_DIR / "smilestrf_asm_embeddings.npy",
}

ASM_NAMES_FILE = OUTPUTS_DIR / "asm_drug_names.txt"

# Embedding dimensions
CLINICAL_DIM = _EXP4_CLINICAL_DIM
TEXT_DIM = 768
EEG_DIM = 256  # SimpleCNN output after aggregation
SMILES_DIMS = {
    "chemberta": 768,
    "smilestrf": 256,
}

# EEG processing parameters (from exp2/exp3)
EEG_CONFIG = {
    "target_sr": 200,
    "min_duration_sec": 600,
    "skip_start_sec": 300,
    "use_duration_sec": 1200,
    "window_sec": 10,
    "lowcut": 0.1,
    "highcut": 75.0,
    "notch_freq": 50.0,
}

MAX_WINDOWS = int(EEG_CONFIG["use_duration_sec"] / EEG_CONFIG["window_sec"])  # 120

# EEG encoder config (reuse SimpleCNN from exp2). encoder_type is the default;
# each EEG config's "eeg_model" selects the window encoder (simplecnn/eeg2vec).
EEG_ENCODER_CONFIG = {
    "encoder_type": "simplecnn",
    "n_channels": N_CHANNELS,
    "n_times": 2000,  # 10s @ 200Hz
    "embed_dim": 256,
    "num_heads": 4,
    "num_layers": 2,
    "max_windows": MAX_WINDOWS,
    "window_chunk_size": 32,
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size_text": 16,
    "batch_size_eeg": 8,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "dropout": 0.3,
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

# Experiment definitions
EXPERIMENTS = [
    # Exp6a: Clinical + SMILES + Text
    {
        "name": "exp6a_clinicalbert_chemberta",
        "modality": "text",
        "text_model": "clinicalbert",
        "smiles_model": "chemberta",
    },
    {
        "name": "exp6a_clinicalbert_smilestrf",
        "modality": "text",
        "text_model": "clinicalbert",
        "smiles_model": "smilestrf",
    },
    {
        "name": "exp6a_pubmedbert_chemberta",
        "modality": "text",
        "text_model": "pubmedbert",
        "smiles_model": "chemberta",
    },
    {
        "name": "exp6a_pubmedbert_smilestrf",
        "modality": "text",
        "text_model": "pubmedbert",
        "smiles_model": "smilestrf",
    },
    # Exp6b: Clinical + SMILES + EEG
    {
        "name": "exp6b_simplecnn_chemberta",
        "modality": "eeg",
        "eeg_model": "simplecnn",
        "smiles_model": "chemberta",
    },
    {
        "name": "exp6b_simplecnn_smilestrf",
        "modality": "eeg",
        "eeg_model": "simplecnn",
        "smiles_model": "smilestrf",
    },
    # Pre-specified Clinical + EEG table row (analysis plan section 3): EEG2Vec
    # 256-d windows + the same two-layer transformer aggregator as exp5c.
    {
        "name": "exp6b_eeg2vec_chemberta",
        "modality": "eeg",
        "eeg_model": "eeg2vec",
        "smiles_model": "chemberta",
    },
]
