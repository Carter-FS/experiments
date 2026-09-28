"""Configuration for Experiment 4: Clinical features baseline."""

from pathlib import Path

# Paths (following exp3 pattern)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "asm_data"
CSV_PATH = DATA_DIR / "alfred_1st_regimen.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "exp4_results"

# Age bins following Hakeem et al. 2022 (JAMA Neurology)
# Tertile-based from Glasgow cohort, <18 added for HEP cohort compatibility
AGE_BINS = [0, 18, 29, 46, float("inf")]
AGE_BIN_LABELS = ["under_18", "18_to_28", "29_to_45", "46_plus"]

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
    # Final dimension: 13 binary + 4 age bins + 2 x 3 one-hot categorical = 23
    # (analysis plan B.3; 19 before 2026-09-28 when the categories were
    # collapsed to normal/abnormal). Must equal ClinicalFeaturePreprocessor.n_features().
    "input_dim": 23,
}
CLINICAL_DIM = CLINICAL_CONFIG["input_dim"]

# Experiment 4a: Simple MLP
CONFIG_4A = {
    "batch_size": 16,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "hidden_dims": [64, 32],
    "dropout": 0.3,
    "num_classes": 2,
}

# Experiment 4b: MLP with Self-Attention (following Feng et al. 2025)
CONFIG_4B = {
    "batch_size": 16,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "hidden_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.2,
    "num_classes": 2,
}

# Cross-validation configuration (matching exp1-3)
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# Outcome mapping: raw 1 = success -> 1, raw 2 = failure -> 0
from shared.cohort import OUTCOME_MAPPING  # single source of truth  # noqa: E402,F401

# Experiments to run
EXPERIMENTS = [
    {"name": "exp4a_mlp", "model": "mlp"},
    {"name": "exp4b_attention", "model": "attention"},
]
