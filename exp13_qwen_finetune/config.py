"""Configuration for Experiment 13: Qwen 2.5 0.5B Fine-tuning.

Tests fine-tuning Qwen 2.5 0.5B (decoder-only) with different numbers
of unfrozen layers. Frozen Qwen achieves AUC 0.689 (exp10), matching
fine-tuned ClinicalBERT (0.691). Fine-tuning may push it higher.
"""

from exp4_baseline.config import CLINICAL_DIM as _EXP4_CLINICAL_DIM  # single source of truth
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "exp13_results"

# Qwen model configuration
QWEN_CONFIG = {
    "model_name": "Qwen/Qwen2.5-0.5B",
    "embed_dim": 896,
    "pooling": "last_token",
}

# Clinical feature dimension (from exp10)
CLINICAL_DIM = _EXP4_CLINICAL_DIM

# Fine-tuning configurations to test
FINETUNE_CONFIGS = [
    {
        "name": "exp13_qwen_finetune_1layer",
        "unfreeze_layers": 1,
        "batch_size": 4,
        "encoder_lr": 2e-5,
        "head_lr": 1e-3,
    },
    {
        "name": "exp13_qwen_finetune_2layer",
        "unfreeze_layers": 2,
        "batch_size": 4,
        "encoder_lr": 2e-5,
        "head_lr": 1e-3,
    },
    {
        "name": "exp13_qwen_finetune_4layer",
        "unfreeze_layers": 4,
        "batch_size": 2,  # More layers unfrozen = more memory
        "encoder_lr": 1e-5,  # Lower LR for more unfrozen layers
        "head_lr": 1e-3,
    },
]

# Shared training settings
TRAINING_CONFIG = {
    "weight_decay": 1e-4,
    "epochs": 50,
    "patience": 15,
    "dropout": 0.3,
    "num_classes": 2,
    "max_token_length": 512,
}

# Cross-validation config
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}
