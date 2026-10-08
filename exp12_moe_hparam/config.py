"""Configuration for Experiment 12: FuseMoE Hyperparameter Investigation.

Investigates the exp3b FuseMoE regression (AUC 0.753 -> 0.677) by testing
a grid of hyperparameters on the ClinicalBERT + ChemBERTa configuration.
"""

from pathlib import Path
from exp2_fusion.config import N_CHANNELS  # the 19-channel cache montage

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "exp12_results"

# Fixed experiment settings (use best combo from exp3b)
TEXT_MODEL = "clinicalbert"
SMILES_MODEL = "chemberta"
TEXT_DIM = 768
SMILES_DIM = 768

# EEG config (keep SimpleCNN to match exp3b baseline)
EEG_ENCODER_CONFIG = {
    "encoder_type": "simplecnn",
    "n_channels": N_CHANNELS,
    "n_times": 2000,
    "embed_dim": 256,
    "num_heads": 4,
    "num_layers": 2,
    "max_windows": 120,
    "window_chunk_size": 32,
}

# Base training config
BASE_CONFIG = {
    "batch_size": 8,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "hidden_dim": 256,
    "dropout": 0.1,
    "num_classes": 2,
    "aux_loss_weight": 0.1,
    "num_heads": 4,
}

# Cross-validation config
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# Hyperparameter grid
HP_GRID = {
    "learning_rate": [5e-5, 1e-4, 5e-4],
    "num_experts": [2, 4],
    "temp_decay": [0.9995, None],  # None = no temperature annealing
}

def generate_experiments():
    """Generate experiment configs from hyperparameter grid."""
    experiments = []
    for lr in HP_GRID["learning_rate"]:
        for n_exp in HP_GRID["num_experts"]:
            for temp in HP_GRID["temp_decay"]:
                top_k = min(2, n_exp)  # top-k <= num_experts
                temp_str = f"t{temp}" if temp else "notmp"
                name = f"exp12_lr{lr}_e{n_exp}_k{top_k}_{temp_str}"
                experiments.append({
                    "name": name,
                    "learning_rate": lr,
                    "num_experts": n_exp,
                    "top_k": top_k,
                    "temp_decay": temp,
                })
    return experiments

EXPERIMENTS = generate_experiments()
