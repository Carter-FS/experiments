"""Configuration for Experiment 11: EEG2Vec 128D Re-runs with Aggregator Variants."""

from pathlib import Path
from exp2_fusion.config import N_CHANNELS  # the 19-channel cache montage

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "exp11_results"

# EEG encoder configuration (128D based on exp9 findings)
EEG_CONFIG = {
    "encoder_type": "eeg2vec",
    "n_channels": N_CHANNELS,
    "n_times": 2000,  # 10s @ 200Hz
    "embed_dim": 128,  # Reduced from 256 based on exp9 ablation
    "num_heads": 4,
    "num_layers": 2,
    "max_windows": 120,
    "window_chunk_size": 32,
}

# SMILES embedding dimensions
SMILES_DIMS = {
    "chemberta": 768,
    "smilestrf": 256,
}

# Training configs (match parent experiments)
CONFIG_EXP3A = {
    "batch_size": 8,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "hidden_dim": 256,
    "dropout": 0.3,
    "num_classes": 2,
}

CONFIG_EXP6B = {
    "batch_size_eeg": 8,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "hidden_dim": 64,
    "dropout": 0.3,
    "num_classes": 2,
}

CONFIG_EXP7A = {
    "batch_size": 8,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "hidden_dim": 64,
    "dropout": 0.3,
    "num_classes": 2,
}

# Cross-validation config
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# Experiment matrix
EXPERIMENTS = [
    # Exp3a re-runs: Triple MLP (Text + EEG + SMILES)
    {"name": "exp11_3a_clinicalbert_chemberta_trf", "base": "exp3a", "text": "clinicalbert", "smiles": "chemberta", "aggregator": "transformer"},
    {"name": "exp11_3a_clinicalbert_smilestrf_trf", "base": "exp3a", "text": "clinicalbert", "smiles": "smilestrf", "aggregator": "transformer"},
    {"name": "exp11_3a_pubmedbert_chemberta_trf", "base": "exp3a", "text": "pubmedbert", "smiles": "chemberta", "aggregator": "transformer"},
    {"name": "exp11_3a_pubmedbert_smilestrf_trf", "base": "exp3a", "text": "pubmedbert", "smiles": "smilestrf", "aggregator": "transformer"},
    {"name": "exp11_3a_clinicalbert_chemberta_meanmax", "base": "exp3a", "text": "clinicalbert", "smiles": "chemberta", "aggregator": "meanmax"},
    {"name": "exp11_3a_clinicalbert_smilestrf_meanmax", "base": "exp3a", "text": "clinicalbert", "smiles": "smilestrf", "aggregator": "meanmax"},
    {"name": "exp11_3a_pubmedbert_chemberta_meanmax", "base": "exp3a", "text": "pubmedbert", "smiles": "chemberta", "aggregator": "meanmax"},
    {"name": "exp11_3a_pubmedbert_smilestrf_meanmax", "base": "exp3a", "text": "pubmedbert", "smiles": "smilestrf", "aggregator": "meanmax"},
    # Exp6b re-runs: Clinical + SMILES + EEG
    {"name": "exp11_6b_chemberta_trf", "base": "exp6b", "smiles": "chemberta", "aggregator": "transformer"},
    {"name": "exp11_6b_smilestrf_trf", "base": "exp6b", "smiles": "smilestrf", "aggregator": "transformer"},
    {"name": "exp11_6b_chemberta_meanmax", "base": "exp6b", "smiles": "chemberta", "aggregator": "meanmax"},
    {"name": "exp11_6b_smilestrf_meanmax", "base": "exp6b", "smiles": "smilestrf", "aggregator": "meanmax"},
    # Exp7a re-runs: Quad MLP (Clinical + Text + EEG + SMILES)
    {"name": "exp11_7a_clinicalbert_chemberta_trf", "base": "exp7a", "text": "clinicalbert", "smiles": "chemberta", "aggregator": "transformer"},
    {"name": "exp11_7a_pubmedbert_chemberta_trf", "base": "exp7a", "text": "pubmedbert", "smiles": "chemberta", "aggregator": "transformer"},
    {"name": "exp11_7a_clinicalbert_chemberta_meanmax", "base": "exp7a", "text": "clinicalbert", "smiles": "chemberta", "aggregator": "meanmax"},
    {"name": "exp11_7a_pubmedbert_chemberta_meanmax", "base": "exp7a", "text": "pubmedbert", "smiles": "chemberta", "aggregator": "meanmax"},
]
