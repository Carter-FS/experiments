"""Configuration for Experiment 3: LLM + EEG + SMILES triple fusion."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "asm_data"
EEG_DIR = DATA_DIR / "Alfred" / "EEG"
CSV_PATH = DATA_DIR / "alfred_1st_regimen.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "exp3_results"

# Text embeddings from Experiment 1
TEXT_EMBEDDINGS = {
    "clinicalbert": OUTPUTS_DIR / "bert_alfred_1stregimen_eeg_embeddings.npy",
    "pubmedbert": OUTPUTS_DIR / "pubmedBert_alfred_1stregimen_eeg_embeddings.npy",
}

# SMILES embeddings
SMILES_EMBEDDINGS = {
    "chemberta": OUTPUTS_DIR / "chemberta_asm_embeddings.npy",
    "smilestrf": OUTPUTS_DIR / "smilestrf_asm_embeddings.npy",
}

ASM_NAMES_FILE = OUTPUTS_DIR / "asm_drug_names.txt"

# Embedding dimensions
TEXT_DIM = 768
EEG_DIM = 256  # SimpleCNN output
SMILES_DIMS = {
    "chemberta": 768,
    "smilestrf": 256,
}

# EEG processing parameters (reuse from exp2)
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

# Training configuration - Exp3a (MLP)
CONFIG_3A = {
    "batch_size": 8,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "hidden_dim": 256,
    "dropout": 0.3,
    "num_classes": 2,
}

# Training configuration - Exp3b (FuseMoE)
CONFIG_3B = {
    "batch_size": 8,
    "learning_rate": 5e-5,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "hidden_dim": 256,
    "num_experts": 4,
    "top_k": 2,
    "num_heads": 4,
    "num_moe_layers": 2,
    "dropout": 0.1,
    "aux_loss_weight": 0.1,
    "num_classes": 2,
}

# Cross-validation config
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# EEG encoder config. A config's optional "eeg_encoder" key overrides
# encoder_type (e.g. the EEG2Vec headline row); other settings are shared.
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

# ASM name mapping
from shared.cohort import ASM_NAME_MAPPING  # single source of truth  # noqa: E402,F401

# Outcome mapping: raw 1 = success -> 1, raw 2 = failure -> 0
from shared.cohort import OUTCOME_MAPPING  # single source of truth  # noqa: E402,F401

# Experiment matrix (9 experiments)
EXPERIMENTS = [
    # Exp3a: MLP fusion
    {"name": "exp3a_clinicalbert_chemberta", "text": "clinicalbert", "smiles": "chemberta", "fusion": "mlp"},
    {"name": "exp3a_clinicalbert_smilestrf", "text": "clinicalbert", "smiles": "smilestrf", "fusion": "mlp"},
    {"name": "exp3a_pubmedbert_chemberta", "text": "pubmedbert", "smiles": "chemberta", "fusion": "mlp"},
    {"name": "exp3a_pubmedbert_smilestrf", "text": "pubmedbert", "smiles": "smilestrf", "fusion": "mlp"},
    # Exp3b: FuseMoE fusion
    {"name": "exp3b_clinicalbert_chemberta", "text": "clinicalbert", "smiles": "chemberta", "fusion": "fusemoe"},
    {"name": "exp3b_clinicalbert_smilestrf", "text": "clinicalbert", "smiles": "smilestrf", "fusion": "fusemoe"},
    {"name": "exp3b_pubmedbert_chemberta", "text": "pubmedbert", "smiles": "chemberta", "fusion": "fusemoe"},
    {"name": "exp3b_pubmedbert_smilestrf", "text": "pubmedbert", "smiles": "smilestrf", "fusion": "fusemoe"},
    # Exp3a with EEG2Vec (256-d, transformer aggregator): the analysis plan's
    # pre-specified Text + EEG row. Kept last: configs share one RNG stream
    # (determinism is seeded once per run), so inserting it earlier would
    # shift every later config's legacy output.
    {"name": "exp3a_clinicalbert_chemberta_eeg2vec", "text": "clinicalbert", "smiles": "chemberta", "fusion": "mlp",
     "eeg_encoder": "eeg2vec"},
]
