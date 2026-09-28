"""Configuration for Experiment 1 fusion experiments."""

import os

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, '..', 'asm_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'exp1_results')

# Data files
CSV_PATH = os.path.join(DATA_DIR, 'alfred_1st_regimen.csv')

# Embedding files
TEXT_EMBEDDINGS = {
    'clinicalbert': os.path.join(OUTPUT_DIR, 'bert_alfred_1stregimen_eeg_embeddings.npy'),
    'pubmedbert': os.path.join(OUTPUT_DIR, 'pubmedBert_alfred_1stregimen_eeg_embeddings.npy'),
}

SMILES_EMBEDDINGS = {
    'chemberta': os.path.join(OUTPUT_DIR, 'chemberta_asm_embeddings.npy'),
    'smilestrf': os.path.join(OUTPUT_DIR, 'smilestrf_asm_embeddings.npy'),
}

ASM_NAMES_FILE = os.path.join(OUTPUT_DIR, 'asm_drug_names.txt')

# Embedding dimensions
TEXT_DIM = 768
SMILES_DIMS = {
    'chemberta': 768,
    'smilestrf': 256,
}

# =============================================================================
# DATA MAPPINGS
# =============================================================================
# Map CSV ASM abbreviations to ASM_SMILES dictionary keys
from shared.cohort import ASM_NAME_MAPPING  # single source of truth  # noqa: E402,F401

# Outcome mapping: raw 1 = success -> 1, raw 2 = failure -> 0
from shared.cohort import OUTCOME_MAPPING  # single source of truth  # noqa: E402,F401

# =============================================================================
# EXPERIMENT 1A: CONCAT + MLP
# =============================================================================
CONFIG_1A = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 100,
    'patience': 15,
    'hidden_dims': [512, 256, 128],
    'dropout_rates': [0.3, 0.3, 0.2],
    'num_classes': 2,
}

# =============================================================================
# EXPERIMENT 1B: FUSEMOE
# =============================================================================
CONFIG_1B = {
    'batch_size': 16,
    'learning_rate': 5e-5,
    'weight_decay': 1e-4,
    'epochs': 100,
    'patience': 20,
    'hidden_dim': 256,
    'num_experts': 4,
    'top_k': 2,
    'num_heads': 4,
    'num_layers': 2,
    'dropout': 0.1,
    'aux_loss_weight': 0.1,
    'num_classes': 2,
}

# =============================================================================
# CROSS-VALIDATION
# =============================================================================
CV_CONFIG = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42,
}

# =============================================================================
# EXPERIMENT COMBINATIONS
# =============================================================================
EXPERIMENTS = [
    # Exp 1a: Concat + MLP
    {'name': 'exp1a_clinicalbert_chemberta', 'text': 'clinicalbert', 'smiles': 'chemberta', 'fusion': 'mlp'},
    {'name': 'exp1a_clinicalbert_smilestrf', 'text': 'clinicalbert', 'smiles': 'smilestrf', 'fusion': 'mlp'},
    {'name': 'exp1a_pubmedbert_chemberta', 'text': 'pubmedbert', 'smiles': 'chemberta', 'fusion': 'mlp'},
    {'name': 'exp1a_pubmedbert_smilestrf', 'text': 'pubmedbert', 'smiles': 'smilestrf', 'fusion': 'mlp'},
    # Exp 1b: FuseMoE
    {'name': 'exp1b_clinicalbert_chemberta', 'text': 'clinicalbert', 'smiles': 'chemberta', 'fusion': 'fusemoe'},
    {'name': 'exp1b_clinicalbert_smilestrf', 'text': 'clinicalbert', 'smiles': 'smilestrf', 'fusion': 'fusemoe'},
    {'name': 'exp1b_pubmedbert_chemberta', 'text': 'pubmedbert', 'smiles': 'chemberta', 'fusion': 'fusemoe'},
    {'name': 'exp1b_pubmedbert_smilestrf', 'text': 'pubmedbert', 'smiles': 'smilestrf', 'fusion': 'fusemoe'},
]
