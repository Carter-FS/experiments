"""Configuration for Experiment 9: EEG Variance Investigation."""

from pathlib import Path

from exp2_fusion.config import N_CHANNELS  # the 19-channel cache montage
from shared.eeg_cache import CACHE_PATHS as _EEG_CACHE_PATHS

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "asm_data"
CSV_PATH = DATA_DIR / "alfred_1st_regimen.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "exp9_results"
EEG_CACHE_PATH = _EEG_CACHE_PATHS["alfred"]   # version-2 cache (shared/eeg_cache.py)

# Create results directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# EEG processing parameters (from exp2)
EEG_CONFIG = {
    "target_sr": 200,
    "min_duration_sec": 600,  # 10 minutes minimum
    "skip_start_sec": 300,    # Skip first 5 minutes
    "use_duration_sec": 1200, # Use up to 20 minutes
    "window_sec": 10,         # 10-second windows
    "lowcut": 0.1,
    "highcut": 75.0,
    "notch_freq": 50.0,
}

MAX_WINDOWS = int(EEG_CONFIG["use_duration_sec"] / EEG_CONFIG["window_sec"])  # 120

# Cross-validation configuration
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# Stratification features (from exp8)
STRATIFICATION_FEATURES = ["focal", "sex"]

# Known per-fold AUC results for reference
EXP5C_FOLD_AUCS = [0.600, 0.604, 0.604, 0.866, 0.545]  # outcome-only stratification
EXP2A_FOLD_AUCS = [0.554, 0.631, 0.662, 0.634, 0.688]  # SimpleCNN + SMILES-Trf MLP

# Quality thresholds for EEG filtering
QUALITY_CONFIG = {
    "min_valid_windows": 60,    # Minimum 10 minutes usable data
    "max_padding_ratio": 0.20,  # Maximum 20% padded windows
    "snr_threshold": 1.0,       # Minimum signal-to-noise ratio
}
