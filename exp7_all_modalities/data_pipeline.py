"""Data pipeline for Experiment 7: All Four Modalities Fusion."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from exp2_fusion.config import N_CHANNELS
from torch.utils.data import Dataset

from .config import (
    ASM_NAMES_FILE,
    CSV_PATH,
    EEG_CACHE_PATH,
    OUTCOME_MAPPING,
    SMILES_EMBEDDINGS,
    TEXT_EMBEDDINGS,
)

# Import from exp4 for clinical preprocessing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp4_baseline.data_pipeline import (
    ClinicalFeaturePreprocessor,
    clean_lesion_column,
    clean_outcome_column,
    clean_psy_column,
)
from shared.eeg_cache import load_cache

EEG_CONVENTION = "zscore_window"   # per-window, per-channel z-score in microvolts
from shared.cohort import dedupe_by_pid, filter_and_map_outcome, smiles_vector

logger = logging.getLogger("exp7")


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_clinical_data(filepath: Path = CSV_PATH) -> pd.DataFrame:
    """Load and preprocess the clinical data."""
    df = pd.read_csv(filepath)
    df = clean_psy_column(df)
    df = clean_lesion_column(df)
    df = clean_outcome_column(df)
    df = df.reset_index(drop=True)
    logger.info(f"Loaded {len(df)} patients with valid outcomes")
    return df


def load_asm_drug_names(filepath: Path = ASM_NAMES_FILE) -> List[str]:
    """Load ordered drug names from file."""
    with open(filepath, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_smiles_embeddings(smiles_model: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load SMILES embeddings and create index mapping."""
    emb_path = SMILES_EMBEDDINGS[smiles_model]
    embeddings = np.load(emb_path)
    drug_names = load_asm_drug_names()
    index_map = {name: i for i, name in enumerate(drug_names)}
    return embeddings, index_map


def load_csv_for_text(filepath: Path = CSV_PATH, filter_outcome: bool = True) -> pd.DataFrame:
    """Load CSV with text-specific filtering."""
    df = pd.read_csv(filepath)

    # Filter for patients with valid EEG reports
    df = df[df["eeg_report"].notna()].copy()
    df = df[df["eeg_report"].str.strip() != ""].copy()

    # Filter out reports that are too short
    MIN_REPORT_LENGTH = 20
    df["report_length"] = df["eeg_report"].str.len()
    df = df[df["report_length"] >= MIN_REPORT_LENGTH].copy()

    # Remove error patterns
    error_patterns = ["Err:", "Exceed time window", "#N/A", "No EEG data"]
    for pattern in error_patterns:
        df = df[~df["eeg_report"].str.contains(pattern, na=False)]

    # Convert outcome to numeric
    df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")

    if filter_outcome:
        df = df[df["outcome"].isin([1, 2])].copy()
        df["outcome"] = df["outcome"].map(OUTCOME_MAPPING).astype(int)

    df = df.reset_index(drop=True)
    return df


def load_text_embeddings(text_model: str, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Load text embeddings and align with patient IDs."""
    emb_path = TEXT_EMBEDDINGS[text_model]
    all_embeddings = np.load(emb_path)

    # Load full CSV without outcome filtering to match embedding order
    df_all = load_csv_for_text(filter_outcome=False)

    if len(all_embeddings) != len(df_all):
        raise ValueError(
            f"Text embeddings ({len(all_embeddings)}) don't match "
            f"CSV rows ({len(df_all)}). Regenerate embeddings."
        )

    # Create pid -> embedding mapping
    pid_to_emb = {}
    for idx, row in df_all.iterrows():
        pid = str(row["pid"])
        pid_to_emb[pid] = all_embeddings[idx]

    # Filter to only valid patients from filtered df
    text_embeddings = {}
    for _, row in df.iterrows():
        pid = str(row["pid"])
        if pid in pid_to_emb:
            text_embeddings[pid] = pid_to_emb[pid]

    return text_embeddings


def load_eeg_data(cache_path: Path = EEG_CACHE_PATH) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """EEG windows from the version-2 cache, normalised per window and channel.

    Returns:
        Dict mapping patient ID to (windows, padding_mask); see ``shared.eeg_cache``.
    """
    if not cache_path.exists():
        raise FileNotFoundError(
            f"EEG cache not found at {cache_path}. Build it with: python -m shared.eeg_cache build --cohort alfred"
        )
    logger.info(f"Loading EEG windows from {cache_path} ({EEG_CONVENTION})")
    eeg_data = load_cache(cache_path, EEG_CONVENTION)
    logger.info(f"Loaded {len(eeg_data)} patients from cache")
    return eeg_data



# ============================================================================
# Dataset Classes
# ============================================================================

class QuadModalityDataset(Dataset):
    """Dataset combining all 4 modalities: Clinical, Text, EEG, SMILES."""

    def __init__(
        self,
        clinical_features: np.ndarray,
        text_embeddings: np.ndarray,
        eeg_windows: List[np.ndarray],
        padding_masks: List[np.ndarray],
        smiles_embeddings: np.ndarray,
        smiles_indices: Dict[str, int],
        asm_drugs: List[str],
        labels: np.ndarray,
        max_channels: int = N_CHANNELS,
        pids: List = None,
        return_pid: bool = False,
    ):
        """Initialise dataset.

        Args:
            clinical_features: Clinical feature array (n_samples, 20).
            text_embeddings: Text embeddings array (n_samples, 768).
            eeg_windows: List of EEG window arrays per patient.
            padding_masks: List of padding masks per patient.
            smiles_embeddings: SMILES embeddings array (n_drugs, 768).
            smiles_indices: Drug name -> embedding index mapping.
            asm_drugs: List of ASM drug abbreviations per patient.
            labels: Label array (n_samples,).
            max_channels: Maximum number of EEG channels.
            pids: Optional list of patient IDs aligned with the other arrays.
            return_pid: If True, ``__getitem__`` appends the pid as an extra
                element. Defaults to False so existing call sites are
                unaffected.
        """
        self.clinical_features = torch.from_numpy(clinical_features).float()
        self.text_embeddings = torch.from_numpy(text_embeddings).float()
        self.eeg_windows = eeg_windows
        self.padding_masks = padding_masks
        self.smiles_embeddings = smiles_embeddings
        self.smiles_indices = smiles_indices
        self.asm_drugs = asm_drugs
        self.labels = torch.from_numpy(labels).long()
        self.max_channels = max_channels
        self.pids = [str(p) for p in pids] if pids is not None else None
        self.return_pid = return_pid
        if self.return_pid and self.pids is None:
            raise ValueError("return_pid=True requires pids to be supplied.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple:
        """Get a sample.

        Returns:
            Tuple of (clinical, text, eeg_windows, padding_mask, smiles, label).
            If ``return_pid`` was set on the dataset, the patient ID string is
            appended as a final element.
        """
        clinical = self.clinical_features[idx]
        text = self.text_embeddings[idx]

        # EEG windows
        windows = self.eeg_windows[idx]
        padding_mask = self.padding_masks[idx]

        n_windows, n_channels, n_times = windows.shape

        # Pad channels if needed
        if n_channels < self.max_channels:
            padded = np.zeros((n_windows, self.max_channels, n_times), dtype=np.float32)
            padded[:, :n_channels, :] = windows
            windows = padded

        eeg = torch.from_numpy(windows).float()
        mask = torch.from_numpy(padding_mask).bool()

        # SMILES embedding (mean fallback if the drug is unknown)
        smiles = torch.from_numpy(
            smiles_vector(self.asm_drugs[idx], self.smiles_embeddings, self.smiles_indices)
        ).float()

        label = self.labels[idx]

        if self.return_pid:
            return clinical, text, eeg, mask, smiles, label, self.pids[idx]
        return clinical, text, eeg, mask, smiles, label


# ============================================================================
# Dataset Creation Functions
# ============================================================================

def prepare_quad_modality_data(
    text_model: str = "clinicalbert",
    smiles_model: str = "chemberta",
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, int], Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Prepare data for all 4 modalities.

    Args:
        text_model: 'clinicalbert' or 'pubmedbert'
        smiles_model: 'chemberta'

    Returns:
        Tuple of (df, smiles_embeddings, smiles_indices, text_embeddings, eeg_data).
    """
    logger.info(f"Preparing quad modality data: {text_model}, {smiles_model}")

    # Step 1: the EEG cohort is every patient with a recording in the version-2 cache
    # (the same base as exp2, exp3, exp5c and exp6b).
    eeg_data = load_eeg_data()
    valid_pids = set(eeg_data.keys())
    logger.info(f"Loaded EEG windows for {len(valid_pids)} patients")

    # Step 2: Load full clinical data with all columns
    df = pd.read_csv(CSV_PATH)

    # Filter to patients with valid EEG files
    df = df[df["pid"].astype(str).isin(valid_pids)].copy()

    # Filter for valid outcomes and map
    df = filter_and_map_outcome(df)

    # Clean clinical columns
    df = clean_psy_column(df)
    df = clean_lesion_column(df)
    df = df.reset_index(drop=True)
    logger.info(f"Patients with valid clinical data: {len(df)}")

    # Load SMILES embeddings
    smiles_embeddings, smiles_indices = load_smiles_embeddings(smiles_model)
    logger.info(f"Loaded SMILES embeddings: shape={smiles_embeddings.shape}")

    # Load text embeddings
    text_embeddings = load_text_embeddings(text_model, df)
    logger.info(f"Loaded text embeddings for {len(text_embeddings)} patients")

    # Filter to patients with clinical + text + EEG (SMILES is a fixed per-drug
    # input attached to every patient via smiles_vector, so it does not gate the
    # cohort), then dedupe by pid before the fold split.
    valid_mask = (
        df["pid"].astype(str).isin(text_embeddings.keys()) &
        df["pid"].astype(str).isin(eeg_data.keys())
    )
    df = dedupe_by_pid(df[valid_mask])
    logger.info(f"Patients with all 4 modalities: {len(df)}")

    return df, smiles_embeddings, smiles_indices, text_embeddings, eeg_data


def create_quad_modality_datasets(
    df: pd.DataFrame,
    smiles_embeddings: np.ndarray,
    smiles_indices: Dict[str, int],
    text_embeddings: Dict[str, np.ndarray],
    eeg_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    max_channels: int = N_CHANNELS,
    return_pid: bool = False,
) -> Tuple[QuadModalityDataset, QuadModalityDataset, ClinicalFeaturePreprocessor]:
    """Create train/val datasets for all 4 modalities.

    Args:
        df: DataFrame with patient data.
        smiles_embeddings: SMILES embedding array.
        smiles_indices: Drug name -> index mapping.
        text_embeddings: Dict mapping patient ID to text embedding.
        eeg_data: Dict mapping patient ID to (windows, padding_mask).
        train_indices: Training fold indices.
        val_indices: Validation fold indices.
        max_channels: Maximum number of EEG channels.

    Returns:
        Tuple of (train_dataset, val_dataset, preprocessor).
    """
    # Split data
    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()

    # Fit clinical preprocessor on training data only
    preprocessor = ClinicalFeaturePreprocessor()
    train_clinical = preprocessor.fit_transform(train_df)
    val_clinical = preprocessor.transform(val_df)

    # Get text embeddings in order
    train_text = np.array([text_embeddings[str(pid)] for pid in train_df["pid"]])
    val_text = np.array([text_embeddings[str(pid)] for pid in val_df["pid"]])

    # Get EEG data in order
    train_windows = [eeg_data[str(pid)][0] for pid in train_df["pid"]]
    train_masks = [eeg_data[str(pid)][1] for pid in train_df["pid"]]
    val_windows = [eeg_data[str(pid)][0] for pid in val_df["pid"]]
    val_masks = [eeg_data[str(pid)][1] for pid in val_df["pid"]]

    # Get ASM drugs and labels
    train_asm = train_df["ASM"].tolist()
    val_asm = val_df["ASM"].tolist()
    train_labels = train_df["outcome"].values
    val_labels = val_df["outcome"].values

    # Carry pids through so callers can opt in to pid-aware sampling.
    train_pids = train_df["pid"].astype(str).tolist()
    val_pids = val_df["pid"].astype(str).tolist()

    # Create datasets
    train_dataset = QuadModalityDataset(
        clinical_features=train_clinical,
        text_embeddings=train_text,
        eeg_windows=train_windows,
        padding_masks=train_masks,
        smiles_embeddings=smiles_embeddings,
        smiles_indices=smiles_indices,
        asm_drugs=train_asm,
        labels=train_labels,
        max_channels=max_channels,
        pids=train_pids,
        return_pid=return_pid,
    )

    val_dataset = QuadModalityDataset(
        clinical_features=val_clinical,
        text_embeddings=val_text,
        eeg_windows=val_windows,
        padding_masks=val_masks,
        smiles_embeddings=smiles_embeddings,
        smiles_indices=smiles_indices,
        asm_drugs=val_asm,
        labels=val_labels,
        max_channels=max_channels,
        pids=val_pids,
        return_pid=return_pid,
    )

    return train_dataset, val_dataset, preprocessor


# ============================================================================
# Testing
# ============================================================================

def test_data_pipeline():
    """Test the data pipeline."""
    logging.basicConfig(level=logging.INFO)
    print("Testing Exp7 data pipeline...\n")

    print("=" * 50)
    print("Testing Quad Modality Data (ClinicalBERT + ChemBERTa)")
    print("=" * 50)
    try:
        df, smiles_emb, smiles_idx, text_emb, eeg_data = prepare_quad_modality_data(
            "clinicalbert", "chemberta"
        )
        print(f"  Patients with ALL 4 modalities: {len(df)}")
        print(f"  SMILES shape: {smiles_emb.shape}")
        print(f"  Text embeddings: {len(text_emb)}")
        print(f"  EEG data: {len(eeg_data)}")

        # Test dataset creation
        n = len(df)
        train_idx = np.arange(int(0.8 * n))
        val_idx = np.arange(int(0.8 * n), n)
        train_ds, val_ds, _ = create_quad_modality_datasets(
            df, smiles_emb, smiles_idx, text_emb, eeg_data, train_idx, val_idx
        )
        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
        clinical, text, eeg, mask, smiles, label = train_ds[0]
        print(f"  Sample shapes:")
        print(f"    clinical={clinical.shape}, text={text.shape}")
        print(f"    eeg={eeg.shape}, mask={mask.shape}")
        print(f"    smiles={smiles.shape}, label={label.item()}")
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")

    print("\nData pipeline tests complete.")


if __name__ == "__main__":
    test_data_pipeline()
