"""Data pipeline for Experiment 6: Clinical + SMILES + Third Modality Fusion."""

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
    CLINICAL_CONFIG,
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
    clean_psy_column,
    load_clinical_data as _exp4_load_clinical_data,
)
from shared.eeg_cache import load_cache

EEG_CONVENTION = "zscore_window"   # per-window, per-channel z-score in microvolts
from shared.cohort import dedupe_by_pid, smiles_vector

logger = logging.getLogger("exp6")


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_clinical_data(filepath: Path = CSV_PATH) -> pd.DataFrame:
    """Load and preprocess the clinical data.

    Args:
        filepath: Path to CSV file.

    Returns:
        Cleaned DataFrame with valid outcomes.
    """
    # Delegate to exp4's loader (clean + dedupe-by-pid), the single shared
    # clinical cohort definition.
    return _exp4_load_clinical_data(filepath)


def load_asm_drug_names(filepath: Path = ASM_NAMES_FILE) -> List[str]:
    """Load ordered drug names from file."""
    with open(filepath, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_smiles_embeddings(smiles_model: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load SMILES embeddings and create index mapping.

    Args:
        smiles_model: 'chemberta' or 'smilestrf'

    Returns:
        Tuple of (embeddings array, drug name -> index mapping).
    """
    emb_path = SMILES_EMBEDDINGS[smiles_model]
    embeddings = np.load(emb_path)
    drug_names = load_asm_drug_names()
    index_map = {name: i for i, name in enumerate(drug_names)}
    return embeddings, index_map


def load_csv_for_text(filepath: Path = CSV_PATH, filter_outcome: bool = True) -> pd.DataFrame:
    """Load CSV with text-specific filtering (matching exp3 logic)."""
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
    """Load text embeddings and align with patient IDs.

    Args:
        text_model: 'clinicalbert' or 'pubmedbert'
        df: Filtered DataFrame with patient info.

    Returns:
        Dict mapping patient ID to text embedding.
    """
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

class ClinicalSMILESTextDataset(Dataset):
    """Dataset combining clinical features, SMILES embeddings, and text embeddings."""

    def __init__(
        self,
        clinical_features: np.ndarray,
        smiles_embeddings: np.ndarray,
        smiles_indices: Dict[str, int],
        asm_drugs: List[str],
        text_embeddings: np.ndarray,
        labels: np.ndarray,
    ):
        """Initialise dataset.

        Args:
            clinical_features: Clinical feature array (n_samples, 20).
            smiles_embeddings: SMILES embeddings array (n_drugs, embed_dim).
            smiles_indices: Drug name -> embedding index mapping.
            asm_drugs: List of ASM drug abbreviations per patient.
            text_embeddings: Text embeddings array (n_samples, 768).
            labels: Label array (n_samples,).
        """
        self.clinical_features = torch.from_numpy(clinical_features).float()
        self.smiles_embeddings = smiles_embeddings
        self.smiles_indices = smiles_indices
        self.asm_drugs = asm_drugs
        self.text_embeddings = torch.from_numpy(text_embeddings).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample.

        Returns:
            Tuple of (clinical_features, smiles_embedding, text_embedding, label).
        """
        clinical = self.clinical_features[idx]

        # SMILES embedding for this patient's drug (mean fallback if unknown)
        smiles = torch.from_numpy(
            smiles_vector(self.asm_drugs[idx], self.smiles_embeddings, self.smiles_indices)
        ).float()

        text = self.text_embeddings[idx]
        label = self.labels[idx]

        return clinical, smiles, text, label


class ClinicalSMILESEEGDataset(Dataset):
    """Dataset combining clinical features, SMILES embeddings, and EEG data."""

    def __init__(
        self,
        clinical_features: np.ndarray,
        smiles_embeddings: np.ndarray,
        smiles_indices: Dict[str, int],
        asm_drugs: List[str],
        eeg_windows: List[np.ndarray],
        padding_masks: List[np.ndarray],
        labels: np.ndarray,
        max_channels: int = N_CHANNELS,
    ):
        """Initialise dataset.

        Args:
            clinical_features: Clinical feature array (n_samples, 20).
            smiles_embeddings: SMILES embeddings array (n_drugs, embed_dim).
            smiles_indices: Drug name -> embedding index mapping.
            asm_drugs: List of ASM drug abbreviations per patient.
            eeg_windows: List of EEG window arrays per patient.
            padding_masks: List of padding masks per patient.
            labels: Label array (n_samples,).
            max_channels: Maximum number of EEG channels.
        """
        self.clinical_features = torch.from_numpy(clinical_features).float()
        self.smiles_embeddings = smiles_embeddings
        self.smiles_indices = smiles_indices
        self.asm_drugs = asm_drugs
        self.eeg_windows = eeg_windows
        self.padding_masks = padding_masks
        self.labels = torch.from_numpy(labels).long()
        self.max_channels = max_channels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample.

        Returns:
            Tuple of (clinical_features, smiles_embedding, eeg_windows, padding_mask, label).
        """
        clinical = self.clinical_features[idx]

        # SMILES embedding for this patient's drug (mean fallback if unknown)
        smiles = torch.from_numpy(
            smiles_vector(self.asm_drugs[idx], self.smiles_embeddings, self.smiles_indices)
        ).float()

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
        label = self.labels[idx]

        return clinical, smiles, eeg, mask, label


# ============================================================================
# Dataset Creation Functions
# ============================================================================

def prepare_clinical_smiles_text_data(
    smiles_model: str = "chemberta",
    text_model: str = "clinicalbert",
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, int], Dict[str, np.ndarray]]:
    """Prepare data for Clinical + SMILES + Text experiments.

    Args:
        smiles_model: 'chemberta' or 'smilestrf'
        text_model: 'clinicalbert' or 'pubmedbert'

    Returns:
        Tuple of (df, smiles_embeddings, smiles_indices, text_embeddings).
    """
    logger.info(f"Preparing Clinical + SMILES + Text data: {smiles_model}, {text_model}")

    # Load clinical data with text-specific filtering
    df = load_csv_for_text(filter_outcome=True)

    # Clean clinical columns
    df = clean_psy_column(df)
    df = clean_lesion_column(df)

    # Load SMILES embeddings
    smiles_embeddings, smiles_indices = load_smiles_embeddings(smiles_model)
    logger.info(f"Loaded SMILES embeddings: shape={smiles_embeddings.shape}")

    # Load text embeddings
    text_embeddings = load_text_embeddings(text_model, df)
    logger.info(f"Loaded text embeddings for {len(text_embeddings)} patients")

    # SMILES attaches to every patient (mean fallback), so filter on text only,
    # then dedupe by pid (text df is loaded un-deduped for embedding alignment).
    df = dedupe_by_pid(df[df["pid"].astype(str).isin(text_embeddings.keys())])
    logger.info(f"Patients with valid SMILES and text: {len(df)}")

    return df, smiles_embeddings, smiles_indices, text_embeddings


def prepare_clinical_smiles_eeg_data(
    smiles_model: str = "chemberta",
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, int], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Prepare data for Clinical + SMILES + EEG experiments.

    Args:
        smiles_model: 'chemberta' or 'smilestrf'

    Returns:
        Tuple of (df, smiles_embeddings, smiles_indices, eeg_data).
    """
    logger.info(f"Preparing Clinical + SMILES + EEG data: {smiles_model}")

    # Load clinical data
    df = load_clinical_data()

    # Load SMILES embeddings
    smiles_embeddings, smiles_indices = load_smiles_embeddings(smiles_model)
    logger.info(f"Loaded SMILES embeddings: shape={smiles_embeddings.shape}")

    # Load cached EEG data
    eeg_data = load_eeg_data()

    # SMILES attaches to every patient (mean fallback), so filter on EEG only.
    # df is already deduped by load_clinical_data.
    df = df[df["pid"].astype(str).isin(eeg_data.keys())].reset_index(drop=True)
    logger.info(f"Patients with valid SMILES and EEG: {len(df)}")

    return df, smiles_embeddings, smiles_indices, eeg_data


def create_clinical_smiles_text_datasets(
    df: pd.DataFrame,
    smiles_embeddings: np.ndarray,
    smiles_indices: Dict[str, int],
    text_embeddings: Dict[str, np.ndarray],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> Tuple[ClinicalSMILESTextDataset, ClinicalSMILESTextDataset, ClinicalFeaturePreprocessor]:
    """Create train/val datasets for Clinical + SMILES + Text.

    Args:
        df: DataFrame with clinical data.
        smiles_embeddings: SMILES embedding array.
        smiles_indices: Drug name -> index mapping.
        text_embeddings: Dict mapping patient ID to text embedding.
        train_indices: Training fold indices.
        val_indices: Validation fold indices.

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

    # Get ASM drugs and labels
    train_asm = train_df["ASM"].tolist()
    val_asm = val_df["ASM"].tolist()
    train_labels = train_df["outcome"].values
    val_labels = val_df["outcome"].values

    # Get text embeddings in order
    train_text = np.array([text_embeddings[str(pid)] for pid in train_df["pid"]])
    val_text = np.array([text_embeddings[str(pid)] for pid in val_df["pid"]])

    # Create datasets
    train_dataset = ClinicalSMILESTextDataset(
        clinical_features=train_clinical,
        smiles_embeddings=smiles_embeddings,
        smiles_indices=smiles_indices,
        asm_drugs=train_asm,
        text_embeddings=train_text,
        labels=train_labels,
    )

    val_dataset = ClinicalSMILESTextDataset(
        clinical_features=val_clinical,
        smiles_embeddings=smiles_embeddings,
        smiles_indices=smiles_indices,
        asm_drugs=val_asm,
        text_embeddings=val_text,
        labels=val_labels,
    )

    return train_dataset, val_dataset, preprocessor


def create_clinical_smiles_eeg_datasets(
    df: pd.DataFrame,
    smiles_embeddings: np.ndarray,
    smiles_indices: Dict[str, int],
    eeg_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    max_channels: int = N_CHANNELS,
) -> Tuple[ClinicalSMILESEEGDataset, ClinicalSMILESEEGDataset, ClinicalFeaturePreprocessor]:
    """Create train/val datasets for Clinical + SMILES + EEG.

    Args:
        df: DataFrame with clinical data.
        smiles_embeddings: SMILES embedding array.
        smiles_indices: Drug name -> index mapping.
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

    # Get ASM drugs and labels
    train_asm = train_df["ASM"].tolist()
    val_asm = val_df["ASM"].tolist()
    train_labels = train_df["outcome"].values
    val_labels = val_df["outcome"].values

    # Get EEG data in order
    train_windows = [eeg_data[str(pid)][0] for pid in train_df["pid"]]
    train_masks = [eeg_data[str(pid)][1] for pid in train_df["pid"]]
    val_windows = [eeg_data[str(pid)][0] for pid in val_df["pid"]]
    val_masks = [eeg_data[str(pid)][1] for pid in val_df["pid"]]

    # Create datasets
    train_dataset = ClinicalSMILESEEGDataset(
        clinical_features=train_clinical,
        smiles_embeddings=smiles_embeddings,
        smiles_indices=smiles_indices,
        asm_drugs=train_asm,
        eeg_windows=train_windows,
        padding_masks=train_masks,
        labels=train_labels,
        max_channels=max_channels,
    )

    val_dataset = ClinicalSMILESEEGDataset(
        clinical_features=val_clinical,
        smiles_embeddings=smiles_embeddings,
        smiles_indices=smiles_indices,
        asm_drugs=val_asm,
        eeg_windows=val_windows,
        padding_masks=val_masks,
        labels=val_labels,
        max_channels=max_channels,
    )

    return train_dataset, val_dataset, preprocessor


# ============================================================================
# Testing
# ============================================================================

def test_data_pipeline():
    """Test the data pipeline."""
    logging.basicConfig(level=logging.INFO)
    print("Testing Exp6 data pipeline...\n")

    # Test Clinical + SMILES + Text
    print("=" * 50)
    print("Testing Clinical + SMILES + Text (ChemBERTa + ClinicalBERT)")
    print("=" * 50)
    try:
        df, smiles_emb, smiles_idx, text_emb = prepare_clinical_smiles_text_data(
            "chemberta", "clinicalbert"
        )
        print(f"  Patients: {len(df)}")
        print(f"  SMILES shape: {smiles_emb.shape}")
        print(f"  Text embeddings: {len(text_emb)}")

        # Test dataset creation
        n = len(df)
        train_idx = np.arange(int(0.8 * n))
        val_idx = np.arange(int(0.8 * n), n)
        train_ds, val_ds, _ = create_clinical_smiles_text_datasets(
            df, smiles_emb, smiles_idx, text_emb, train_idx, val_idx
        )
        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
        clinical, smiles, text, label = train_ds[0]
        print(f"  Sample shapes: clinical={clinical.shape}, smiles={smiles.shape}, text={text.shape}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test Clinical + SMILES + EEG
    print("\n" + "=" * 50)
    print("Testing Clinical + SMILES + EEG (ChemBERTa + SimpleCNN)")
    print("=" * 50)
    try:
        df, smiles_emb, smiles_idx, eeg_data = prepare_clinical_smiles_eeg_data("chemberta")
        print(f"  Patients: {len(df)}")
        print(f"  SMILES shape: {smiles_emb.shape}")

        n = len(df)
        train_idx = np.arange(int(0.8 * n))
        val_idx = np.arange(int(0.8 * n), n)
        train_ds, val_ds, _ = create_clinical_smiles_eeg_datasets(
            df, smiles_emb, smiles_idx, eeg_data, train_idx, val_idx
        )
        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
        clinical, smiles, eeg, mask, label = train_ds[0]
        print(f"  Sample shapes: clinical={clinical.shape}, smiles={smiles.shape}, eeg={eeg.shape}")
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")

    print("\nData pipeline tests complete.")


if __name__ == "__main__":
    test_data_pipeline()
