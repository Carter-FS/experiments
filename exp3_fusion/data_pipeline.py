"""Data pipeline for Experiment 3: LLM + EEG + SMILES triple fusion."""

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
    SMILES_EMBEDDINGS,
    TEXT_EMBEDDINGS,
)

# Import EEG processing from exp2
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.cohort import OUTCOME_MAPPING, dedupe_by_pid, smiles_vector
from shared.eeg_cache import CACHE_PATHS, load_cache, eeg_patient_frame

EEG_CACHE_PATH = CACHE_PATHS["alfred"]
EEG_CONVENTION = "zscore_window"   # per-window, per-channel z-score in microvolts

logger = logging.getLogger("exp3")


class TripleModalityDataset(Dataset):
    """Dataset combining text embeddings, EEG windows, and SMILES embeddings."""

    def __init__(
        self,
        patient_ids: List[str],
        text_embeddings: Dict[str, np.ndarray],
        eeg_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        smiles_embeddings: np.ndarray,
        smiles_indices: Dict[str, int],
        labels: Dict[str, int],
        asm_drugs: Dict[str, str],
        max_channels: int = N_CHANNELS,
    ):
        """Initialize dataset.

        Args:
            patient_ids: List of patient IDs to include.
            text_embeddings: Dict mapping patient ID to text embedding.
            eeg_data: Dict mapping patient ID to (windows, padding_mask).
            smiles_embeddings: SMILES embeddings array [n_drugs, embed_dim].
            smiles_indices: Dict mapping ASM name to embedding index.
            labels: Dict mapping patient ID to outcome label.
            asm_drugs: Dict mapping patient ID to ASM drug name.
            max_channels: Max number of EEG channels.
        """
        self.patient_ids = patient_ids
        self.text_embeddings = text_embeddings
        self.eeg_data = eeg_data
        self.smiles_embeddings = smiles_embeddings
        self.smiles_indices = smiles_indices
        self.labels = labels
        self.asm_drugs = asm_drugs
        self.max_channels = max_channels

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Returns:
            Tuple of:
            - text_emb: (text_dim,)
            - eeg_windows: (num_windows, max_channels, n_times)
            - padding_mask: (num_windows,) boolean, True for padded
            - smiles_emb: (smiles_dim,)
            - label: scalar
        """
        pid = self.patient_ids[idx]

        # Get text embedding
        text_emb = torch.from_numpy(self.text_embeddings[pid]).float()

        # Get EEG data
        windows, padding_mask = self.eeg_data[pid]
        n_windows, n_channels, n_times = windows.shape

        # Pad channels if needed
        if n_channels < self.max_channels:
            padded = np.zeros((n_windows, self.max_channels, n_times), dtype=np.float32)
            padded[:, :n_channels, :] = windows
            windows = padded

        eeg_windows = torch.from_numpy(windows).float()
        padding_mask = torch.from_numpy(padding_mask).bool()

        # Get SMILES embedding (mean fallback if the drug is unknown)
        smiles_emb = torch.from_numpy(
            smiles_vector(self.asm_drugs[pid], self.smiles_embeddings, self.smiles_indices)
        ).float()

        # Get label
        label = torch.tensor(self.labels[pid], dtype=torch.long)

        return text_emb, eeg_windows, padding_mask, smiles_emb, label


def load_asm_drug_names(filepath: Path = ASM_NAMES_FILE) -> List[str]:
    """Load ordered drug names from file."""
    with open(filepath, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_csv_data(filepath: Path = CSV_PATH, filter_outcome: bool = True) -> pd.DataFrame:
    """Load and preprocess the CSV data."""
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


def load_text_embeddings(
    text_model: str,
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Load text embeddings and align with patient IDs.

    Args:
        text_model: 'clinicalbert' or 'pubmedbert'
        df: Filtered DataFrame with patient info.

    Returns:
        Dict mapping patient ID to text embedding.
    """
    # Load embeddings (generated from same filtered CSV)
    emb_path = TEXT_EMBEDDINGS[text_model]
    all_embeddings = np.load(emb_path)

    # Load full CSV without outcome filtering to match embedding order
    df_all = load_csv_data(filter_outcome=False)

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


def prepare_data(
    text_model: str = "clinicalbert",
    smiles_model: str = "chemberta",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]], np.ndarray, Dict[str, int], pd.DataFrame]:
    """Prepare all data for training.

    The EEG windows come from the version-2 cache (``shared.eeg_cache``), normalised
    per window and channel at load time; the cohort is every Melbourne patient with a
    usable outcome, a cached recording and a text embedding.

    Args:
        text_model: 'clinicalbert' or 'pubmedbert'
        smiles_model: 'chemberta' or 'smilestrf'

    Returns:
        Tuple of (text_embeddings, eeg_data, smiles_embeddings, smiles_indices, df).
    """
    logger.info(f"Preparing data: text={text_model}, smiles={smiles_model}")
    eeg_data = load_cache(EEG_CACHE_PATH, EEG_CONVENTION)
    logger.info(f"Loaded EEG windows for {len(eeg_data)} patients from {EEG_CACHE_PATH}")
    df = eeg_patient_frame(eeg_data.keys())
    logger.info(f"Found {len(df)} patients with a cached recording and a usable outcome")

    # Load SMILES embeddings
    smiles_embeddings, smiles_indices = load_smiles_embeddings(smiles_model)
    logger.info(f"Loaded SMILES embeddings: shape={smiles_embeddings.shape}")

    # Load text embeddings
    text_embeddings = load_text_embeddings(text_model, df)
    logger.info(f"Loaded text embeddings for {len(text_embeddings)} patients")

    # Find intersection of all three modalities
    common_pids = set(text_embeddings.keys()) & set(eeg_data.keys())
    logger.info(f"Patients with all three modalities: {len(common_pids)}")

    # Filter df to common patients, then dedupe by pid before the fold split.
    df = dedupe_by_pid(df[df["pid"].astype(str).isin(common_pids)])

    # Filter embeddings to common patients
    text_embeddings = {pid: emb for pid, emb in text_embeddings.items() if pid in common_pids}
    eeg_data = {pid: data for pid, data in eeg_data.items() if pid in common_pids}

    # Log class distribution
    outcome_counts = df["outcome"].value_counts()
    logger.info(f"Outcome distribution: {dict(outcome_counts)}")

    return text_embeddings, eeg_data, smiles_embeddings, smiles_indices, df


def get_max_channels(eeg_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> int:
    """Get maximum number of channels across all EEG data."""
    return max(data[0].shape[1] for data in eeg_data.values())


def create_datasets(
    text_embeddings: Dict[str, np.ndarray],
    eeg_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    smiles_embeddings: np.ndarray,
    smiles_indices: Dict[str, int],
    df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    max_channels: int = None,
) -> Tuple[TripleModalityDataset, TripleModalityDataset]:
    """Create train and validation datasets from fold indices."""
    # Build lookup dicts
    labels = {str(row["pid"]): int(row["outcome"]) for _, row in df.iterrows()}
    asm_drugs = {str(row["pid"]): row["ASM"] for _, row in df.iterrows()}

    # Get patient IDs for each split
    pids = df["pid"].astype(str).values
    train_pids = [pids[i] for i in train_indices]
    val_pids = [pids[i] for i in val_indices]

    # Compute max channels if not provided
    if max_channels is None:
        max_channels = get_max_channels(eeg_data)

    # Create datasets
    train_dataset = TripleModalityDataset(
        patient_ids=train_pids,
        text_embeddings=text_embeddings,
        eeg_data=eeg_data,
        smiles_embeddings=smiles_embeddings,
        smiles_indices=smiles_indices,
        labels=labels,
        asm_drugs=asm_drugs,
        max_channels=max_channels,
    )

    val_dataset = TripleModalityDataset(
        patient_ids=val_pids,
        text_embeddings=text_embeddings,
        eeg_data=eeg_data,
        smiles_embeddings=smiles_embeddings,
        smiles_indices=smiles_indices,
        labels=labels,
        asm_drugs=asm_drugs,
        max_channels=max_channels,
    )

    return train_dataset, val_dataset


def test_data_pipeline():
    """Test the data pipeline."""
    logging.basicConfig(level=logging.INFO)
    print("Testing triple modality data pipeline...")

    # Prepare data
    text_emb, eeg_data, smiles_emb, smiles_idx, df = prepare_data(
        text_model="clinicalbert",
        smiles_model="chemberta",
    )

    print(f"\nDataset summary:")
    print(f"  Patients with all modalities: {len(df)}")
    print(f"  Text embeddings: {len(text_emb)}, dim={next(iter(text_emb.values())).shape}")
    print(f"  EEG data: {len(eeg_data)}")
    print(f"  SMILES embeddings: {smiles_emb.shape}")

    # Test dataset creation
    n = len(df)
    indices = np.arange(n)
    train_indices = indices[: int(0.8 * n)]
    val_indices = indices[int(0.8 * n) :]

    train_ds, val_ds = create_datasets(
        text_emb, eeg_data, smiles_emb, smiles_idx, df,
        train_indices, val_indices,
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val: {len(val_ds)}")

    # Test getting a sample
    text, eeg_windows, padding_mask, smiles, label = train_ds[0]
    print(f"\nSample shapes:")
    print(f"  Text embedding: {text.shape}")
    print(f"  EEG windows: {eeg_windows.shape}")
    print(f"  Padding mask: {padding_mask.shape}, valid: {(~padding_mask).sum()}")
    print(f"  SMILES embedding: {smiles.shape}")
    print(f"  Label: {label}")


if __name__ == "__main__":
    test_data_pipeline()
