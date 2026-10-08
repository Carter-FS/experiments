"""Data pipeline for Experiment 2: EEG + SMILES fusion."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    ASM_NAME_MAP,
    CHEMBERTA_EMBEDDINGS,
    SMILESTRF_EMBEDDINGS,
)
from shared.cohort import smiles_vector
from shared.eeg_cache import CACHE_PATHS, load_cache, eeg_patient_frame

EEG_CACHE_PATH = CACHE_PATHS["alfred"]
EEG_CONVENTION = "zscore_window"   # per-window, per-channel z-score in microvolts

logger = logging.getLogger("exp2")


class EEGSMILESDataset(Dataset):
    """Dataset combining EEG windows and SMILES embeddings."""

    def __init__(
        self,
        patient_ids: List[str],
        eeg_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        smiles_embeddings: np.ndarray,
        smiles_indices: Dict[str, int],
        labels: Dict[str, int],
        asm_drugs: Dict[str, str],
        max_channels: int = None,
    ):
        """Initialize dataset.

        Args:
            patient_ids: List of patient IDs to include.
            eeg_data: Dict mapping patient ID to (windows, padding_mask).
            smiles_embeddings: SMILES embeddings array [n_drugs, embed_dim].
            smiles_indices: Dict mapping ASM name to embedding index.
            labels: Dict mapping patient ID to outcome label.
            asm_drugs: Dict mapping patient ID to ASM drug name.
            max_channels: Max number of EEG channels (pad smaller to this).
        """
        self.patient_ids = patient_ids
        self.eeg_data = eeg_data
        self.smiles_embeddings = smiles_embeddings
        self.smiles_indices = smiles_indices
        self.labels = labels
        self.asm_drugs = asm_drugs

        # Determine max channels if not provided
        if max_channels is None:
            self.max_channels = max(
                eeg_data[pid][0].shape[1] for pid in patient_ids
            )
        else:
            self.max_channels = max_channels

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Returns:
            Tuple of:
            - eeg_windows: (num_windows, max_channels, n_times)
            - padding_mask: (num_windows,) boolean, True for padded
            - smiles_embedding: (embed_dim,)
            - label: scalar
        """
        pid = self.patient_ids[idx]

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

        return eeg_windows, padding_mask, smiles_emb, label


def load_smiles_embeddings(smiles_model: str = "chemberta") -> Tuple[np.ndarray, Dict[str, int]]:
    """Load precomputed SMILES embeddings from Experiment 1.

    Args:
        smiles_model: Which SMILES model embeddings to load.

    Returns:
        Tuple of (embeddings array, index mapping dict).
    """
    if smiles_model == "chemberta":
        emb_path = CHEMBERTA_EMBEDDINGS
    elif smiles_model == "smilestrf":
        emb_path = SMILESTRF_EMBEDDINGS
    else:
        raise ValueError(f"Unknown SMILES model: {smiles_model}")

    embeddings = np.load(emb_path)

    # Create index mapping (order from ASM_NAME_MAP)
    index_map = {name: i for i, name in enumerate(ASM_NAME_MAP.values())}

    return embeddings, index_map


def prepare_data(
    smiles_model: str = "chemberta",
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], np.ndarray, Dict[str, int], pd.DataFrame]:
    """Prepare all data for training.

    The EEG windows come from the version-2 cache (``shared.eeg_cache``), normalised
    per window and channel at load time, and the cohort is every Melbourne patient
    with a usable outcome and a recording in that cache.

    Args:
        smiles_model: Which SMILES model to use ('chemberta' or 'smilestrf').

    Returns:
        Tuple of (eeg_data, smiles_embeddings, smiles_indices, patient_df).
    """
    logger.info("Preparing data...")
    eeg_data = load_cache(EEG_CACHE_PATH, EEG_CONVENTION)
    logger.info(f"Loaded EEG windows for {len(eeg_data)} patients from {EEG_CACHE_PATH}")

    logger.info(f"Loading SMILES embeddings for model: {smiles_model}")
    smiles_embeddings, smiles_indices = load_smiles_embeddings(smiles_model)
    logger.info(f"Loaded SMILES embeddings: shape={smiles_embeddings.shape}")

    # One row per patient with a usable outcome and a cached recording.
    df = eeg_patient_frame(eeg_data.keys())
    logger.info(f"Final dataset: {len(df)} patients")

    # Log class distribution
    outcome_counts = df["outcome"].value_counts()
    logger.info(f"Outcome distribution: {dict(outcome_counts)}")

    return eeg_data, smiles_embeddings, smiles_indices, df


def get_max_channels(eeg_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> int:
    """Get maximum number of channels across all EEG data."""
    return max(data[0].shape[1] for data in eeg_data.values())


def create_datasets(
    eeg_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    smiles_embeddings: np.ndarray,
    smiles_indices: Dict[str, int],
    df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    max_channels: int = None,
) -> Tuple[EEGSMILESDataset, EEGSMILESDataset]:
    """Create train and validation datasets from fold indices.

    Args:
        eeg_data: Dict mapping patient ID to (windows, padding_mask).
        smiles_embeddings: SMILES embeddings array.
        smiles_indices: Dict mapping ASM name to embedding index.
        df: DataFrame with patient info.
        train_indices: Array of training sample indices.
        val_indices: Array of validation sample indices.
        max_channels: Maximum number of EEG channels (computed if None).

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    # Build lookup dicts
    labels = {str(row["pid"]): int(row["outcome"]) for _, row in df.iterrows()}
    asm_drugs = {str(row["pid"]): row["ASM"] for _, row in df.iterrows()}

    # Get patient IDs for each split
    pids = df["pid"].astype(str).values
    train_pids = [pids[i] for i in train_indices]
    val_pids = [pids[i] for i in val_indices]

    # Compute max channels if not provided (use all data for consistency)
    if max_channels is None:
        max_channels = get_max_channels(eeg_data)

    # Create datasets
    train_dataset = EEGSMILESDataset(
        patient_ids=train_pids,
        eeg_data=eeg_data,
        smiles_embeddings=smiles_embeddings,
        smiles_indices=smiles_indices,
        labels=labels,
        asm_drugs=asm_drugs,
        max_channels=max_channels,
    )

    val_dataset = EEGSMILESDataset(
        patient_ids=val_pids,
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
    print("Testing data pipeline...")

    # Prepare data (use cache)
    eeg_data, smiles_embeddings, smiles_indices, df = prepare_data(
        smiles_model="chemberta",
    )

    print(f"\nDataset summary:")
    print(f"  Patients with EEG: {len(eeg_data)}")
    print(f"  SMILES embeddings: {smiles_embeddings.shape}")
    print(f"  Outcome distribution:")
    print(df["outcome"].value_counts())

    # Test dataset creation
    n = len(df)
    indices = np.arange(n)
    train_indices = indices[:int(0.8 * n)]
    val_indices = indices[int(0.8 * n):]

    train_ds, val_ds = create_datasets(
        eeg_data, smiles_embeddings, smiles_indices, df,
        train_indices, val_indices,
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val: {len(val_ds)}")

    # Test getting a sample
    eeg_windows, padding_mask, smiles_emb, label = train_ds[0]
    print(f"\nSample shapes:")
    print(f"  EEG windows: {eeg_windows.shape}")
    print(f"  Padding mask: {padding_mask.shape}, valid windows: {(~padding_mask).sum()}")
    print(f"  SMILES embedding: {smiles_emb.shape}")
    print(f"  Label: {label}")


if __name__ == "__main__":
    test_data_pipeline()
