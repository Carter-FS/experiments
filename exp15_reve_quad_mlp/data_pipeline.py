"""Data pipeline for Experiment 15: quad-modal fusion on stored EEG features.

Mirrors exp7_all_modalities/data_pipeline.py but replaces the raw-EEG cache load with
a stored per-window feature set (``shared.eeg_features``: REVE-base by default, or
LaBraM-base), written from the version-2 EEG cache by a frozen pretrained encoder.
Cohort intersection is the same logic (clinical outcome valid AND features available
AND text embedding available AND SMILES vocabulary entry available).

A feature file holds ``features`` float32 (n_patients, max_windows=120, dim), ``pids``
and ``valid_window_counts``; padded window positions are zero-filled and the loader
reconstructs the per-patient padding mask as ``mask[j] = (j >= valid_window_counts)``.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    ASM_NAMES_FILE,
    CSV_PATH,
    MAX_WINDOWS,
    OUTCOME_MAPPING,
    DEFAULT_FEATURE_SET,
    SMILES_EMBEDDINGS,
    TEXT_EMBEDDINGS,
)
from shared.cohort import dedupe_by_pid, filter_and_map_outcome, smiles_vector
from shared.eeg_features import load_features

# Reuse exp4's clinical preprocessor + cleaning utilities. No EEG cache is
# opened here: the REVE features were extracted from the version-2 cache
# (shared.eeg_cache), so the pids in the .npz already give the EEG-valid
# patient set.
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp4_baseline.data_pipeline import (  # noqa: E402
    ClinicalFeaturePreprocessor,
    clean_lesion_column,
    clean_outcome_column,
    clean_psy_column,
)

logger = logging.getLogger("exp15")


# ============================================================================
# Loading helpers (mostly delegate to exp7)
# ============================================================================

def load_asm_drug_names(filepath: Path = ASM_NAMES_FILE) -> List[str]:
    with open(filepath, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_smiles_embeddings(smiles_model: str) -> Tuple[np.ndarray, Dict[str, int]]:
    emb_path = SMILES_EMBEDDINGS[smiles_model]
    embeddings = np.load(emb_path)
    drug_names = load_asm_drug_names()
    index_map = {name: i for i, name in enumerate(drug_names)}
    return embeddings, index_map


def load_csv_for_text(filepath: Path = CSV_PATH, filter_outcome: bool = True) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df[df["eeg_report"].notna()].copy()
    df = df[df["eeg_report"].str.strip() != ""].copy()
    MIN_REPORT_LENGTH = 20
    df["report_length"] = df["eeg_report"].str.len()
    df = df[df["report_length"] >= MIN_REPORT_LENGTH].copy()
    error_patterns = ["Err:", "Exceed time window", "#N/A", "No EEG data"]
    for pattern in error_patterns:
        df = df[~df["eeg_report"].str.contains(pattern, na=False)]
    df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")
    if filter_outcome:
        df = df[df["outcome"].isin([1, 2])].copy()
        df["outcome"] = df["outcome"].map(OUTCOME_MAPPING).astype(int)
    df = df.reset_index(drop=True)
    return df


def load_text_embeddings(text_model: str, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    emb_path = TEXT_EMBEDDINGS[text_model]
    all_embeddings = np.load(emb_path)
    df_all = load_csv_for_text(filter_outcome=False)
    if len(all_embeddings) != len(df_all):
        raise ValueError(
            f"Text embeddings ({len(all_embeddings)}) don't match "
            f"CSV rows ({len(df_all)}). Regenerate embeddings."
        )
    pid_to_emb = {}
    for idx, row in df_all.iterrows():
        pid = str(row["pid"])
        pid_to_emb[pid] = all_embeddings[idx]
    text_embeddings = {}
    for _, row in df.iterrows():
        pid = str(row["pid"])
        if pid in pid_to_emb:
            text_embeddings[pid] = pid_to_emb[pid]
    return text_embeddings


def load_window_features(feature_set: str = DEFAULT_FEATURE_SET) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Per-window features and padding masks for every cached recording
    (``shared.eeg_features.load_features`` on the Melbourne cohort)."""
    return load_features(feature_set, "alfred")


# ============================================================================
# Dataset class
# ============================================================================


class ReveQuadDataset(Dataset):
    """Quad-modal dataset with pre-computed REVE per-window features.

    Yields the same 6-tuple shape as exp7's QuadModalityDataset so the
    existing training loops can be reused with minimal changes:
        (clinical, text, eeg_windows, padding_mask, smiles, label)
    where eeg_windows is (max_windows, 512) - REVE features - rather
    than (max_windows, channels, time) raw EEG windows.
    """

    def __init__(
        self,
        clinical_features: np.ndarray,
        text_embeddings: np.ndarray,
        reve_windows: List[np.ndarray],
        padding_masks: List[np.ndarray],
        smiles_embeddings: np.ndarray,
        smiles_indices: Dict[str, int],
        asm_drugs: List[str],
        labels: np.ndarray,
        pids: List = None,
        return_pid: bool = False,
    ):
        self.clinical_features = torch.from_numpy(clinical_features).float()
        self.text_embeddings = torch.from_numpy(text_embeddings).float()
        self.reve_windows = reve_windows
        self.padding_masks = padding_masks
        # width of the stored per-window features (512 REVE-base, 200 LaBraM-base)
        widths = {int(w.shape[-1]) for w in reve_windows}
        if len(widths) != 1:
            raise ValueError(f"per-window feature widths differ across patients: {sorted(widths)}")
        self.feature_dim = widths.pop()
        self.smiles_embeddings = smiles_embeddings
        self.smiles_indices = smiles_indices
        self.asm_drugs = asm_drugs
        self.labels = torch.from_numpy(labels).long()
        self.pids = [str(p) for p in pids] if pids is not None else None
        self.return_pid = return_pid
        if self.return_pid and self.pids is None:
            raise ValueError("return_pid=True requires pids to be supplied.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple:
        clinical = self.clinical_features[idx]
        text = self.text_embeddings[idx]
        eeg = torch.from_numpy(self.reve_windows[idx]).float()
        mask = torch.from_numpy(self.padding_masks[idx]).bool()

        smiles = torch.from_numpy(
            smiles_vector(self.asm_drugs[idx], self.smiles_embeddings, self.smiles_indices)
        ).float()

        label = self.labels[idx]

        if self.return_pid:
            return clinical, text, eeg, mask, smiles, label, self.pids[idx]
        return clinical, text, eeg, mask, smiles, label


# ============================================================================
# Cohort preparation + dataset construction
# ============================================================================


def prepare_quad_modality_data_reve(
    text_model: str = "clinicalbert",
    smiles_model: str = "chemberta",
    feature_set: str = DEFAULT_FEATURE_SET,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    Dict[str, int],
    Dict[str, np.ndarray],
    Dict[str, Tuple[np.ndarray, np.ndarray]],
]:
    """Build the quad-modal cohort with stored EEG features (``feature_set``).

    Returns:
        (df, smiles_embeddings, smiles_indices, text_embeddings, reve_data)
    """
    logger.info(f"Preparing exp15 quad-modal data ({feature_set} + {text_model} + {smiles_model})")

    # Load full clinical CSV (no outcome filter yet)
    df = pd.read_csv(CSV_PATH)

    # Outcome filter + clean clinical columns
    df = filter_and_map_outcome(df)
    df = clean_psy_column(df)
    df = clean_lesion_column(df)
    df = df.reset_index(drop=True)
    logger.info(f"Patients with valid clinical+outcome: {len(df)}")

    # Load SMILES embeddings
    smiles_embeddings, smiles_indices = load_smiles_embeddings(smiles_model)

    # Load text embeddings
    text_embeddings = load_text_embeddings(text_model, df)
    logger.info(f"Patients with valid text embeddings: {len(text_embeddings)}")

    # Load REVE features
    reve_data = load_window_features(feature_set)

    # Intersect clinical + text + REVE (SMILES attaches to every patient via
    # smiles_vector), then dedupe by pid before the fold split.
    valid_mask = (
        df["pid"].astype(str).isin(text_embeddings.keys())
        & df["pid"].astype(str).isin(reve_data.keys())
    )
    df = dedupe_by_pid(df[valid_mask])
    logger.info(f"Patients with all 4 modalities: {len(df)}")

    return df, smiles_embeddings, smiles_indices, text_embeddings, reve_data


def create_reve_quad_datasets(
    df: pd.DataFrame,
    smiles_embeddings: np.ndarray,
    smiles_indices: Dict[str, int],
    text_embeddings: Dict[str, np.ndarray],
    reve_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    return_pid: bool = False,
) -> Tuple[ReveQuadDataset, ReveQuadDataset, ClinicalFeaturePreprocessor]:
    """Create train/val ReveQuadDataset pair with fold-fitted clinical preproc."""
    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()

    # Fit clinical preprocessor on training fold only
    preprocessor = ClinicalFeaturePreprocessor()
    train_clinical = preprocessor.fit_transform(train_df)
    val_clinical = preprocessor.transform(val_df)

    # Order text/REVE/ASM/label by training-fold and val-fold patient IDs
    train_text = np.array([text_embeddings[str(pid)] for pid in train_df["pid"]])
    val_text = np.array([text_embeddings[str(pid)] for pid in val_df["pid"]])

    train_windows = [reve_data[str(pid)][0] for pid in train_df["pid"]]
    train_masks = [reve_data[str(pid)][1] for pid in train_df["pid"]]
    val_windows = [reve_data[str(pid)][0] for pid in val_df["pid"]]
    val_masks = [reve_data[str(pid)][1] for pid in val_df["pid"]]

    train_asm = train_df["ASM"].tolist()
    val_asm = val_df["ASM"].tolist()
    train_labels = train_df["outcome"].values
    val_labels = val_df["outcome"].values

    train_pids = train_df["pid"].astype(str).tolist()
    val_pids = val_df["pid"].astype(str).tolist()

    train_dataset = ReveQuadDataset(
        clinical_features=train_clinical,
        text_embeddings=train_text,
        reve_windows=train_windows,
        padding_masks=train_masks,
        smiles_embeddings=smiles_embeddings,
        smiles_indices=smiles_indices,
        asm_drugs=train_asm,
        labels=train_labels,
        pids=train_pids,
        return_pid=return_pid,
    )
    val_dataset = ReveQuadDataset(
        clinical_features=val_clinical,
        text_embeddings=val_text,
        reve_windows=val_windows,
        padding_masks=val_masks,
        smiles_embeddings=smiles_embeddings,
        smiles_indices=smiles_indices,
        asm_drugs=val_asm,
        labels=val_labels,
        pids=val_pids,
        return_pid=return_pid,
    )
    return train_dataset, val_dataset, preprocessor
