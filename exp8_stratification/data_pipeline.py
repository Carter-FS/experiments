"""Data pipeline for Experiment 8: Stratification Analysis.

Reuses exp7 data pipeline with improved data cleaning and returns
the dataframe for stratification purposes.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from exp7_all_modalities.data_pipeline import (
    QuadModalityDataset,
    load_eeg_data,
    load_smiles_embeddings,
    load_text_embeddings,
)
from exp7_all_modalities.config import (
    ASM_NAME_MAPPING,
    CSV_PATH,
    OUTCOME_MAPPING,
)
from exp4_baseline.data_pipeline import ClinicalFeaturePreprocessor

from .data_cleaning import load_and_clean_data

logger = logging.getLogger("exp8")


def prepare_quad_modality_data_with_df(
    text_model: str = "clinicalbert",
    smiles_model: str = "chemberta",
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    Dict[str, int],
    Dict[str, np.ndarray],
    Dict[str, Tuple[np.ndarray, np.ndarray]],
]:
    """Prepare data for all 4 modalities with cleaned dataframe.

    Similar to exp7 but uses improved data cleaning and returns
    the dataframe for stratification.

    Args:
        text_model: 'clinicalbert' or 'pubmedbert'
        smiles_model: 'chemberta'

    Returns:
        Tuple of (df, smiles_embeddings, smiles_indices, text_embeddings, eeg_data).
        The df contains cleaned data suitable for stratification.
    """
    logger.info(f"Preparing quad modality data: {text_model}, {smiles_model}")

    # Step 1: the EEG cohort is every patient with a recording in the version-2 cache
    eeg_data = load_eeg_data()
    valid_pids = set(eeg_data.keys())
    logger.info(f"Loaded EEG windows for {len(valid_pids)} patients")

    # Step 2: Load and clean clinical data
    df, cleaning_report = load_and_clean_data()
    logger.info(f"Loaded {len(df)} patients after cleaning")

    # Filter to patients with valid EEG files
    df = df[df["pid"].astype(str).isin(valid_pids)].copy()
    logger.info(f"Patients with valid EEG: {len(df)}")

    # Load SMILES embeddings
    smiles_embeddings, smiles_indices = load_smiles_embeddings(smiles_model)
    logger.info(f"Loaded SMILES embeddings: shape={smiles_embeddings.shape}")

    # Load text embeddings
    text_embeddings = load_text_embeddings(text_model, df)
    logger.info(f"Loaded text embeddings for {len(text_embeddings)} patients")

    # Load cached EEG data

    # Filter to patients with ALL 4 modalities
    valid_rows = []
    for idx, row in df.iterrows():
        pid = str(row["pid"])
        has_text = pid in text_embeddings
        has_eeg = pid in eeg_data
        has_asm = row["asm_1"] is not None and pd.notna(row["asm_1"])

        if has_text and has_eeg and has_asm:
            valid_rows.append(idx)

    df = df.loc[valid_rows].copy().reset_index(drop=True)
    logger.info(f"Final dataset: {len(df)} patients with all 4 modalities")

    return df, smiles_embeddings, smiles_indices, text_embeddings, eeg_data


def create_dataset_from_indices(
    df: pd.DataFrame,
    indices: np.ndarray,
    smiles_embeddings: np.ndarray,
    smiles_indices: Dict[str, int],
    text_embeddings: Dict[str, np.ndarray],
    eeg_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> QuadModalityDataset:
    """Create a QuadModalityDataset from dataframe indices.

    Args:
        df: Full dataframe.
        indices: Row indices to include.
        smiles_embeddings: SMILES embedding array.
        smiles_indices: Drug name -> index mapping.
        text_embeddings: PID -> text embedding mapping.
        eeg_data: PID -> (windows, mask) mapping.

    Returns:
        QuadModalityDataset for the specified indices.
    """
    subset_df = df.iloc[indices].copy()

    # Preprocess clinical features
    preprocessor = ClinicalFeaturePreprocessor()
    clinical_features = preprocessor.fit_transform(subset_df)

    # Gather text embeddings
    text_emb_list = []
    for _, row in subset_df.iterrows():
        pid = str(row["pid"])
        text_emb_list.append(text_embeddings[pid])
    text_emb_array = np.stack(text_emb_list)

    # Gather EEG data
    eeg_windows_list = []
    padding_masks_list = []
    for _, row in subset_df.iterrows():
        pid = str(row["pid"])
        windows, mask = eeg_data[pid]
        eeg_windows_list.append(windows)
        padding_masks_list.append(mask)

    # Get ASM drugs and labels
    asm_drugs = subset_df["asm_1"].tolist()
    labels = subset_df["outcome"].values

    return QuadModalityDataset(
        clinical_features=clinical_features,
        text_embeddings=text_emb_array,
        eeg_windows=eeg_windows_list,
        padding_masks=padding_masks_list,
        smiles_embeddings=smiles_embeddings,
        smiles_indices=smiles_indices,
        asm_drugs=asm_drugs,
        labels=labels,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test data loading
    df, smiles_emb, smiles_idx, text_emb, eeg_data = prepare_quad_modality_data_with_df()

    print(f"\nDataset summary:")
    print(f"  Patients: {len(df)}")
    print(f"  Outcome distribution: {df['outcome'].value_counts().to_dict()}")
    print(f"  Focal distribution: {df['focal'].value_counts().to_dict()}")
    print(f"  Sex distribution: {df['sex'].value_counts().to_dict()}")
