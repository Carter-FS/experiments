"""Melbourne (Alfred + RMH) and HEP1 cohort loading and harmonisation.

Single source of truth for everything that makes the two cohorts comparable:
HEP1 string-to-Alfred-code clinical maps, ASM name normalisation, the
ChemBERTa SMILES lookup, the mean-pooled ClinicalBERT report embeddings and the
shared clinical preprocessor. Used by the HEP1 external-validation scripts in
thesisStandalone/analysis and by exp18_mixed_cohort. Moved here unchanged from
thesisStandalone/analysis/hep_external_validation.py.

Data never lives in this repo: CSVs resolve through ``asm_data`` (``$ASM_DATA_DIR``
or the repo sibling ``../asm_data``), embeddings through ``outputs/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent


def find_asm_data_dir() -> Path:
    """``$ASM_DATA_DIR`` if set, else ``asm_data`` beside the experiments repo."""
    env = os.environ.get("ASM_DATA_DIR")
    if env and Path(env).exists():
        return Path(env)
    sibling = EXPERIMENTS_ROOT.parent / "asm_data"
    if sibling.exists():
        return sibling
    raise FileNotFoundError(
        "asm_data not found: set ASM_DATA_DIR or place asm_data beside the experiments repo."
    )


_ASM_DATA_DIR = find_asm_data_dir()
ALFRED_CSV = _ASM_DATA_DIR / "alfred_1st_regimen.csv"
HEP_CSV = _ASM_DATA_DIR / "hep_1st_regimen.csv"
_OUTPUTS = EXPERIMENTS_ROOT / "outputs"
ASM_DRUG_NAMES_PATH = _OUTPUTS / "asm_drug_names.txt"
CHEMBERTA_PATH = _OUTPUTS / "chemberta_asm_embeddings.npy"
ALFRED_TEXT_EMB_PATH = _OUTPUTS / "bert_alfred_1stregimen_eeg_embeddings.npy"
HEP_TEXT_EMB_PATH = _OUTPUTS / "hep_clinicalbert_eeg_embeddings.npy"
HEP_TEXT_PIDS_PATH = _OUTPUTS / "hep_text_pids.txt"


# -----------------------------------------------------------------------
# Cohort harmonisation
# -----------------------------------------------------------------------

HEP_ASM_TO_ALFRED_ABBREV = {
    "LEVETIRACETAM": "LEV", "LAMOTRIGINE": "LTG", "OXCARBAZEPINE": "OXC",
    "CARBAMAZEPINE": "CBZ", "TOPIRAMATE": "TPM", "PHENYTOIN": "PTN",
    "VALPROATE": "VPA", "VALPROIC ACID": "VPA", "ZONISAMIDE": "ZNS",
    "LACOSAMIDE": "LAC", "BRIVARACETAM": "BRV", "GABAPENTIN": "GBP",
    "PREGABALIN": "PGB", "CLOBAZAM": "CLB", "CLONAZEPAM": "CLZ",
}

SMILES_VOCAB_TO_ABBREV = {
    "Levetiracetam": "LEV", "Valproic_acid": "VPA", "Carbamazepine": "CBZ",
    "Lamotrigine": "LTG", "Oxcarbazepine": "OXC", "Topiramate": "TPM",
    "Phenytoin": "PTN", "Lacosamide": "LAC", "Brivaracetam": "BRV",
    "Perampanel": "PER", "Zonisamide": "ZNS", "Gabapentin": "GBP",
    "Pregabalin": "PGB", "Clobazam": "CLB", "Clonazepam": "CLZ",
}

# Binary features constant in HEP1 (every patient "No", "No", focal). In any
# model that sees both cohorts they only identify the cohort, so exp18 and the
# HEP transfer scripts drop them (analysis plan B.4).
CROSS_COHORT_DROP = ("drug", "alcohol", "focal")

ALFRED_TRAINING_ASMS = {"LEV", "VPA", "CBZ", "LTG", "PTN", "TPM"}


def normalise_alfred_asm(value: object) -> str:
    if not isinstance(value, str):
        return ""
    v = value.strip().upper()
    # Alfred 'cBZ' typo
    if v == "CBZ" or v == "":
        return v
    return v


def load_alfred() -> pd.DataFrame:
    from exp4_baseline.data_pipeline import (
        clean_outcome_column, clean_psy_column, clean_lesion_column,
    )
    df = pd.read_csv(ALFRED_CSV)
    df = clean_psy_column(df)
    df = clean_lesion_column(df)
    df = clean_outcome_column(df)
    df["ASM"] = df["ASM"].apply(normalise_alfred_asm)
    # De-duplicate by pid (leakage fix): one row per patient before the CV split,
    # so the Alfred ensemble matches the corrected 198-patient training cohort.
    from shared.cohort import dedupe_by_pid
    df = dedupe_by_pid(df.reset_index(drop=True))
    return df.reset_index(drop=True)


# HEP1 clinical features are stored as strings; Alfred's are numeric. These maps
# harmonise HEP1 to Alfred's exact CSV coding (verified against Table 1 /
# thesisStandalone/analysis/build_cohort_table.py). Without this, the shared ClinicalFeaturePreprocessor
# (fit on numeric Alfred) coerces every HEP string to NaN via
# pd.to_numeric(errors="coerce") and mode-imputes it, silently zeroing 15 of 19
# clinical inputs on HEP (only the age bins survive). MRI/EEG 3-way coding is
# verified against Alfred's raw report text: code 1 = NORMAL, 2 = non-epileptiform
# abnormality, 3 = epileptiform/abnormal (Table 1 / build_cohort_table.py use the
# same polarity since thesisStandalone 2eb26a1). The preprocessor one-hots the
# three levels (analysis plan B.3), so HEP must use the same codes: Normal -> 1.
_HEP_YESNO = {"No": 0.0, "Yes": 1.0}
_HEP_CLINICAL_MAPS = {
    "sex": {"Male": 0.0, "Female": 1.0},
    "pretrt_sz_5": {"<=5": 0.0, ">5": 1.0},
    "focal": {"Focal": 1.0, "Generalised": 0.0, "Yes": 1.0, "No": 0.0},
    "fam_hx": _HEP_YESNO, "febrile": _HEP_YESNO, "ci": _HEP_YESNO,
    "birth_t": _HEP_YESNO, "head": _HEP_YESNO, "drug": _HEP_YESNO,
    "alcohol": _HEP_YESNO, "cvd": _HEP_YESNO, "psy": _HEP_YESNO, "ld": _HEP_YESNO,
    "lesion": {"Normal": 1.0, "Abnormal": 2.0, "Epileptiogenic": 3.0,
               "Non-epileptiform abnormality": 2.0, "Epileptiform abnormality": 3.0},
    "eeg_cat": {"Normal": 1.0, "Abnormal": 2.0, "Epileptiform": 3.0,
                "Non-epileptiform abnormality": 2.0, "Epileptiform abnormality": 3.0},
}


def _harmonise_hep_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """Map HEP1 string clinical features to Alfred's numeric coding (in place-safe)."""
    for col, mapping in _HEP_CLINICAL_MAPS.items():
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            raw = df[col].astype(str).str.strip()
            mapped = raw.map(mapping)
            # Guard against a future data refresh introducing an unmapped label
            # (which would silently NaN -> mode-impute and quietly degrade a feature).
            unmapped = sorted(set(raw[mapped.isna() & df[col].notna()]))
            if unmapped:
                sys.stderr.write(f"WARN: unmapped HEP1 '{col}' values, will be mode-imputed: {unmapped}\n")
            df[col] = mapped
    return df


def load_hep() -> pd.DataFrame:
    df = pd.read_csv(HEP_CSV)
    df = df.rename(columns={"patient": "pid", "age": "age_init"})
    df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")
    df = df[df["outcome"].isin([0, 1])].copy()
    df["outcome"] = df["outcome"].astype(int)
    df["ASM"] = df["ASM"].astype(str).str.strip().str.upper().map(
        lambda a: HEP_ASM_TO_ALFRED_ABBREV.get(a, a)
    )
    df = _harmonise_hep_clinical(df)
    if "mri_report" not in df.columns:
        df["mri_report"] = pd.NA
    df = df.reset_index(drop=True)
    return df


# -----------------------------------------------------------------------
# Embedding lookups
# -----------------------------------------------------------------------

def build_smiles_lookup() -> dict[str, np.ndarray]:
    names = [n.strip() for n in ASM_DRUG_NAMES_PATH.read_text().splitlines() if n.strip()]
    embeddings = np.load(CHEMBERTA_PATH)
    lookup: dict[str, np.ndarray] = {}
    for name, emb in zip(names, embeddings):
        abbrev = SMILES_VOCAB_TO_ABBREV.get(name)
        if abbrev:
            lookup[abbrev] = emb
    lookup["__MEAN__"] = embeddings.mean(axis=0)
    return lookup


def lookup_smiles(asm: str, smiles_lookup: dict[str, np.ndarray]) -> np.ndarray:
    return smiles_lookup.get(asm, smiles_lookup["__MEAN__"])


def build_smiles_feature_matrix(df: pd.DataFrame, smiles_lookup: dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([lookup_smiles(a, smiles_lookup) for a in df["ASM"]]).astype(np.float32)


def load_alfred_text_aligned() -> tuple[pd.DataFrame, np.ndarray]:
    """Return Alfred text-aligned (df, embeddings) using exp5's pipeline.

    The 122-row text embedding file is row-aligned to
    ``load_csv_for_text(filter_outcome=False)`` (i.e. text-report-only
    Alfred subset). We use exp5's existing ``prepare_clinical_text_data``
    to get the aligned (df, dict[pid -> emb]) pair, then convert the
    dict to a row-aligned array.
    """
    from exp5_clinical_fusion.data_pipeline import prepare_clinical_text_data
    df, text_dict = prepare_clinical_text_data(text_model="clinicalbert")
    emb = np.vstack([text_dict[str(p)] for p in df["pid"]]).astype(np.float32)
    return df.reset_index(drop=True), emb


def load_hep_text_embeddings() -> tuple[dict[str, np.ndarray], list[str]]:
    if not HEP_TEXT_EMB_PATH.exists():
        raise FileNotFoundError(
            f"HEP text embeddings not found at {HEP_TEXT_EMB_PATH}; "
            "run analysis/hep_text_embeddings.py first."
        )
    emb = np.load(HEP_TEXT_EMB_PATH)
    pids = HEP_TEXT_PIDS_PATH.read_text().strip().splitlines()
    return {p: emb[i] for i, p in enumerate(pids)}, pids


# -----------------------------------------------------------------------
# Clinical feature builder (reuse exp4_baseline preprocessor)
# -----------------------------------------------------------------------

def build_clinical_features(df: pd.DataFrame, preprocessor=None, drop: tuple[str, ...] = ()):
    """Clinical tensor for ``df``. With no ``preprocessor`` one is fitted on
    ``df`` itself, which is only correct when ``df`` is the training data."""
    from exp4_baseline.data_pipeline import ClinicalFeaturePreprocessor
    if preprocessor is None:
        preprocessor = ClinicalFeaturePreprocessor(drop=drop)
        preprocessor.fit(df)
    feature_matrix = preprocessor.transform(df)
    features = torch.from_numpy(feature_matrix).float()
    labels = torch.from_numpy(df["outcome"].values.astype(np.int64)).long()
    return features, labels, preprocessor
