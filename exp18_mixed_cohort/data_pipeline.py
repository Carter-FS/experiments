"""Pooled Melbourne + HEP1 cohort for Experiment 18.

One frame per configuration: the deduplicated Melbourne cohort and HEP1,
restricted to patients who have every modality the configuration uses, with a
``cohort`` column ("MEL" / "HEP") and cohort-prefixed pids (the two cohorts
use unrelated pid schemes; Melbourne pids keep their leading zeros as
strings). Modality tensors are row-aligned with the frame. Clinical features
are not precomputed: every arm refits the preprocessor on its own fit rows
(``clinical_features``).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from shared.cv_splits import joint_key
from shared.hep_cohort import (
    CROSS_COHORT_DROP,
    build_smiles_feature_matrix,
    build_smiles_lookup,
    load_alfred,
    load_alfred_text_aligned,
    load_hep,
    load_hep_text_embeddings,
)
from shared.portable_models import (
    filter_to_intersection,
    load_eeg_cache,
    refit_clinical,
    stack_eeg_for_pids,
)

from .config import HEP_EEG_CACHE, LF_INPUTS, MEL_EEG_CACHE, MODALITIES, RMH_PREFIX, STRATIFY


@dataclass
class PooledCohort:
    config: str
    df: pd.DataFrame            # pooled rows; columns include pid, source_pid, cohort, outcome
    modalities: dict            # name -> tensor, row-aligned (clinical added per arm)
    labels: torch.Tensor        # outcome, long
    key: np.ndarray             # outcome x cohort stratification key

    def cohort_mask(self, cohort: str) -> np.ndarray:
        return (self.df["cohort"] == cohort).to_numpy()


def _tag(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    df = df.copy()
    df["source_pid"] = df["pid"].astype(str)
    df["pid"] = cohort + "_" + df["source_pid"]
    df["cohort"] = cohort
    return df


def load_pooled(config: str, exclude_rmh: bool = False, exclude_hep_pids=(),
                hep_outcome: str = "provided") -> PooledCohort:
    """Build the pooled cohort for one configuration.

    ``exclude_rmh`` drops every HEP1 Royal Melbourne Hospital patient (the
    sensitivity analysis); ``exclude_hep_pids`` drops confirmed cross-cohort
    duplicates from HEP1 (they stay in Melbourne); ``hep_outcome`` selects the
    provided or 12-month harmonised HEP1 label (analysis plan B.5).
    """
    mods = MODALITIES[config]
    mel, hep = load_alfred(), load_hep(hep_outcome)
    if len(exclude_hep_pids):
        # Accept source pids or pooled "HEP_" pids; every listed pid must be a
        # HEP1 patient, so a typo or the wrong id scheme cannot exclude nobody.
        wanted = {str(p).removeprefix("HEP_") for p in exclude_hep_pids}
        unknown = sorted(wanted - set(hep["pid"].astype(str)))
        if unknown:
            raise ValueError(f"{len(unknown)} listed duplicate pid(s) are not HEP1 pids")
        hep = hep[~hep["pid"].astype(str).isin(wanted)]
    if exclude_rmh:
        hep = hep[~hep["pid"].astype(str).str.startswith(RMH_PREFIX)]

    text_maps = None
    if "text" in mods:
        mel_text_df, mel_text = load_alfred_text_aligned()
        mel_map = {str(p): mel_text[i] for i, p in enumerate(mel_text_df["pid"])}
        hep_map, _ = load_hep_text_embeddings()
        mel, hep = filter_to_intersection(mel, mel_map), filter_to_intersection(hep, hep_map)
        text_maps = {"MEL": mel_map, "HEP": hep_map}

    eeg_caches = None
    if "eeg" in mods:
        eeg_caches = {"MEL": load_eeg_cache(MEL_EEG_CACHE), "HEP": load_eeg_cache(HEP_EEG_CACHE)}
        mel = filter_to_intersection(mel, eeg_caches["MEL"])
        hep = filter_to_intersection(hep, eeg_caches["HEP"])

    df = pd.concat([_tag(mel, "MEL"), _tag(hep, "HEP")], ignore_index=True)
    assert df["pid"].is_unique, "pooled pids must be unique"

    modalities: dict[str, torch.Tensor] = {}
    if "smiles" in mods:
        modalities["smiles"] = torch.from_numpy(build_smiles_feature_matrix(df, build_smiles_lookup()))
    if text_maps is not None:
        modalities["text"] = torch.from_numpy(np.vstack(
            [text_maps[c][p] for c, p in zip(df["cohort"], df["source_pid"])]
        ).astype(np.float32))
    if eeg_caches is not None:
        parts = [stack_eeg_for_pids(eeg_caches[c], df.loc[df["cohort"] == c, "source_pid"].tolist())
                 for c in ("MEL", "HEP")]
        modalities["eeg_windows"] = torch.cat([w for w, _ in parts])
        modalities["eeg_mask"] = torch.cat([m for _, m in parts])
        # The raw caches are ~4 GB; only the stacked cohort windows are needed.
        del eeg_caches, parts
        gc.collect()

    if config in LF_INPUTS:
        # Fold-independent exp19 text embeddings (standardised per arm at training time).
        from exp19_serialised_clinical.run_experiments import embed_variant, lookup
        from exp19_serialised_clinical.texts import texts
        for name in LF_INPUTS[config]:
            if name.startswith("emb:"):
                _, variant, encoder = name.split(":")
                modalities[name] = embed_variant(encoder, "mean", df, variant)
            elif name.startswith("rep:"):
                modalities[name] = lookup(name[4:], "mean", texts(df, "rep"))

    labels = torch.from_numpy(df["outcome"].to_numpy().astype(np.int64))
    return PooledCohort(config, df, modalities, labels, joint_key(df, STRATIFY))


def clinical_features(pooled: PooledCohort, fit_idx: np.ndarray) -> torch.Tensor:
    """Clinical tensor for every pooled row, preprocessor fitted on ``fit_idx``
    only, without the features constant in HEP1 (analysis plan B.4)."""
    clinical, _ = refit_clinical(pooled.df, pooled.df, fit_idx, drop=CROSS_COHORT_DROP)
    return clinical
