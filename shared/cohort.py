"""Single source of truth for cohort construction.

Centralises logic that was copied across the exp1..exp15 data pipelines:
outcome filtering/mapping, ASM name canonicalisation, SMILES-index resolution,
and de-duplication by patient id.

The de-dup fixes a data-leakage bug: duplicate patient rows in the source CSV
were landing in different cross-validation folds, so a patient could sit in both
the training and the held-out set. Dedup by pid before the fold split removes it.

Keep-rule (see `dedupe_pid_mask`), validated against the real 5 duplicate pids:
- identical rows (475) or one row missing fields (119/883): keep the fuller row.
- conflicting features but agreeing outcome (N009): keep the first, log it.
- conflicting outcome label (954): drop the pid (ambiguous target).
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("cohort")

# Raw ASM code -> canonical drug name (must match entries in asm_drug_names.txt).
ASM_NAME_MAPPING: dict[str, str] = {
    "LEV": "Levetiracetam", "VPA": "Valproic_acid", "LTG": "Lamotrigine",
    "CBZ": "Carbamazepine", "cBZ": "Carbamazepine", "PTN": "Phenytoin",
    "TPM": "Topiramate", "OXC": "Oxcarbazepine", "LCM": "Lacosamide",
    "BRV": "Brivaracetam", "PER": "Perampanel", "ZNS": "Zonisamide",
    "GBP": "Gabapentin", "PGB": "Pregabalin", "CLB": "Clobazam",
    "CZP": "Clonazepam",
}

# Raw outcome 1 (success) -> 1, 2 (failure) -> 0.
# Source: the data dictionary shipped with the cohort
# (List_Missing_clinical_factors_07Nov2025.xlsx, sheet ASM_regimen, field
# outcome_12m: "1 = success, 2 = failure"). Success = seizure-free for the first
# 12 months while still on the regimen; failure = not seizure-free, or another
# ASM switched to or added within 12 months. Until 2026-09-28 this map was
# inverted ({1: 0, 2: 1}); see docs/analysis_plan_clean_rerun_exp18.md, B.1.
# This is the ONLY place the raw coding may be mapped
# (shared/tests/test_outcome_polarity.py enforces that).
OUTCOME_MAPPING: dict[int, int] = {1: 1, 2: 0}

# Columns whose non-blankness ranks duplicate rows (fuller row wins).
KEEP_RULE_COLS: tuple[str, ...] = (
    "outcome", "ASM", "age_init", "sex", "eeg_report", "mri_report",
)


def canonical_asm(name, mapping: dict[str, str] = ASM_NAME_MAPPING) -> str:
    """Map a raw ASM code (e.g. 'cBZ') to its canonical drug name."""
    s = str(name).strip()
    return mapping.get(s, s)


def filter_and_map_outcome(
    df: pd.DataFrame, *, col: str = "outcome", mapping: dict[int, int] = OUTCOME_MAPPING,
) -> pd.DataFrame:
    """Keep rows whose raw outcome is a mapping key, then map to {0, 1}.

    Must be called exactly once on RAW outcomes: raw value 1 is a valid key that
    maps to 0, so applying this to already-mapped data corrupts labels (this is
    the bug fixed in commit 237083e).
    """
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[out[col].isin(list(mapping))].copy()
    out[col] = out[col].map(mapping).astype(int)
    return out.reset_index(drop=True)


def valid_smiles_mask(df: pd.DataFrame, smiles_indices: dict, *, asm_col: str = "ASM") -> np.ndarray:
    """Boolean mask of rows whose canonical ASM is present in the SMILES index."""
    return df[asm_col].map(lambda a: canonical_asm(a) in smiles_indices).to_numpy()


def smiles_vector(asm, embeddings: np.ndarray, smiles_indices: dict) -> np.ndarray:
    """Return the SMILES embedding row for an ASM; unknown -> the dataset mean.

    The single, uniform SMILES resolver for every experiment. Replaces the
    silent ``.get(asm, 0)`` fallback (which mislabelled unknown drugs as the
    first indexed drug) with a neutral dataset-mean vector.
    """
    idx = smiles_indices.get(canonical_asm(asm))
    return embeddings[idx] if idx is not None else embeddings.mean(axis=0)


def _nonblank(v) -> bool:
    return not (pd.isna(v) or str(v).strip() == "")


def dedupe_pid_mask(
    df: pd.DataFrame, *, pid_col: str = "pid", cols: Sequence[str] = KEEP_RULE_COLS,
    outcome_col: str = "outcome", on_outcome_conflict: str = "drop",
) -> np.ndarray:
    """Row-aligned boolean mask selecting one row per pid.

    Returned as a mask (not a filtered frame) so the SAME selection can be applied
    to any array row-aligned with ``df`` (e.g. a precomputed embedding matrix).

    ``on_outcome_conflict``: 'drop' removes a pid whose rows disagree on outcome;
    'first' keeps the first row (logged). Feature-only conflicts always keep the
    fuller row (tie -> first) and are logged. Asserts surviving pids are unique.

    Call AFTER outcome filtering (``filter_and_map_outcome`` or an ``isin`` on the
    raw codes). On unfiltered outcomes an invalid code (e.g. raw ``0``) alongside
    a valid one reads as two distinct labels and the pid is wrongly dropped as a
    conflict; every pipeline filters first, so this is a precondition, not a bug.
    """
    if on_outcome_conflict not in ("drop", "first"):
        raise ValueError(f"on_outcome_conflict must be 'drop' or 'first', got {on_outcome_conflict!r}")
    d = df.reset_index(drop=True)
    pid = d[pid_col].astype(str)
    present = [c for c in cols if c in d.columns]
    fill = (
        d[present].apply(lambda s: s.map(_nonblank)).sum(axis=1)
        if present else pd.Series(0, index=d.index)
    )
    keep = np.zeros(len(d), dtype=bool)
    for p, idx in pid.groupby(pid).groups.items():
        pos = list(idx)
        if len(pos) == 1:
            keep[pos[0]] = True
            continue
        if outcome_col in d.columns and d.loc[pos, outcome_col].nunique() > 1:
            outs = sorted(set(d.loc[pos, outcome_col].tolist()))
            logger.warning("pid %s: conflicting outcome %s -> %s", p, outs,
                           "drop" if on_outcome_conflict == "drop" else "keep-first")
            if on_outcome_conflict == "drop":
                continue
            keep[pos[0]] = True  # 'first' keeps the first row, as documented
            continue
        best = max(pos, key=lambda i: (fill.iat[i], -i))
        differing = [c for c in d.columns if c != pid_col and d.loc[pos, c].astype(str).nunique() > 1]
        if differing:
            logger.warning("pid %s: duplicate rows differ on %s; kept row %d", p, differing, best)
        keep[best] = True
    assert pid[keep].is_unique, "dedupe left duplicate pids"
    return keep


def dedupe_by_pid(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Return ``df`` with one row per pid (see `dedupe_pid_mask`)."""
    return df[dedupe_pid_mask(df, **kwargs)].reset_index(drop=True)


def assert_oof_no_leakage(fold_pids: Sequence[Sequence]) -> None:
    """Assert no pid is duplicated within a fold or shared across folds."""
    seen: dict[str, int] = {}
    for fi, pids in enumerate(fold_pids):
        s = [str(p) for p in pids]
        assert len(s) == len(set(s)), f"fold {fi} contains duplicate pids"
        for p in set(s):
            if p in seen:
                raise AssertionError(f"pid {p} leaks across folds {seen[p]} and {fi}")
            seen[p] = fi
