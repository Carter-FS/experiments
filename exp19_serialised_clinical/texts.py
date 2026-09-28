"""Cohort frames, fold fills and the text for every exp19 variant (Addendum A / A.1).

Shared by embed.py (which embeds every distinct text once) and
run_experiments.py (which rebuilds each fold's texts and looks their
embeddings up), so the two can never disagree about a text.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.cv_splits import fold_indices, outer_splits
from shared.epoch_selection import inner_folds
from shared.hep_cohort import (
    CROSS_COHORT_DROP, load_alfred, load_alfred_text_aligned, load_hep, load_hep_text_embeddings,
)
from shared.serialise_clinical import clean_report, fit_fills, impute, serialise_frame, with_report

from .config import INNER_FRAC, SEEDS

SPLITTER = "multilabel"
# v1xc: v1 without the facts constant in HEP1, for models that see both
# cohorts (exp18, plan B.4).
FOLD_INDEPENDENT = ("v1", "v1nodrug", "v2", "rep", "v1xc")
FOLD_IMPUTED = ("v1nodrugimp",)


def load_frames() -> dict:
    """{(cohort, scope): frame}; scope "text" = the patients with a usable EEG report (the
    paper's text cohorts: 117 Melbourne via the exp5 text pipeline, 207 HEP1)."""
    mel, hep = load_alfred(), load_hep()
    mel_text = {str(p) for p in load_alfred_text_aligned()[0]["pid"]}
    hep_text = set(load_hep_text_embeddings()[0])
    return {
        ("MEL", "full"): mel.reset_index(drop=True),
        ("HEP", "full"): hep.reset_index(drop=True),
        ("MEL", "text"): mel[mel["pid"].astype(str).isin(mel_text)].reset_index(drop=True),
        ("HEP", "text"): hep[hep["pid"].astype(str).isin(hep_text)].reset_index(drop=True),
    }


def fold_plan(frame: pd.DataFrame, seed: int):
    """[(fold, test_idx, train_idx, fit_idx, es_idx)] for one seed, exactly as the
    driver splits; fit/es is the inner split (unused under the refit protocol)."""
    y = frame["outcome"].to_numpy()
    plan = []
    for fold, (tr, te) in enumerate(outer_splits(frame, mode=SPLITTER, seed=seed)):
        fit, es, test = fold_indices(y, tr, te, fold, INNER_FRAC, seed=seed)
        plan.append((fold, test, tr, fit, es))
    return plan


def training_sets(frame: pd.DataFrame, seed: int, n_inner: int = 5):
    """Every row set whose fills a run may fit, for both protocols: the inner
    split's fit rows, each inner fold's training rows and the whole outer
    training fold."""
    y = frame["outcome"].to_numpy()
    for fold, _, tr, fit, _ in fold_plan(frame, seed):
        yield fit
        yield tr
        for inner_tr, _ in inner_folds(y, tr, fold, n_inner, seed=seed):
            yield inner_tr


def texts(frame: pd.DataFrame, variant: str, fills: dict | None = None) -> list[str]:
    """The ``variant`` text for every row of ``frame`` (``fills`` for imputed variants)."""
    if variant == "v1":
        return serialise_frame(frame)
    if variant == "v1nodrug":
        return serialise_frame(frame, include_asm=False)
    if variant == "v1xc":
        return serialise_frame(frame, omit=CROSS_COHORT_DROP)
    if variant == "v1nodrugimp":
        if fills is None:
            raise ValueError("imputed variants need the fold's fills")
        return serialise_frame(impute(frame, fills), include_asm=False)
    if variant == "v2":
        return [with_report(t, r) for t, r in zip(serialise_frame(frame), frame["eeg_report"])]
    if variant == "rep":
        return [clean_report(r) for r in frame["eeg_report"]]
    raise ValueError(f"unknown variant {variant!r}")


def all_texts(frames: dict, seeds=SEEDS) -> list[str]:
    """Every distinct text any configuration, fold or seed will ask for."""
    out: set[str] = set()
    for cohort in ("MEL", "HEP"):
        out.update(texts(frames[(cohort, "full")], "v1"))
        out.update(texts(frames[(cohort, "full")], "v1nodrug"))
        out.update(texts(frames[(cohort, "text")], "v2"))
        out.update(texts(frames[(cohort, "text")], "rep"))
        out.update(texts(frames[(cohort, "full")], "v1xc"))
    mel = frames[("MEL", "full")]
    for seed in seeds:
        for rows in training_sets(mel, seed):
            fills = fit_fills(mel.iloc[rows])
            for cohort in ("MEL", "HEP"):
                out.update(texts(frames[(cohort, "full")], "v1nodrugimp", fills))
    return sorted(out)


def seen_drug_mask(hep: pd.DataFrame, mel: pd.DataFrame) -> np.ndarray:
    return hep["ASM"].isin(set(mel["ASM"])).to_numpy()


def complete_case_mask(df: pd.DataFrame) -> np.ndarray:
    from shared.serialise_clinical import BINARY_COLS, CATEGORICAL_COLS
    return df[BINARY_COLS + CATEGORICAL_COLS + ["age_init"]].notna().all(axis=1).to_numpy()
