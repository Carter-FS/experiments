"""Information-matched tabular clinical features for exp19 (T4-full, T5a-full; Addendum A.1).

Carries what the serialised text carries: every binary feature (mode-filled)
plus a missing indicator; age as a continuous value (z-scored, mean-filled)
plus a missing indicator; CT/MRI and EEG findings one-hot over normal /
non-epileptiform / epileptiform / missing. Fitted on a fold's fit rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.serialise_clinical import BINARY_COLS, CATEGORICAL_COLS


class FullClinicalPreprocessor:
    """``drop`` removes binary features (exp18's cross-cohort set, plan B.4)."""

    def __init__(self, drop: tuple[str, ...] = ()):
        self.binary = [c for c in BINARY_COLS if c not in drop]

    def fit(self, df: pd.DataFrame) -> "FullClinicalPreprocessor":
        self.modes = {}
        for col in self.binary:
            mode = pd.to_numeric(df[col], errors="coerce").mode()
            self.modes[col] = float(mode.iloc[0]) if len(mode) else 0.0
        age = pd.to_numeric(df["age_init"], errors="coerce")
        self.age_mean = float(age.mean())
        self.age_sd = float(age.std(ddof=0)) or 1.0
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for col in self.binary:
            v = pd.to_numeric(df[col], errors="coerce")
            cols += [v.fillna(self.modes[col]).to_numpy(), v.isna().to_numpy()]
        age = pd.to_numeric(df["age_init"], errors="coerce")
        cols += [((age.fillna(self.age_mean) - self.age_mean) / self.age_sd).to_numpy(), age.isna().to_numpy()]
        for col in CATEGORICAL_COLS:
            v = pd.to_numeric(df[col], errors="coerce")
            cols += [(v == level).to_numpy() for level in (1, 2, 3)] + [v.isna().to_numpy()]
        return np.column_stack(cols).astype(np.float32)

    @property
    def n_features(self) -> int:
        return 2 * len(self.binary) + 2 + 4 * len(CATEGORICAL_COLS)
