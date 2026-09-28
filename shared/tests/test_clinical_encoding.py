"""Clinical encoding (analysis plan B.3/B.4): 3-level one-hot categories and
the cross-cohort drop set."""
import numpy as np
import pandas as pd
import pytest

from exp4_baseline.config import CLINICAL_CONFIG, CLINICAL_DIM
from exp4_baseline.data_pipeline import ClinicalFeaturePreprocessor
from shared.hep_cohort import CROSS_COHORT_DROP

BINARY = CLINICAL_CONFIG["binary_features"]


def _frame(n=6):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.integers(0, 2, n).astype(float) for c in BINARY})
    df["age_init"] = [15, 20, 30, 50, np.nan, 40]
    df["lesion"] = [1, 2, 3, np.nan, 2, 2]
    df["eeg_cat"] = [3, 3, 1, 2, 9, np.nan]  # 9 is out of range -> treated as missing
    return df


def test_width_and_constant_agree():
    assert ClinicalFeaturePreprocessor.n_features() == CLINICAL_DIM == 23
    assert ClinicalFeaturePreprocessor.n_features(CROSS_COHORT_DROP) == 20


def test_one_hot_levels_and_imputation():
    df = _frame()
    x = ClinicalFeaturePreprocessor().fit(df).transform(df)
    assert x.shape == (6, 23) and not np.isnan(x).any()
    lesion, eeg = x[:, 17:20], x[:, 20:23]
    assert (lesion.sum(1) == 1).all() and (eeg.sum(1) == 1).all()
    assert lesion[0].tolist() == [1, 0, 0] and lesion[2].tolist() == [0, 0, 1]
    assert lesion[3].tolist() == [0, 1, 0]          # NaN -> training mode (2)
    assert eeg[0].tolist() == [0, 0, 1]
    assert eeg[4].tolist() == eeg[5].tolist() == [0, 0, 1]  # 9 and NaN -> mode (3)
    ages = x[:, 13:17]
    assert (ages.sum(1) == 1).all()


def test_drop_removes_exactly_named_columns():
    df = _frame()
    full = ClinicalFeaturePreprocessor().fit(df).transform(df)
    red = ClinicalFeaturePreprocessor(drop=CROSS_COHORT_DROP).fit(df).transform(df)
    keep = [i for i, c in enumerate(BINARY) if c not in CROSS_COHORT_DROP]
    assert red.shape[1] == 20
    np.testing.assert_array_equal(red[:, :len(keep)], full[:, keep])
    np.testing.assert_array_equal(red[:, len(keep):], full[:, 13:])


def test_drop_rejects_non_binary():
    with pytest.raises(ValueError):
        ClinicalFeaturePreprocessor(drop=("age_init",))


def test_both_cohorts_round_trip():
    pytest.importorskip("shared.hep_cohort")
    from shared.hep_cohort import load_alfred, load_hep
    try:
        mel, hep = load_alfred(), load_hep()
    except FileNotFoundError:
        pytest.skip("cohort data not available")
    pre = ClinicalFeaturePreprocessor(drop=CROSS_COHORT_DROP).fit(mel)
    for df in (mel, hep):
        x = pre.transform(df)
        assert x.shape[1] == 20 and not np.isnan(x).any()
