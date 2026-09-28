"""Tests for shared.cohort - dedup keep-rule, leakage guard, SMILES fallback."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from shared.cohort import (  # noqa: E402
    assert_oof_no_leakage, canonical_asm, dedupe_by_pid, dedupe_pid_mask,
    filter_and_map_outcome, smiles_vector,
)

COLS = ["pid", "outcome", "ASM", "age_init", "sex", "eeg_report", "mri_report"]


def _row(pid, outcome, asm="LEV", age="40", sex="1", eeg="report text", mri="mri text"):
    return dict(pid=pid, outcome=outcome, ASM=asm, age_init=age, sex=sex,
                eeg_report=eeg, mri_report=mri)


def _frame(rows):
    return pd.DataFrame(rows, columns=COLS)


def test_outcome_filter_and_map():
    df = _frame([_row("a", "1"), _row("b", "2"), _row("c", " "), _row("d", "0")])
    out = filter_and_map_outcome(df)
    assert out["outcome"].tolist() == [1, 0]  # 1 (success)->1, 2 (failure)->0; blank and 0 dropped
    assert out["pid"].tolist() == ["a", "b"]


def test_dedupe_identical_and_fuller_row():
    df = _frame([
        _row("solo", 0),
        _row("475", 1), _row("475", 1),                       # identical -> one kept
        _row("119", 1, mri="full"), _row("119", 1, mri=""),   # keep the mri-full row
    ])
    out = dedupe_by_pid(df)
    assert sorted(out["pid"]) == ["119", "475", "solo"]
    assert out.loc[out.pid == "119", "mri_report"].item() == "full"


def test_dedupe_feature_conflict_keeps_first_outcome_agrees():
    df = _frame([_row("N009", 0, eeg="a"), _row("N009", 0, eeg="b")])  # focal-like conflict, outcome agrees
    # make the two rows equally full but differ on a feature
    df.loc[0, "sex"] = "0"; df.loc[1, "sex"] = "1"
    out = dedupe_by_pid(df)
    assert out["pid"].tolist() == ["N009"]
    assert out["sex"].item() == "0"  # first kept


def test_dedupe_outcome_conflict_drops_pid():
    df = _frame([_row("954", 1), _row("954", 0), _row("keep", 1)])
    out = dedupe_by_pid(df)  # default on_outcome_conflict='drop'
    assert out["pid"].tolist() == ["keep"]
    # 'first' keeps the FIRST row's label, even when a later row is fuller
    # (row 0 here is blanker but must still win, per the documented contract).
    df2 = _frame([_row("954", 1, mri=""), _row("954", 0), _row("keep", 1)])
    kept = dedupe_by_pid(df2, on_outcome_conflict="first")
    assert set(kept["pid"]) == {"954", "keep"}
    assert kept.loc[kept.pid == "954", "outcome"].item() == 1  # first row, not the fuller outcome-0 row


def test_dedupe_mask_aligns_to_external_array():
    df = _frame([_row("x", 1), _row("y", 1), _row("y", 1)])
    emb = np.arange(len(df))
    mask = dedupe_pid_mask(df)
    assert emb[mask].tolist() == [0, 1]  # y's second row dropped, array stays aligned


def test_no_leakage_after_dedupe():
    # A pid duplicated into rows that a 5-fold split would separate -> would leak.
    rows = [_row(f"p{i}", i % 2) for i in range(40)]
    rows += [_row("dup", 1), _row("dup", 1)]  # duplicate patient
    df = dedupe_by_pid(_frame(rows))
    y = df["outcome"].to_numpy()
    folds = [list(df.iloc[te]["pid"]) for _, te in
             StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)]
    assert_oof_no_leakage(folds)  # passes: 'dup' appears once
    assert (df["pid"] == "dup").sum() == 1


def test_leakage_guard_detects_shared_pid():
    with pytest.raises(AssertionError):
        assert_oof_no_leakage([["a", "b"], ["b", "c"]])


def test_smiles_vector_resolves_and_falls_back_to_mean():
    emb = np.array([[1.0, 1.0], [3.0, 3.0]])
    idx = {"Levetiracetam": 0, "Carbamazepine": 1}
    assert np.allclose(smiles_vector("LEV", emb, idx), [1.0, 1.0])
    assert np.allclose(smiles_vector("cBZ", emb, idx), [3.0, 3.0])      # canonicalised
    assert np.allclose(smiles_vector("UNKNOWN", emb, idx), [2.0, 2.0])  # mean, not index 0


def test_canonical_asm():
    assert canonical_asm("cBZ") == "Carbamazepine"
    assert canonical_asm(" LEV ") == "Levetiracetam"
    assert canonical_asm("XYZ") == "XYZ"


# --- Validation against the real cohort (skipped if the CSV is absent) --------

def _find_csv():
    for p in ("/Users/carter/carter_massive/asm_data/alfred_1st_regimen.csv",
              pathlib.Path(__file__).resolve().parents[2] / "asm_data" / "alfred_1st_regimen.csv"):
        if pathlib.Path(p).exists():
            return str(p)
    return None


@pytest.mark.skipif(_find_csv() is None, reason="alfred CSV not present")
def test_real_cohort_counts_and_no_leakage():
    df = pd.read_csv(_find_csv(), dtype=str)
    out = dedupe_by_pid(filter_and_map_outcome(df))
    assert len(out) == 198, f"expected clinical 198 after dropping 954, got {len(out)}"
    assert "954" not in set(out["pid"])                 # outcome-conflict dropped
    for pid in ("475", "119", "883", "N009"):
        assert (out["pid"] == pid).sum() == 1           # each kept once
    y = out["outcome"].to_numpy()
    folds = [list(out.iloc[te]["pid"]) for _, te in
             StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)]
    assert_oof_no_leakage(folds)
