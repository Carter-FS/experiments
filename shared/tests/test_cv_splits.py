"""Tests for the shared outer/inner CV splitters (shared/cv_splits.py)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from shared.cv_splits import (
    current_seed, cv_suffix, fold_indices, inner_val_split, joint_key, outer_splits, set_repeat_seed,
)


def _cohort(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "outcome": rng.integers(0, 2, n),
        "focal": rng.integers(0, 2, n),
        "sex": rng.integers(0, 2, n),
        "cohort": np.where(rng.random(n) < 0.3, "MEL", "HEP"),
    })


def _assert_partition(splits, n):
    tests = [set(te) for _, te in splits]
    assert sum(len(t) for t in tests) == n
    assert set().union(*tests) == set(range(n))
    for tr, te in splits:
        assert not set(tr) & set(te)
        assert len(tr) + len(te) == n


def test_legacy_matches_original_construction():
    df = _cohort()
    want = list(StratifiedKFold(5, shuffle=True, random_state=42).split(
        np.zeros(len(df)), df["outcome"].values))
    got = outer_splits(df, mode="legacy")
    for (wt, we), (gt, ge) in zip(want, got):
        assert np.array_equal(wt, gt) and np.array_equal(we, ge)


@pytest.mark.parametrize("mode", ["legacy", "multilabel", "joint"])
def test_outer_splits_partition_the_cohort(mode):
    df = _cohort()
    splits = outer_splits(df, mode=mode)
    assert len(splits) == 5
    _assert_partition(splits, len(df))


def test_joint_splitter_balances_each_outcome_cohort_cell():
    df = _cohort(n=500)
    key = joint_key(df, ["outcome", "cohort"])
    cells, totals = np.unique(key, return_counts=True)
    for _, te in outer_splits(df, mode="joint"):
        for cell, total in zip(cells, totals):
            # StratifiedKFold keeps every cell within one patient of total/5.
            assert abs((key[te] == cell).sum() - total / 5) <= 1


def test_inner_val_split_is_a_disjoint_stratified_subset():
    df = _cohort(n=300)
    y = df["outcome"].to_numpy()
    for fold, (tr, te) in enumerate(outer_splits(df, mode="multilabel")):
        fit, es = inner_val_split(y, tr, frac=0.2, seed=42 + fold)
        assert not set(fit) & set(es)
        assert set(fit) | set(es) == set(tr)
        assert not (set(fit) | set(es)) & set(te)
        assert abs(len(es) - 0.2 * len(tr)) <= 1
        assert abs(y[es].mean() - y[tr].mean()) < 0.05


def test_inner_val_split_rejects_degenerate_fraction():
    with pytest.raises(ValueError):
        inner_val_split(np.array([0, 1] * 10), np.arange(20), frac=0.0)


def test_cv_suffix():
    assert cv_suffix("legacy", 0.0) == ""
    assert cv_suffix("multilabel", 0.2) == "_sp-multilabel_iv20"
    assert cv_suffix("legacy", 0.2) == "_sp-legacy_iv20"
    assert cv_suffix("multilabel", 0.0) == "_sp-multilabel_iv0"
    assert cv_suffix("multilabel", 0.2, 43) == "_sp-multilabel_iv20_s43"
    assert cv_suffix("legacy", 0.0, 42) == "_s42"


def test_repeat_seed_overrides_split_seeds_and_suffix():
    df = _cohort()
    try:
        set_repeat_seed(43)
        want = list(StratifiedKFold(5, shuffle=True, random_state=43).split(
            np.zeros(len(df)), df["outcome"].values))
        got = outer_splits(df, mode="legacy", seed=42)  # explicit 42 is overridden
        assert all(np.array_equal(w[1], g[1]) for w, g in zip(want, got))
        assert cv_suffix("multilabel", 0.2) == "_sp-multilabel_iv20_s43"
        assert current_seed() == 43
        y = df["outcome"].to_numpy()
        tr, te = got[0]
        fit43, _, _ = fold_indices(y, tr, te, 0, 0.2)
        set_repeat_seed(None)
        fit42, _, _ = fold_indices(y, tr, te, 0, 0.2)
        assert not np.array_equal(fit42, fit43)
    finally:
        set_repeat_seed(None)
    assert current_seed() == 42 and cv_suffix("legacy", 0) == ""


def test_multilabel_refuses_missing_columns():
    df = _cohort().drop(columns=["sex"])
    with pytest.raises(ValueError, match="sex"):
        outer_splits(df, mode="multilabel")


def test_smoke_mode_limits_folds_epochs_and_tags_outputs():
    import argparse
    from shared import cv_splits as cv
    from shared.prediction_logger import protocol_metadata
    df = _cohort()
    parser = argparse.ArgumentParser()
    cv.add_cv_args(parser)
    args = parser.parse_args(["--splitter", "multilabel", "--refit-folds", "5", "--cv-seed", "42", "--smoke"])
    try:
        cv.apply_cv_args(args)
        assert cv.smoke() and cv.refit_folds() == cv.SMOKE_INNER_FOLDS == 2
        assert cv.max_epochs(100) == cv.SMOKE_EPOCHS == 2 and cv.max_epochs(1) == 1
        full = cv._outer_splits(df, "legacy", 5, 42, None)
        got = cv.outer_splits(df, mode="legacy", seed=42)
        assert len(got) == 1 and np.array_equal(got[0][1], full[0][1])   # the first fold, unchanged
        assert cv.cv_suffix("multilabel", 0.0) == "_sp-multilabel_rf5_s42_smoke"
        assert protocol_metadata(0.0) == {"protocol": "refit", "refit_folds": 2, "smoke": True}
    finally:
        cv.apply_cv_args(parser.parse_args([]))
    assert not cv.smoke() and cv.refit_folds() == 0 and cv.max_epochs(100) == 100
    assert len(cv.outer_splits(df, mode="legacy", seed=42)) == 5 and cv.cv_suffix("legacy", 0.0) == ""
    assert protocol_metadata(0.2) == {"protocol": "innersplit", "refit_folds": 0, "smoke": False}


# Every module with a training loop that rerun_clean.sh runs (the HEP scripts train
# through shared.portable_models) must take its epoch budget through max_epochs.
SMOKE_RUNNERS = [
    "exp1_fusion.training", "exp2_fusion.training", "exp3_fusion.training", "exp4_baseline.training",
    "exp5_clinical_fusion.training", "exp6_clinical_triple.training", "exp7_all_modalities.training",
    "exp9_eeg_investigation.run_experiments", "exp11_eeg_upgrade.run_experiments",
    "exp15_reve_quad_mlp.training", "exp16_reduced_capacity.training", "shared.portable_models",
    "analysis.reve_standalone", "analysis.reve_ablation", "analysis.reve_followups",
]


def _import_runner(module):
    import importlib
    import sys
    from pathlib import Path
    if module.startswith("analysis."):
        root = str(Path(__file__).resolve().parents[2] / "thesisStandalone")
        if root not in sys.path:
            sys.path.insert(0, root)
    return importlib.import_module(module)


@pytest.mark.parametrize("module", SMOKE_RUNNERS)
def test_every_epoch_loop_takes_its_budget_through_max_epochs(module):
    import inspect
    import re
    mod = _import_runner(module)
    assert callable(getattr(mod, "max_epochs", None)), f"{module} does not import max_epochs at module level"
    src = inspect.getsource(mod)
    loops = [b for b in re.findall(r"for \w+ in range\(([^\n]*)\):", src) if re.search(r"epoch", b, re.I)]
    assert loops, f"{module} has no epoch loop"
    for bound in loops:
        assert "max_epochs(" in bound, f"{module}: epoch loop bound {bound!r} bypasses max_epochs"
    assert not re.search(r"^\s*while .*epoch", src, re.I | re.M), f"{module}: a while-loop over epochs is not budgeted"


SUMMARY_WRITERS = ["exp3_fusion.run_experiments", "exp4_baseline.run_experiments", "exp5_clinical_fusion.run_experiments",
                   "exp6_clinical_triple.run_experiments", "exp7_all_modalities.run_experiments",
                   "exp9_eeg_investigation.run_experiments", "exp11_eeg_upgrade.run_experiments"]


def test_smoke_tag_marks_files_only_in_smoke_mode():
    from pathlib import Path
    from shared import cv_splits as cv
    assert cv.smoke_tag(Path("outputs/x/results_1.json")) == Path("outputs/x/results_1.json")
    try:
        cv.set_smoke(True)
        assert cv.smoke_tag("outputs/x/results_1.json") == Path("outputs/x/results_1_smoke.json")
    finally:
        cv.set_smoke(False)


@pytest.mark.parametrize("module", SUMMARY_WRITERS)
def test_every_timestamped_summary_path_is_smoke_tagged(module):
    import importlib
    import inspect
    import re
    mod = importlib.import_module(module)
    assert callable(getattr(mod, "smoke_tag", None)), f"{module} does not import smoke_tag at module level"
    src = inspect.getsource(mod)
    for line in src.splitlines():
        if "{timestamp}" in line and ".json" in line:
            assert "smoke_tag(" in line, f"{module}: untagged summary path: {line.strip()}"
