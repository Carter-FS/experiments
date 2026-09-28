"""Refit protocol helpers (analysis plan Addendum B.2)."""
import numpy as np
import pytest

from shared import cv_splits
from shared.epoch_selection import (
    inner_folds, median_schedule, refit_protocol, select_epoch, smooth,
)


def test_inner_folds_partition_train_and_skip_test():
    y = np.array([0, 1] * 30)
    train = np.arange(0, 50)
    folds = inner_folds(y, train, fold=2)
    assert len(folds) == 5
    vals = np.concatenate([v for _, v in folds])
    assert sorted(vals.tolist()) == train.tolist()
    for tr, va in folds:
        assert not set(tr) & set(va)
        assert set(tr) | set(va) == set(train)
        assert not set(va) & set(range(50, 60))  # outer test rows never used


def test_inner_folds_follow_repeat_seed():
    y = np.array([0, 1] * 20)
    a = inner_folds(y, np.arange(40), fold=0)
    cv_splits.set_repeat_seed(43)
    try:
        b = inner_folds(y, np.arange(40), fold=0)
    finally:
        cv_splits.set_repeat_seed(None)
    assert any(not np.array_equal(x[1], z[1]) for x, z in zip(a, b))


def test_smooth_centred_with_short_edges():
    np.testing.assert_allclose(smooth([0, 3, 0, 3]), [1.5, 1.0, 2.0, 1.5])


def _blocks(y, good_epochs, n_epochs, stop=None):
    """Two inner folds whose probabilities are perfect only at good_epochs."""
    v1, v2 = np.arange(0, 10), np.arange(10, 20)
    blocks = []
    for v in (v1, v2):
        probs = []
        for e in range(1, (stop or n_epochs) + 1):
            probs.append(y[v].astype(float) if e in good_epochs else 1.0 - y[v])
        blocks.append((v, probs))
    return blocks


def test_select_epoch_picks_smoothed_peak_and_threshold():
    y = np.array([0, 1] * 10)
    choice = select_epoch(y, _blocks(y, {4, 5, 6}, 8))
    assert choice.epoch == 5  # centre of the plateau after smoothing
    assert choice.curve[4] == 1.0
    assert 0.0 < choice.threshold <= 1.0


def test_select_epoch_ties_take_earliest():
    y = np.array([0, 1] * 10)
    blocks = [(np.arange(0, 10), [y[:10].astype(float)] * 4),
              (np.arange(10, 20), [y[10:].astype(float)] * 4)]
    assert select_epoch(y, blocks).epoch == 1


def test_carry_forward_after_early_stop():
    y = np.array([0, 1] * 10)
    v1, v2 = np.arange(0, 10), np.arange(10, 20)
    good1, bad = y[v1].astype(float), 1.0 - y[v2]
    # run 1 stops after epoch 2 (perfect); run 2 is bad until epoch 5
    blocks = [(v1, [good1, good1]),
              (v2, [bad, bad, bad, bad, y[v2].astype(float), y[v2].astype(float)])]
    choice = select_epoch(y, blocks)
    assert choice.inner_epochs == [2, 6]
    assert choice.curve[5] == 1.0  # run 1 carried forward to epoch 6
    assert choice.epoch in (5, 6)


def test_cohort_stratified_criterion_ignores_base_rate():
    # Cohort A all label 1 except one; cohort B all 0 except one. A score that
    # only separates cohorts has high pooled AUC but 0.5 within cohorts.
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    coh = np.array(["A"] * 4 + ["B"] * 4)
    cohort_score = np.where(coh == "A", 0.9, 0.1)
    blocks = [(np.arange(8), [cohort_score])]
    plain = select_epoch(y, blocks)
    strat = select_epoch(y, blocks, cohorts=coh)
    assert plain.curve[0] >= 0.75
    assert strat.curve[0] == pytest.approx(0.5)


def test_select_epoch_rejects_misaligned_or_overlapping():
    y = np.array([0, 1] * 5)
    with pytest.raises(ValueError):
        select_epoch(y, [(np.arange(5), [np.zeros(4)])])
    with pytest.raises(ValueError):
        select_epoch(y, [(np.arange(5), [np.zeros(5)]), (np.arange(4, 9), [np.zeros(5)])])


def test_median_schedule_carries_forward():
    assert median_schedule([[1.0, 0.5], [1.0, 1.0, 0.25], [1.0]], 3) == [1.0, 1.0, 0.5]
    assert median_schedule([[1.0, 0.5], [1.0, 0.5, 0.25], [1.0, 0.5]], 3) == [1.0, 0.5, 0.5]


def test_refit_protocol_never_touches_outer_test():
    y = np.array([0, 1] * 30)
    train, test = np.arange(50), np.arange(50, 60)
    seen = set()

    def run_inner(fit, val):
        seen.update(fit.tolist()); seen.update(val.tolist())
        return {"val_probs": [y[val].astype(float)] * 3, "lr": [1e-3, 5e-4, 5e-4],
                "steps_per_epoch": 4}

    def run_refit(tr, choice):
        seen.update(tr.tolist())
        return choice.epoch

    choice, result = refit_protocol(y, y, train, 0, run_inner, run_refit)
    assert not seen & set(test.tolist())
    assert result == choice.epoch == 1
    assert choice.lr_schedule == [1e-3] and choice.steps_per_epoch == 4


def test_cv_suffix_refit_and_apply_args():
    import argparse
    ap = argparse.ArgumentParser()
    cv_splits.add_cv_args(ap)
    args = ap.parse_args(["--splitter", "multilabel", "--refit-folds", "5", "--cv-seed", "44"])
    cv_splits.apply_cv_args(args)
    try:
        assert cv_splits.cv_suffix("multilabel", 0.0) == "_sp-multilabel_rf5_s44"
        assert cv_splits.protocol_name(0.0) == "refit"
    finally:
        cv_splits.set_repeat_seed(None)
        cv_splits.set_refit_folds(0)
    bad = ap.parse_args(["--refit-folds", "5", "--inner-val", "0.2"])
    with pytest.raises(SystemExit):
        cv_splits.apply_cv_args(bad)
    assert cv_splits.cv_suffix("multilabel", 0.2, 42) == "_sp-multilabel_iv20_s42"
