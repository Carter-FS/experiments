"""Refit protocol: choose the epoch count by inner cross-validation, then refit.

Analysis plan Addendum B.2. For one outer fold (``train_idx`` = the whole outer
training fold):

1. ``inner_folds`` splits ``train_idx`` into five stratified inner folds.
2. The caller trains one model per inner fold with its usual loop (patience
   included), recording the inner-validation probabilities after every epoch.
3. ``select_epoch`` pools the inner out-of-fold probabilities epoch by epoch
   (a run that stopped early carries its last probabilities forward), scores
   each epoch by pooled AUC (cohort-stratified when ``cohorts`` is given),
   smooths the curve with a centred 3-epoch moving average and takes the
   earliest argmax as E*. The threshold is Youden J on the pooled inner
   probabilities at E*.
4. The caller refits a fresh model on all of ``train_idx`` for exactly E*
   epochs and scores the outer test fold once.

``refit_protocol`` wires the steps together around two caller closures, so
every training loop keeps its own datasets, model and epoch function.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from shared.cv_splits import DEFAULT_SEED, current_seed, youden_threshold

N_INNER = 5
SMOOTH_WINDOW = 3


@dataclass
class EpochChoice:
    epoch: int                          # E*, 1-based
    threshold: float                    # Youden J on pooled inner OOF at E*
    curve: list[float]                  # raw pooled criterion per epoch
    smoothed: list[float]
    inner_best: list[int]               # each inner run's own best epoch (1-based)
    inner_epochs: list[int]             # epochs each inner run trained
    lr_schedule: list[float] | None = None     # per-epoch median LR (exp1/exp2)
    steps_per_epoch: float | None = None       # median inner optimiser steps per epoch
    n_inner: int = N_INNER
    extra: dict = field(default_factory=dict)

    def as_metadata(self) -> dict:
        d = asdict(self)
        d["curve"] = [round(float(v), 5) for v in self.curve]
        d["smoothed"] = [round(float(v), 5) for v in self.smoothed]
        return d


def inner_folds(strat_labels, train_idx, fold: int, n_inner: int = N_INNER,
                seed: int = DEFAULT_SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified inner folds of ``train_idx`` as sorted (inner_train, inner_val)
    position arrays. Seed = active repeated-CV seed + fold, as for the inner
    split of the inner-split protocol."""
    train_idx = np.asarray(train_idx)
    strat = np.asarray(strat_labels)[train_idx]
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=current_seed(seed) + fold)
    return [(np.sort(train_idx[a]), np.sort(train_idx[b]))
            for a, b in skf.split(np.zeros(len(train_idx)), strat)]


def _criterion(y, p, cohorts) -> float:
    if cohorts is not None:
        from shared.stats_util import cohort_stratified_auc
        v = cohort_stratified_auc(y, p, cohorts)
        return 0.5 if not np.isfinite(v) else v
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, p))


def smooth(curve: Sequence[float], window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Centred moving average; the ends average over the neighbours that exist."""
    c = np.asarray(curve, dtype=float)
    half = window // 2
    return np.array([c[max(0, i - half): i + half + 1].mean() for i in range(len(c))])


def select_epoch(y, blocks: Sequence[tuple[np.ndarray, Sequence[np.ndarray]]],
                 cohorts=None, window: int = SMOOTH_WINDOW) -> EpochChoice:
    """E* and threshold from the inner runs.

    ``y`` (and ``cohorts``) are indexed by cohort position. ``blocks`` holds one
    (inner_val_idx, [probs after epoch 1, probs after epoch 2, ...]) per inner
    fold; the inner-validation sets must partition the outer training fold.
    """
    y = np.asarray(y)
    cohorts = None if cohorts is None else np.asarray(cohorts)
    blocks = [(np.asarray(v), [np.asarray(p, dtype=float) for p in probs]) for v, probs in blocks]
    if any(len(probs) == 0 for _, probs in blocks):
        raise ValueError("every inner run must record at least one epoch")
    for v, probs in blocks:
        if any(len(p) != len(v) for p in probs):
            raise ValueError("per-epoch probabilities must align with the inner-validation indices")
    idx = np.concatenate([v for v, _ in blocks])
    if len(np.unique(idx)) != len(idx):
        raise ValueError("inner-validation sets overlap")
    horizon = max(len(probs) for _, probs in blocks)

    def pooled(e: int) -> np.ndarray:  # e is 0-based; carry the last epoch forward
        return np.concatenate([probs[min(e, len(probs) - 1)] for _, probs in blocks])

    yy = y[idx]
    cc = None if cohorts is None else cohorts[idx]
    curve = np.array([_criterion(yy, pooled(e), cc) for e in range(horizon)])
    sm = smooth(curve, window)
    best = int(np.argmax(sm))  # first occurrence = earliest epoch on ties
    inner_best = []
    for v, probs in blocks:
        per = [_criterion(y[v], p, None if cohorts is None else cohorts[v]) for p in probs]
        inner_best.append(int(np.argmax(per)) + 1)
    return EpochChoice(
        epoch=best + 1,
        threshold=youden_threshold(yy, pooled(best)),
        curve=curve.tolist(), smoothed=sm.tolist(),
        inner_best=inner_best, inner_epochs=[len(p) for _, p in blocks],
    )


def median_schedule(schedules: Sequence[Sequence[float]], n_epochs: int) -> list[float]:
    """Per-epoch median of the inner runs' learning rates, each carried forward
    past its last epoch, for the first ``n_epochs`` epochs."""
    mat = np.array([[s[min(e, len(s) - 1)] for e in range(n_epochs)] for s in schedules], dtype=float)
    return np.median(mat, axis=0).tolist()


def refit_protocol(y, strat_labels, train_idx, fold: int,
                   run_inner: Callable[[np.ndarray, np.ndarray], dict],
                   run_refit: Callable[[np.ndarray, EpochChoice], object],
                   *, cohorts=None, n_inner: int = N_INNER, seed: int = DEFAULT_SEED):
    """Run the whole protocol for one outer fold; returns (EpochChoice, refit result).

    ``run_inner(inner_train_idx, inner_val_idx)`` trains one inner model and
    returns a trace dict with ``val_probs`` (list of per-epoch probability
    arrays aligned with ``inner_val_idx``) and optionally ``lr`` (per-epoch
    learning rate) and ``steps_per_epoch``. ``run_refit(train_idx, choice)``
    trains the fresh model for ``choice.epoch`` epochs and returns whatever the
    caller needs (typically the outer-test metrics).
    """
    blocks, traces = [], []
    for fit_idx, val_idx in inner_folds(strat_labels, train_idx, fold, n_inner, seed):
        trace = run_inner(fit_idx, val_idx)
        blocks.append((val_idx, trace["val_probs"]))
        traces.append(trace)
    choice = select_epoch(y, blocks, cohorts=cohorts)
    choice.n_inner = n_inner
    lrs = [t["lr"] for t in traces if t.get("lr")]
    if lrs:
        choice.lr_schedule = median_schedule(lrs, choice.epoch)
    spe = [t["steps_per_epoch"] for t in traces if t.get("steps_per_epoch")]
    if spe:
        choice.steps_per_epoch = float(np.median(spe))
    return choice, run_refit(np.asarray(train_idx), choice)


def run_outer_fold(strat_labels, train_idx, test_idx, fold: int, inner_val: float,
                   make_datasets: Callable, train: Callable, *, y=None, cohorts=None,
                   log: Callable[[str], None] | None = None):
    """One outer fold under whichever protocol is active; returns
    (outer-test metrics, selection metadata or None).

    ``make_datasets(a_idx, b_idx)`` returns (train dataset, eval dataset) with
    every preprocessor fitted on ``a_idx`` only. ``train(train_ds, val_ds,
    test_dataset=None, trace=None, fixed_epochs=None, refit_choice=None)`` is the
    experiment's train_fold. Legacy and inner-split runs follow the original
    code path exactly (same index sets, same call order); the refit protocol
    runs ``refit_protocol`` around the same two callables.
    """
    from shared.cv_splits import fold_indices, refit_folds, rethreshold

    y = np.asarray(strat_labels if y is None else y)
    if not refit_folds():
        fit_idx, es_idx, clean_test = fold_indices(strat_labels, train_idx, test_idx, fold, inner_val)
        train_ds, val_ds = make_datasets(fit_idx, es_idx)
        test_ds = make_datasets(fit_idx, clean_test)[1] if clean_test is not None else None
        if log:
            log(f"  Train: {len(train_ds)}, Early-stop: {len(val_ds)}"
                + (f", Test: {len(test_ds)}" if test_ds is not None else ""))
        return train(train_ds, val_ds, test_dataset=test_ds), None

    def run_inner(fit_idx, val_idx):
        train_ds, val_ds = make_datasets(fit_idx, val_idx)
        trace: dict = {}
        train(train_ds, val_ds, trace=trace)
        return trace

    def run_refit(full_idx, choice):
        train_ds, test_ds = make_datasets(full_idx, np.asarray(test_idx))
        if log:
            log(f"  Refit on {len(train_ds)} for {choice.epoch} epochs "
                f"(inner best {choice.inner_best}), test {len(test_ds)}")
        result = train(train_ds, None, test_dataset=test_ds, fixed_epochs=choice.epoch, refit_choice=choice)
        if "y_prob" not in result and "metrics" in result:
            # train_fold_with_predictions-style result: the metrics are nested.
            return {**result, "metrics": rethreshold(result["metrics"], choice.threshold)}
        return rethreshold(result, choice.threshold)

    choice, metrics = refit_protocol(y, strat_labels, train_idx, fold, run_inner, run_refit,
                                     cohorts=cohorts, n_inner=refit_folds())
    return metrics, choice.as_metadata()
