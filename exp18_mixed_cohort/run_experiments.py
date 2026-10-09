"""Run Experiment 18: mixed-cohort training with per-cohort evaluation.

For each configuration and seed, one outer 5-fold split of the pooled
Melbourne + HEP1 cohort (stratified on outcome x cohort). Every arm trains on
its share of each outer training fold (mixed / Melbourne only / HEP1 only),
selects its epoch count on inner data of that share (a 20% inner split, or
under --refit-folds five inner folds and a refit on the whole share, analysis
plan B.2; the mixed arm then selects on the cohort-stratified AUC, B.6), and
scores every outer test patient in both cohorts, so all arms are compared on
the same test patients. Exp4a also runs size-matched mixed training (10 draws
per test cohort). Clinical inputs omit the features constant in HEP1 (B.4).

    python -m exp18_mixed_cohort.run_experiments --config Exp4a            # all its seeds
    python -m exp18_mixed_cohort.run_experiments --config Exp4a Exp5a --seeds 42
    python -m exp18_mixed_cohort.run_experiments --config Exp4a --exclude-rmh   # sensitivity
    python -m exp18_mixed_cohort.run_experiments --config Exp4a --smoke    # 1 fold, 2 epochs
    python -m exp18_mixed_cohort.run_experiments --config Exp4a --refit-folds 5 [--hep-outcome harmonised12]

Outputs (outputs/exp18_mixed_cohort/, patient-level, gitignored):
    predictions_<cfg><variant>_seed<k>.csv   one row per (arm, draw, test patient);
                                             variant = [_rf5][_h12][_noRMH][_dedup]
    folds_<cfg><variant>_seed<k>.csv         per-fold train/test composition
    run_<cfg><variant>_seed<k>.json          arguments + provenance
Existing outputs are skipped unless --force, so slurm array tasks can resume.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import shared.portable_models as portable
from shared.cv_splits import (
    inner_val_split, outer_splits, refit_folds, set_refit_folds, set_repeat_seed, youden_threshold,
)
from shared.determinism import enable_determinism
from shared.epoch_selection import refit_protocol
from shared.hep_cohort import HEP_OUTCOMES, hep_outcome_tag
from shared.eeg_cache import LOADED_INPUTS
from shared.prediction_logger import run_provenance

from .config import (
    ARM_COHORTS,
    ARMS,
    CONFIGS,
    INNER_FRAC,
    LF_INPUTS,
    N_SPLITS,
    OUT_DIR,
    PORTABLE_MODEL,
    SEEDS,
    SIZEMATCH_CONFIGS,
    SIZEMATCH_DRAWS,
    STRATIFY,
)
from .data_pipeline import PooledCohort, clinical_features, load_pooled

logger = logging.getLogger("exp18")


def _fit(pooled: PooledCohort, fit_idx, es_idx, device, *, trace=None, fixed_epochs=None, val_cohorts=None):
    """Train on ``fit_idx`` (early-stopping on ``es_idx`` unless ``fixed_epochs``);
    returns (model, predict) where predict(idx) gives probabilities."""
    cfg, y = PORTABLE_MODEL[pooled.config], pooled.labels
    es_labels = None if es_idx is None else y[es_idx]
    kw = dict(trace=trace, fixed_epochs=fixed_epochs, val_cohorts=val_cohorts)
    if pooled.config in LF_INPUTS:
        return _fit_late_fusion(pooled, cfg, fit_idx, es_idx, device, **kw)
    mods = dict(pooled.modalities)
    mods["clinical"] = clinical_features(pooled, fit_idx)
    if cfg in portable.EEG_CONFIGS:
        sub = lambda idx: portable.index_modalities(mods, idx)  # noqa: E731
        model = portable.train_fold_eeg(cfg, sub(fit_idx), None if es_idx is None else sub(es_idx),
                                        y[fit_idx], es_labels, device, **kw)
        return model, lambda idx: portable.predict_eeg(model, sub(idx), cfg, device)
    tensors = [mods["clinical"]] + [mods[k] for k in ("smiles", "text") if k in mods]
    sub = lambda idx: [t[idx] for t in tensors]  # noqa: E731
    model = portable.train_fold(cfg, sub(fit_idx), None if es_idx is None else sub(es_idx),
                                y[fit_idx], es_labels, device, **kw)
    return model, lambda idx: portable.predict(model, sub(idx), cfg, device)


def _fit_late_fusion(pooled: PooledCohort, cfg: str, fit_idx, es_idx, device, **kw):
    """exp19 configurations (LateFusionMLP): embedding inputs z-scored and clinical
    preprocessors fitted on this arm's fit rows."""
    from exp19_serialised_clinical.tabular import FullClinicalPreprocessor
    from shared.hep_cohort import CROSS_COHORT_DROP
    y, tensors = pooled.labels, []
    for name in LF_INPUTS[pooled.config]:
        if name == "clinical":
            tensors.append(clinical_features(pooled, fit_idx))
        elif name == "clinical_full":
            pre = FullClinicalPreprocessor(drop=CROSS_COHORT_DROP).fit(pooled.df.iloc[fit_idx])
            tensors.append(torch.from_numpy(pre.transform(pooled.df)))
        else:
            x = pooled.modalities[name].float()
            mu, sd = x[fit_idx].mean(0), x[fit_idx].std(0, unbiased=False).clamp(min=1e-6)
            tensors.append((x - mu) / sd)
    dims = [t.shape[1] for t in tensors]
    sub = lambda idx: [t[idx] for t in tensors]  # noqa: E731
    model = portable.train_fold(cfg, sub(fit_idx), None if es_idx is None else sub(es_idx), y[fit_idx],
                                None if es_idx is None else y[es_idx], device,
                                model_factory=lambda: portable.LateFusionMLP(dims), **kw)
    return model, lambda idx: portable.predict(model, sub(idx), cfg, device)


def train_predict(pooled: PooledCohort, fit_idx, es_idx, test_idx, device) -> tuple[np.ndarray, float]:
    """Inner-split protocol: train on ``fit_idx``, early-stop on ``es_idx``;
    return (test probs, es threshold)."""
    _, predict = _fit(pooled, fit_idx, es_idx, device)
    return predict(test_idx), youden_threshold(pooled.labels[es_idx].numpy(), predict(es_idx))


def fit_arm(pooled: PooledCohort, arm: str, arm_tr, test_idx, fold: int, seed: int, device):
    """Train one arm on its share ``arm_tr`` of the outer training fold under the
    active protocol; returns (test probs, threshold, n_fit, n_es, selection)."""
    if not refit_folds():
        fit, es = inner_val_split(pooled.key, arm_tr, INNER_FRAC, seed + fold)
        probs, thr = train_predict(pooled, fit, es, test_idx, device)
        return probs, thr, len(fit), len(es), None
    y = pooled.labels.numpy()
    # A model that sees both cohorts selects on within-cohort ranking only (B.6).
    cohorts = pooled.df["cohort"].to_numpy() if arm == "mixed" else None

    def run_inner(fit, val):
        trace: dict = {}
        _fit(pooled, fit, val, device, trace=trace, val_cohorts=None if cohorts is None else cohorts[val])
        return trace

    def run_refit(full, choice):
        return _fit(pooled, full, None, device, fixed_epochs=choice.epoch)[1](test_idx)

    choice, probs = refit_protocol(y, pooled.key, arm_tr, fold, run_inner, run_refit,
                                   cohorts=cohorts, n_inner=refit_folds(), seed=seed)
    return probs, choice.threshold, len(arm_tr), 0, choice.as_metadata()


def sizematched_subsample(pooled: PooledCohort, train_idx, n: int, random_state: int) -> np.ndarray:
    """``n`` patients from ``train_idx`` keeping its outcome x cohort proportions."""
    if n >= len(train_idx):
        return np.sort(train_idx)
    sub, _ = train_test_split(train_idx, train_size=n, stratify=pooled.key[train_idx],
                              random_state=random_state)
    return np.sort(sub)


def run_seed(pooled: PooledCohort, seed: int, device, arms=ARMS, sizematch: bool = False,
             max_folds: int | None = None, draws: int = SIZEMATCH_DRAWS):
    # The shared EEG loop draws its batch order from the repeat seed; set it so
    # each exp18 seed shuffles differently (splits below pass ``seed`` explicitly).
    set_repeat_seed(seed)
    try:
        return _run_seed(pooled, seed, device, arms, sizematch, max_folds, draws)
    finally:
        set_repeat_seed(None)


def _run_seed(pooled: PooledCohort, seed: int, device, arms, sizematch: bool,
              max_folds: int | None, draws: int):
    df, y = pooled.df, pooled.labels.numpy()
    cohort = df["cohort"].to_numpy()
    splits = outer_splits(df.assign(outcome=y), mode="joint", n_splits=N_SPLITS, seed=seed,
                          key_cols=STRATIFY)
    seen_test: list[np.ndarray] = []
    rows, fold_rows = [], []

    def record(arm, draw, fold, test_idx, probs, thr, n_fit, n_es, selection=None):
        epoch = selection["epoch"] if selection else None
        for i, p in zip(test_idx, probs):
            rows.append({"config": pooled.config, "seed": seed, "fold": fold, "arm": arm, "draw": draw,
                         "pid": df["pid"].iat[i], "cohort": cohort[i], "y_true": int(y[i]),
                         "y_prob": float(p), "threshold": thr, "n_fit": n_fit, "n_es": n_es,
                         "selected_epoch": epoch})

    for fold, (tr, te) in enumerate(splits[:max_folds]):
        assert not set(tr) & set(te)
        seen_test.append(te)
        fold_rows.append({
            "config": pooled.config, "seed": seed, "fold": fold, "n_train": len(tr), "n_test": len(te),
            **{f"n_train_{c}": int((cohort[tr] == c).sum()) for c in ("MEL", "HEP")},
            **{f"n_test_{c}": int((cohort[te] == c).sum()) for c in ("MEL", "HEP")},
            **{f"train_rate_{c}": float(y[tr][cohort[tr] == c].mean()) for c in ("MEL", "HEP")},
        })
        for arm in arms:
            arm_tr = tr[np.isin(cohort[tr], ARM_COHORTS[arm])]
            enable_determinism(seed + fold)  # same init for every arm of a fold
            t0 = time.time()
            probs, thr, n_fit, n_es, selection = fit_arm(pooled, arm, arm_tr, te, fold, seed, device)
            record(arm, -1, fold, te, probs, thr, n_fit, n_es, selection)
            aucs = {c: roc_auc_score(y[te][cohort[te] == c], probs[cohort[te] == c])
                    for c in ("MEL", "HEP") if len(set(y[te][cohort[te] == c])) > 1}
            logger.info(f"{pooled.config} seed {seed} fold {fold} {arm:8s} fit {n_fit:3d}  "
                        + "  ".join(f"{c} AUC {a:.3f}" for c, a in aucs.items())
                        + f"  ({time.time() - t0:.0f}s)")
        if sizematch:
            for c in ("MEL", "HEP"):
                n_own = int((cohort[tr] == c).sum())
                test_c = te[cohort[te] == c]
                for draw in range(draws):
                    rs = seed * 10_000 + fold * 100 + draw * 2 + (c == "HEP")
                    sub = sizematched_subsample(pooled, tr, n_own, rs)
                    enable_determinism(seed + fold)
                    probs, thr, n_fit, n_es, selection = fit_arm(pooled, "mixed", sub, test_c, fold, seed, device)
                    record(f"sizematched_{c}", draw, fold, test_c, probs, thr, n_fit, n_es, selection)
    if max_folds is None:
        all_test = np.concatenate(seen_test)
        assert len(all_test) == len(df) == len(set(all_test)), "outer folds must partition the cohort"
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 18: mixed-cohort training")
    parser.add_argument("--config", nargs="+", choices=CONFIGS, required=True)
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="default: the frozen seeds for each configuration")
    parser.add_argument("--arms", nargs="*", choices=ARMS, default=list(ARMS))
    parser.add_argument("--no-sizematch", action="store_true")
    parser.add_argument("--exclude-rmh", action="store_true", help="sensitivity: drop HEP1 RMH patients")
    parser.add_argument("--exclude-hep-pids", type=Path, default=None,
                        help="file of confirmed duplicate HEP1 pids (one per line) to drop")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true", help="1 fold, 2 epochs, 1 size-matched draw")
    parser.add_argument("--refit-folds", type=int, default=0, dest="refit_folds",
                        help="refit protocol (analysis plan B.2): inner folds for epoch selection; 0 = inner split")
    parser.add_argument("--hep-outcome", choices=HEP_OUTCOMES, default="provided", dest="hep_outcome",
                        help="HEP1 label: provided, or the 12-month harmonised sensitivity label (B.5)")
    args = parser.parse_args()
    set_refit_folds(args.refit_folds)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.smoke:
        portable.N_EPOCHS_MAX = portable.EEG_N_EPOCHS_MAX = 2
        portable.EARLY_STOP_PATIENCE = portable.EEG_EARLY_STOP_PATIENCE = 1
        if args.refit_folds:
            args.refit_folds = min(args.refit_folds, 2)   # as shared.cv_splits SMOKE_INNER_FOLDS
            set_refit_folds(args.refit_folds)
    excluded = []
    if args.exclude_hep_pids:
        excluded = [s.strip() for s in args.exclude_hep_pids.read_text().splitlines() if s.strip()]
    variant = (f"_rf{args.refit_folds}" if args.refit_folds else "") + hep_outcome_tag(args.hep_outcome) \
        + ("_noRMH" if args.exclude_rmh else "") + ("_dedup" if excluded else "") \
        + ("_smoke" if args.smoke else "")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for cfg in args.config:
        seeds = args.seeds if args.seeds else SEEDS[cfg]
        todo = [s for s in seeds if args.force or not
                (args.out_dir / f"predictions_{cfg}{variant}_seed{s}.csv").exists()]
        if not todo:
            logger.info(f"{cfg}: all seeds done, skipping")
            continue
        pooled = load_pooled(cfg, exclude_rmh=args.exclude_rmh, exclude_hep_pids=excluded,
                             hep_outcome=args.hep_outcome)
        counts = pooled.df.groupby("cohort")["outcome"].agg(["size", "mean"]).round(3).to_dict("index")
        logger.info(f"{cfg}: pooled n={len(pooled.df)} {counts}; device {device}")
        for seed in todo:
            preds, folds = run_seed(
                pooled, seed, device, arms=args.arms,
                sizematch=(cfg in SIZEMATCH_CONFIGS and not args.no_sizematch),
                max_folds=1 if args.smoke else None, draws=1 if args.smoke else SIZEMATCH_DRAWS,
            )
            stem = f"{cfg}{variant}_seed{seed}"
            # Predictions last: their presence marks the seed as done.
            folds.to_csv(args.out_dir / f"folds_{stem}.csv", index=False)
            preds.to_csv(args.out_dir / f"predictions_{stem}.csv", index=False)
            (args.out_dir / f"run_{stem}.json").write_text(json.dumps({
                "config": cfg, "seed": seed, "arms": args.arms, "variant": variant,
                "exclude_rmh": args.exclude_rmh, "excluded_hep_pids": len(excluded),
                "n_pooled": len(pooled.df), "hep_outcome": args.hep_outcome,
                "protocol": "refit" if args.refit_folds else "innersplit", "refit_folds": args.refit_folds,
                "inner_frac": 0.0 if args.refit_folds else INNER_FRAC,
                "provenance": {**run_provenance(), "cv_seed": seed},
                "eeg_inputs": [dict(e) for e in LOADED_INPUTS],   # the caches this process loaded (version, build time, commit)
            }, indent=2))
            logger.info(f"wrote predictions_{stem}.csv ({len(preds)} rows)")


if __name__ == "__main__":
    main()
