"""Run Experiment 18: mixed-cohort training with per-cohort evaluation.

For each configuration and seed, one outer 5-fold split of the pooled
Melbourne + HEP1 cohort (stratified on outcome x cohort). Every arm trains on
its share of each outer training fold (mixed / Melbourne only / HEP1 only),
early-stops on a 20% inner split of that share, and scores every outer test
patient in both cohorts, so all arms are compared on the same test patients.
Exp4a also runs size-matched mixed training (10 draws per test cohort).

    python -m exp18_mixed_cohort.run_experiments --config Exp4a            # all its seeds
    python -m exp18_mixed_cohort.run_experiments --config Exp4a Exp5a --seeds 42
    python -m exp18_mixed_cohort.run_experiments --config Exp4a --exclude-rmh   # sensitivity
    python -m exp18_mixed_cohort.run_experiments --config Exp4a --smoke    # 1 fold, 2 epochs

Outputs (outputs/exp18_mixed_cohort/, patient-level, gitignored):
    predictions_<cfg><variant>_seed<k>.csv   one row per (arm, draw, test patient)
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
from shared.cv_splits import inner_val_split, outer_splits, set_repeat_seed, youden_threshold
from shared.determinism import enable_determinism
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


def train_predict(pooled: PooledCohort, fit_idx, es_idx, test_idx, device) -> tuple[np.ndarray, float]:
    """Train on ``fit_idx``, early-stop on ``es_idx``; return (test probs, es threshold)."""
    cfg, y = PORTABLE_MODEL[pooled.config], pooled.labels
    if pooled.config in LF_INPUTS:
        return _train_predict_late_fusion(pooled, cfg, fit_idx, es_idx, test_idx, device)
    mods = dict(pooled.modalities)
    mods["clinical"] = clinical_features(pooled, fit_idx)
    if cfg in portable.EEG_CONFIGS:
        sub = lambda idx: portable.index_modalities(mods, idx)  # noqa: E731
        model = portable.train_fold_eeg(cfg, sub(fit_idx), sub(es_idx), y[fit_idx], y[es_idx], device)
        es_probs = portable.predict_eeg(model, sub(es_idx), cfg, device)
        test_probs = portable.predict_eeg(model, sub(test_idx), cfg, device)
    else:
        tensors = [mods["clinical"]] + [mods[k] for k in ("smiles", "text") if k in mods]
        sub = lambda idx: [t[idx] for t in tensors]  # noqa: E731
        model = portable.train_fold(cfg, sub(fit_idx), sub(es_idx), y[fit_idx], y[es_idx], device)
        es_probs = portable.predict(model, sub(es_idx), cfg, device)
        test_probs = portable.predict(model, sub(test_idx), cfg, device)
    return test_probs, youden_threshold(y[es_idx].numpy(), es_probs)


def _train_predict_late_fusion(pooled: PooledCohort, cfg: str, fit_idx, es_idx, test_idx, device):
    """exp19 configurations (LateFusionMLP): embedding inputs z-scored and clinical
    preprocessors fitted on this arm's fit rows."""
    from exp19_serialised_clinical.tabular import FullClinicalPreprocessor
    y, tensors = pooled.labels, []
    for name in LF_INPUTS[pooled.config]:
        if name == "clinical":
            tensors.append(clinical_features(pooled, fit_idx))
        elif name == "clinical_full":
            pre = FullClinicalPreprocessor().fit(pooled.df.iloc[fit_idx])
            tensors.append(torch.from_numpy(pre.transform(pooled.df)))
        else:
            x = pooled.modalities[name].float()
            mu, sd = x[fit_idx].mean(0), x[fit_idx].std(0, unbiased=False).clamp(min=1e-6)
            tensors.append((x - mu) / sd)
    dims = [t.shape[1] for t in tensors]
    sub = lambda idx: [t[idx] for t in tensors]  # noqa: E731
    model = portable.train_fold(cfg, sub(fit_idx), sub(es_idx), y[fit_idx], y[es_idx], device,
                                model_factory=lambda: portable.LateFusionMLP(dims))
    es_probs = portable.predict(model, sub(es_idx), cfg, device)
    return portable.predict(model, sub(test_idx), cfg, device), youden_threshold(y[es_idx].numpy(), es_probs)


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

    def record(arm, draw, fold, test_idx, probs, thr, n_fit, n_es):
        for i, p in zip(test_idx, probs):
            rows.append({"config": pooled.config, "seed": seed, "fold": fold, "arm": arm, "draw": draw,
                         "pid": df["pid"].iat[i], "cohort": cohort[i], "y_true": int(y[i]),
                         "y_prob": float(p), "threshold": thr, "n_fit": n_fit, "n_es": n_es})

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
            fit, es = inner_val_split(pooled.key, arm_tr, INNER_FRAC, seed + fold)
            enable_determinism(seed + fold)  # same init for every arm of a fold
            t0 = time.time()
            probs, thr = train_predict(pooled, fit, es, te, device)
            record(arm, -1, fold, te, probs, thr, len(fit), len(es))
            aucs = {c: roc_auc_score(y[te][cohort[te] == c], probs[cohort[te] == c])
                    for c in ("MEL", "HEP") if len(set(y[te][cohort[te] == c])) > 1}
            logger.info(f"{pooled.config} seed {seed} fold {fold} {arm:8s} fit {len(fit):3d}  "
                        + "  ".join(f"{c} AUC {a:.3f}" for c, a in aucs.items())
                        + f"  ({time.time() - t0:.0f}s)")
        if sizematch:
            for c in ("MEL", "HEP"):
                n_own = int((cohort[tr] == c).sum())
                test_c = te[cohort[te] == c]
                for draw in range(draws):
                    rs = seed * 10_000 + fold * 100 + draw * 2 + (c == "HEP")
                    sub = sizematched_subsample(pooled, tr, n_own, rs)
                    fit, es = inner_val_split(pooled.key, sub, INNER_FRAC, seed + fold)
                    enable_determinism(seed + fold)
                    probs, thr = train_predict(pooled, fit, es, test_c, device)
                    record(f"sizematched_{c}", draw, fold, test_c, probs, thr, len(fit), len(es))
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.smoke:
        portable.N_EPOCHS_MAX = portable.EEG_N_EPOCHS_MAX = 2
        portable.EARLY_STOP_PATIENCE = portable.EEG_EARLY_STOP_PATIENCE = 1
    excluded = []
    if args.exclude_hep_pids:
        excluded = [s.strip() for s in args.exclude_hep_pids.read_text().splitlines() if s.strip()]
    variant = ("_noRMH" if args.exclude_rmh else "") + ("_dedup" if excluded else "") \
        + ("_smoke" if args.smoke else "")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for cfg in args.config:
        seeds = args.seeds if args.seeds else SEEDS[cfg]
        todo = [s for s in seeds if args.force or not
                (args.out_dir / f"predictions_{cfg}{variant}_seed{s}.csv").exists()]
        if not todo:
            logger.info(f"{cfg}: all seeds done, skipping")
            continue
        pooled = load_pooled(cfg, exclude_rmh=args.exclude_rmh, exclude_hep_pids=excluded)
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
                "n_pooled": len(pooled.df), "inner_frac": INNER_FRAC, "provenance": {**run_provenance(), "cv_seed": seed},
            }, indent=2))
            logger.info(f"wrote predictions_{stem}.csv ({len(preds)} rows)")


if __name__ == "__main__":
    main()
