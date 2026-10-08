"""Run Experiment 11: EEG2Vec 128D re-runs with aggregator variants.

Re-runs exp3a (triple MLP), exp6b (clinical+SMILES+EEG), and exp7a (quad MLP)
with EEG2Vec encoder at 128D embedding dimension (based on exp9 findings).
Tests both Transformer and MeanMax aggregators.

Usage:
    python -m exp11_eeg_upgrade.run_experiments                    # All 16 configs
    python -m exp11_eeg_upgrade.run_experiments --base exp3a       # Only exp3a variants
    python -m exp11_eeg_upgrade.run_experiments --base exp6b       # Only exp6b variants
    python -m exp11_eeg_upgrade.run_experiments --base exp7a       # Only exp7a variants
    python -m exp11_eeg_upgrade.run_experiments --aggregator meanmax  # Only MeanMax
    python -m exp11_eeg_upgrade.run_experiments --dry-run           # Test data loading
    python -m exp11_eeg_upgrade.run_experiments --splitter multilabel --inner-val 0.2  # Clean CV
"""

import argparse
import copy
import json
import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local config and models
from .config import (
    CONFIG_EXP3A,
    CONFIG_EXP6B,
    CONFIG_EXP7A,
    CV_CONFIG,
    EEG_CONFIG,
    EXPERIMENTS,
    RESULTS_DIR,
    SMILES_DIMS,
)
from .models import ClinicalEEGFusionv2, QuadMLPv2, TripleMLPv2

# Import data pipelines from parent experiments
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp3_fusion.data_pipeline import prepare_data as prepare_exp3_data, create_datasets as create_exp3_datasets, get_max_channels
from exp3_fusion.training import train_epoch as train_epoch_exp3, evaluate as evaluate_exp3
from exp6_clinical_triple.data_pipeline import prepare_clinical_smiles_eeg_data, create_clinical_smiles_eeg_datasets
from exp6_clinical_triple.training import train_epoch_eeg, evaluate_eeg
from exp7_all_modalities.data_pipeline import prepare_quad_modality_data, create_quad_modality_datasets
from exp7_all_modalities.training import train_epoch_mlp as train_epoch_exp7, evaluate_mlp as evaluate_exp7
from exp2_fusion.eeg_pipeline import add_stratification_columns
from shared.asm_balancing import WeightedASMDataset, compute_asm_sample_weights
from shared.epoch_selection import run_outer_fold
from shared.cv_splits import add_cv_args, cv_suffix, outer_splits, rethreshold
from shared.cv_splits import apply_cv_args  # noqa: E402


def _asm_labels(ds):
    """Per-sample ASM list for list-based (exp6/exp7) or pid-keyed (exp3) datasets."""
    if isinstance(getattr(ds, "asm_drugs", None), dict):
        return [ds.asm_drugs[pid] for pid in ds.patient_ids]
    return list(ds.asm_drugs)


def _maybe_weight(train_ds, asm_balance_mode):
    """Wrap the train dataset with inverse-sqrt ASM weights when requested."""
    if asm_balance_mode == "weighted":
        return WeightedASMDataset(train_ds, compute_asm_sample_weights(_asm_labels(train_ds))), True
    return train_ds, False

logger = logging.getLogger("exp11")


def _outer_splits(df, splitter):
    """Outer folds. Legacy is outcome-only StratifiedKFold(**CV_CONFIG); the
    multilabel splitter gets focal/sex joined on when df lacks them (the exp3
    EEG cohort frame)."""
    split_df = df if splitter == "legacy" else add_stratification_columns(df)
    return outer_splits(
        split_df, mode=splitter, n_splits=CV_CONFIG["n_splits"], seed=CV_CONFIG["random_state"],
    )


def _train_fold_generic(model, train_loader, val_loader, config, device, train_fn, eval_fn, asm_weighted=False,
                        test_loader=None, trace=None, fixed_epochs=None, refit_choice=None):
    """Generic training fold with early stopping.

    Refit protocol (analysis plan B.2): an inner run passes ``trace`` to
    collect the early-stopping set's probabilities per epoch; the refit passes
    ``fixed_epochs`` (``val_loader`` None) and gets the outer test metrics of
    the final weights, unthresholded.

    ``val_loader`` is the early-stopping set (the outer fold in legacy runs,
    the inner split in clean runs). ``test_loader`` is given in clean runs
    only: the untouched outer fold, scored once with the best early-stopping
    weights at the early-stopping threshold. None keeps the legacy behaviour
    (report the best epoch's metrics on ``val_loader``).
    """
    # Class weights from training data. When the loader is ASM-weighted the
    # label is batch[-2] (WeightedASMDataset appends the weight as batch[-1]).
    label_pos = -2 if asm_weighted else -1
    train_labels = []
    for batch in train_loader:
        train_labels.extend(batch[label_pos].numpy())

    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_auc = 0.0
    best_metrics = {}
    best_state = None
    patience_counter = 0

    for epoch in range(config["epochs"] if fixed_epochs is None else fixed_epochs):
        train_loss = train_fn(model, train_loader, optimizer, criterion, device,
                              asm_weighted=asm_weighted, class_weights=class_weights)
        if fixed_epochs is not None:
            continue
        val_loss, val_metrics = eval_fn(model, val_loader, criterion, device)
        if trace is not None:
            trace.setdefault("val_probs", []).append(np.asarray(val_metrics["y_prob"], dtype=float))

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_metrics = val_metrics.copy()
            if test_loader is not None:
                best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"    Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_auc={val_metrics['auc']:.4f}"
            )

        if patience_counter >= config["patience"]:
            logger.info(f"    Early stopping at epoch {epoch + 1}")
            break

    if fixed_epochs is not None:
        return eval_fn(model, test_loader, criterion, device)[1]

    if test_loader is None:
        return best_metrics

    # Clean protocol: score the outer fold once with the early-stopping-best
    # weights, at the threshold chosen on the early-stopping set.
    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_metrics = eval_fn(model, test_loader, criterion, device)
    test_metrics = rethreshold(test_metrics, best_metrics.get("optimal_threshold", 0.5))
    test_metrics["es_auc"] = best_val_auc
    return test_metrics


def run_cv_exp3a(exp_config, device, prediction_logger=None, asm_balance_mode="none",
                 splitter="legacy", inner_val=0.0):
    """Run CV for exp3a-type experiment (Triple MLP: Text + EEG + SMILES)."""
    text_model = exp_config["text"]
    smiles_model = exp_config["smiles"]
    aggregator = exp_config["aggregator"]
    config = CONFIG_EXP3A

    logger.info(f"  Data: text={text_model}, smiles={smiles_model}")
    text_emb, eeg_data, smiles_emb, smiles_idx, df = prepare_exp3_data(
        text_model=text_model, smiles_model=smiles_model
    )
    smiles_dim = SMILES_DIMS[smiles_model]
    max_channels = get_max_channels(eeg_data)
    outcomes = df["outcome"].values

    fold_metrics = {"auc": [], "accuracy": [], "f1": [], "f1_tuned": [], "balanced_acc_tuned": []}

    for fold, (train_idx, val_idx) in enumerate(_outer_splits(df, splitter)):
        logger.info(f"  Fold {fold + 1}/{CV_CONFIG['n_splits']}")
        # Clean runs select on inner data (inner split, or inner folds +
        # refit) and score the outer fold once; preprocessing is fitted on
        # each training set only.
        def fit_fold(train_ds, val_ds, test_dataset=None, **kw):
            test_ds = test_dataset
            train_ds, asm_weighted = _maybe_weight(train_ds, asm_balance_mode)

            model = TripleMLPv2(
                text_dim=768,
                smiles_dim=smiles_dim,
                hidden_dim=config["hidden_dim"],
                num_classes=config["num_classes"],
                dropout=config["dropout"],
                eeg_encoder_type=EEG_CONFIG["encoder_type"],
                eeg_embed_dim=EEG_CONFIG["embed_dim"],
                aggregator_type=aggregator,
                n_eeg_channels=EEG_CONFIG["n_channels"],
                n_eeg_times=EEG_CONFIG["n_times"],
                max_windows=EEG_CONFIG["max_windows"],
                window_chunk_size=EEG_CONFIG["window_chunk_size"],
            ).to(device)

            train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
            val_loader = None if val_ds is None else DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
            test_loader = (DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
                           if test_ds is not None else None)

            return _train_fold_generic(
                model, train_loader, val_loader, config, device,
                lambda m, dl, o, c, d, asm_weighted=False, class_weights=None: train_epoch_exp3(
                    m, dl, o, c, d, is_moe=False, global_step=0,
                    asm_weighted=asm_weighted, class_weights=class_weights)[0],
                lambda m, dl, c, d: evaluate_exp3(m, dl, c, d, is_moe=False),
                asm_weighted=asm_weighted,
                test_loader=test_loader,
                **kw,
            )

        metrics, selection = run_outer_fold(
            outcomes, train_idx, val_idx, fold, inner_val,
            make_datasets=lambda a, b: create_exp3_datasets(
                text_emb, eeg_data, smiles_emb, smiles_idx, df, a, b, max_channels,
            ),
            train=fit_fold,
        )

        for key in fold_metrics:
            fold_metrics[key].append(metrics[key])
        if prediction_logger is not None and "y_prob" in metrics:
            pid_series = df["pid"].astype(str).values
            prediction_logger.log_fold(
                fold=fold,
                pids=[pid_series[i] for i in val_idx],
                y_true=metrics["y_true"],
                y_prob=metrics["y_prob"],
                threshold=metrics.get("optimal_threshold"),
                selection=selection,
            )
        logger.info(f"    AUC={metrics['auc']:.4f}, BalAcc={metrics['balanced_acc_tuned']:.4f}")

    return fold_metrics


def run_cv_exp6b(exp_config, device, prediction_logger=None, asm_balance_mode="none",
                 splitter="legacy", inner_val=0.0):
    """Run CV for exp6b-type experiment (Clinical + SMILES + EEG)."""
    smiles_model = exp_config["smiles"]
    aggregator = exp_config["aggregator"]
    config = CONFIG_EXP6B

    logger.info(f"  Data: smiles={smiles_model}")
    df, smiles_embeddings, smiles_indices, eeg_data = prepare_clinical_smiles_eeg_data(smiles_model)
    smiles_dim = SMILES_DIMS[smiles_model]
    outcomes = df["outcome"].values

    fold_metrics = {"auc": [], "accuracy": [], "f1": [], "f1_tuned": [], "balanced_acc_tuned": []}

    for fold, (train_idx, val_idx) in enumerate(_outer_splits(df, splitter)):
        logger.info(f"  Fold {fold + 1}/{CV_CONFIG['n_splits']}")
        # Clean runs select on inner data (inner split, or inner folds +
        # refit) and score the outer fold once; preprocessing is fitted on
        # each training set only.
        def fit_fold(train_ds, val_ds, test_dataset=None, **kw):
            test_ds = test_dataset
            train_ds, asm_weighted = _maybe_weight(train_ds, asm_balance_mode)

            model = ClinicalEEGFusionv2(
                smiles_dim=smiles_dim,
                hidden_dim=config["hidden_dim"],
                num_classes=config["num_classes"],
                dropout=config["dropout"],
                eeg_encoder_type=EEG_CONFIG["encoder_type"],
                eeg_embed_dim=EEG_CONFIG["embed_dim"],
                aggregator_type=aggregator,
                n_channels=EEG_CONFIG["n_channels"],
                n_times=EEG_CONFIG["n_times"],
                max_windows=EEG_CONFIG["max_windows"],
                window_chunk_size=EEG_CONFIG["window_chunk_size"],
            ).to(device)

            batch_size = config["batch_size_eeg"]
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = None if val_ds is None else DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds is not None else None

            return _train_fold_generic(
                model, train_loader, val_loader, config, device,
                train_epoch_eeg, evaluate_eeg, asm_weighted=asm_weighted,
                test_loader=test_loader,
                **kw,
            )

        metrics, selection = run_outer_fold(
            outcomes, train_idx, val_idx, fold, inner_val,
            make_datasets=lambda a, b: create_clinical_smiles_eeg_datasets(
                df, smiles_embeddings, smiles_indices, eeg_data, a, b,
            )[:2],
            train=fit_fold,
        )

        for key in fold_metrics:
            fold_metrics[key].append(metrics[key])
        if prediction_logger is not None and "y_prob" in metrics:
            pid_series = df["pid"].astype(str).values
            prediction_logger.log_fold(
                fold=fold,
                pids=[pid_series[i] for i in val_idx],
                y_true=metrics["y_true"],
                y_prob=metrics["y_prob"],
                threshold=metrics.get("optimal_threshold"),
                selection=selection,
            )
        logger.info(f"    AUC={metrics['auc']:.4f}, BalAcc={metrics['balanced_acc_tuned']:.4f}")

    return fold_metrics


def run_cv_exp7a(exp_config, device, prediction_logger=None, asm_balance_mode="none",
                 splitter="legacy", inner_val=0.0):
    """Run CV for exp7a-type experiment (Quad MLP: Clinical + Text + EEG + SMILES)."""
    text_model = exp_config["text"]
    smiles_model = exp_config["smiles"]
    aggregator = exp_config["aggregator"]
    config = CONFIG_EXP7A

    logger.info(f"  Data: text={text_model}, smiles={smiles_model}")
    df, smiles_emb, smiles_idx, text_emb, eeg_data = prepare_quad_modality_data(
        text_model, smiles_model,
    )
    smiles_dim = SMILES_DIMS[smiles_model]
    outcomes = df["outcome"].values

    fold_metrics = {"auc": [], "accuracy": [], "f1": [], "f1_tuned": [], "balanced_acc_tuned": []}

    for fold, (train_idx, val_idx) in enumerate(_outer_splits(df, splitter)):
        logger.info(f"  Fold {fold + 1}/{CV_CONFIG['n_splits']}")
        # Clean runs select on inner data (inner split, or inner folds +
        # refit) and score the outer fold once; preprocessing is fitted on
        # each training set only.
        def fit_fold(train_ds, val_ds, test_dataset=None, **kw):
            test_ds = test_dataset
            train_ds, asm_weighted = _maybe_weight(train_ds, asm_balance_mode)

            model = QuadMLPv2(
                smiles_dim=smiles_dim,
                hidden_dim=config["hidden_dim"],
                num_classes=config["num_classes"],
                dropout=config["dropout"],
                eeg_encoder_type=EEG_CONFIG["encoder_type"],
                eeg_embed_dim=EEG_CONFIG["embed_dim"],
                aggregator_type=aggregator,
                n_channels=EEG_CONFIG["n_channels"],
                n_times=EEG_CONFIG["n_times"],
                max_windows=EEG_CONFIG["max_windows"],
                window_chunk_size=EEG_CONFIG["window_chunk_size"],
            ).to(device)

            train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
            val_loader = None if val_ds is None else DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
            test_loader = (DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
                           if test_ds is not None else None)

            return _train_fold_generic(
                model, train_loader, val_loader, config, device,
                train_epoch_exp7, evaluate_exp7, asm_weighted=asm_weighted,
                test_loader=test_loader,
                **kw,
            )

        metrics, selection = run_outer_fold(
            outcomes, train_idx, val_idx, fold, inner_val,
            make_datasets=lambda a, b: create_quad_modality_datasets(
                df, smiles_emb, smiles_idx, text_emb, eeg_data, a, b,
            )[:2],
            train=fit_fold,
        )

        for key in fold_metrics:
            fold_metrics[key].append(metrics[key])
        if prediction_logger is not None and "y_prob" in metrics:
            pid_series = df["pid"].astype(str).values
            prediction_logger.log_fold(
                fold=fold,
                pids=[pid_series[i] for i in val_idx],
                y_true=metrics["y_true"],
                y_prob=metrics["y_prob"],
                threshold=metrics.get("optimal_threshold"),
                selection=selection,
            )
        logger.info(f"    AUC={metrics['auc']:.4f}, BalAcc={metrics['balanced_acc_tuned']:.4f}")

    return fold_metrics


CV_RUNNERS = {
    "exp3a": run_cv_exp3a,
    "exp6b": run_cv_exp6b,
    "exp7a": run_cv_exp7a,
}


def main():
    parser = argparse.ArgumentParser(description="Experiment 11: EEG2Vec 128D Re-runs")
    parser.add_argument("--base", choices=["exp3a", "exp6b", "exp7a"], help="Only run this base experiment type")
    parser.add_argument("--aggregator", choices=["transformer", "meanmax"], help="Only run this aggregator")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--output", type=str, help="Override output path")
    parser.add_argument("--dry-run", action="store_true", help="Test configuration only")
    parser.add_argument("--log-predictions", action="store_true",
                        help="Dump per-fold OOF predictions to outputs/exp11_predictions/")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic training (seeds, cuDNN deterministic)")
    parser.add_argument("--asm-balance", type=str, default="none",
                        choices=["none", "weighted"],
                        help="ASM class-balancing mode (weighted = inverse-sqrt sample weighting).")
    parser.add_argument("--predictions-dir", type=str, default=None,
                        help="Directory for OOF prediction files (default: outputs/exp11_predictions).")
    add_cv_args(parser)
    args = parser.parse_args()
    apply_cv_args(args)  # repeated-CV seed and refit protocol (plan B.2)

    if args.deterministic:
        from shared.determinism import enable_determinism
        enable_determinism()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Filter experiments
    experiments = EXPERIMENTS
    if args.base:
        experiments = [e for e in experiments if e["base"] == args.base]
    if args.aggregator:
        experiments = [e for e in experiments if e["aggregator"] == args.aggregator]

    logger.info(f"Running {len(experiments)} experiments")

    if args.dry_run:
        for exp in experiments:
            logger.info(f"  {exp['name']}: base={exp['base']}, aggregator={exp['aggregator']}")
        logger.info("Dry run complete.")
        return

    # Run experiments
    all_results = {}
    for exp in experiments:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {exp['name']}")
        logger.info(f"{'='*60}")

        runner = CV_RUNNERS[exp["base"]]
        pred_logger = None
        if args.log_predictions:
            from shared.prediction_logger import PredictionLogger
            pred_dir = Path(args.predictions_dir) if args.predictions_dir else RESULTS_DIR.parent / "exp11_predictions"
            suffix = {"weighted": "_asmweighted", "stratified_batch": "_asmstratbatch"}.get(args.asm_balance, "")
            suffix += cv_suffix(args.splitter, args.inner_val)
            pred_logger = PredictionLogger(
                exp_id=f"exp11_{exp['name']}",
                output_dir=pred_dir,
                filename=f"predictions_oof_exp11_{exp['name']}{suffix}.json",
                metadata={"splitter": args.splitter, "inner_val": args.inner_val,
                          "asm_balance": args.asm_balance},
            )
        fold_metrics = runner(exp, device, prediction_logger=pred_logger,
                              asm_balance_mode=args.asm_balance,
                              splitter=args.splitter, inner_val=args.inner_val)
        if pred_logger is not None:
            pred_logger.save()

        # Compute summary
        summary = {}
        for key, values in fold_metrics.items():
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        all_results[exp["name"]] = {
            "config": exp,
            "eeg_config": EEG_CONFIG,
            "fold_metrics": {k: [float(v) for v in vals] for k, vals in fold_metrics.items()},
            "summary": summary,
        }

        logger.info(
            f"  Result: AUC={summary['auc']['mean']:.4f} +/- {summary['auc']['std']:.4f}, "
            f"BalAcc={summary['balanced_acc_tuned']['mean']:.4f} +/- {summary['balanced_acc_tuned']['std']:.4f}"
        )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(RESULTS_DIR / f"results_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Print summary table
    print(f"\n{'Experiment':<45} {'AUC':>12} {'Bal Acc':>12} {'F1 Tuned':>12}")
    print("-" * 85)
    for name, result in sorted(all_results.items(), key=lambda x: x[1]["summary"]["auc"]["mean"], reverse=True):
        s = result["summary"]
        print(
            f"{name:<45} "
            f"{s['auc']['mean']:.3f}+/-{s['auc']['std']:.3f} "
            f"{s['balanced_acc_tuned']['mean']:.3f}+/-{s['balanced_acc_tuned']['std']:.3f} "
            f"{s['f1_tuned']['mean']:.3f}+/-{s['f1_tuned']['std']:.3f}"
        )


if __name__ == "__main__":
    main()
