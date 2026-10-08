"""Run Experiment 12: FuseMoE Hyperparameter Investigation.

Investigates the exp3b FuseMoE regression by testing a grid of hyperparameters:
- Learning rate: [5e-5, 1e-4, 5e-4]
- num_experts: [2, 4]
- Temperature decay: [0.9995, None]

Uses ClinicalBERT + ChemBERTa (best exp3b combo) with SimpleCNN encoder.

Usage:
    python -m exp12_moe_hparam.run_experiments               # All 12 configs
    python -m exp12_moe_hparam.run_experiments --dry-run      # Test only
    python -m exp12_moe_hparam.run_experiments --experiments exp12_lr0.0001_e4_k2_t0.9995
"""

import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .config import (
    BASE_CONFIG,
    CV_CONFIG,
    EEG_ENCODER_CONFIG,
    EXPERIMENTS,
    RESULTS_DIR,
    SMILES_DIM,
    SMILES_MODEL,
    TEXT_DIM,
    TEXT_MODEL,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp3_fusion.data_pipeline import prepare_data, create_datasets, get_max_channels
from exp3_fusion.models import TripleModalityFuseMoE
from exp3_fusion.training import train_epoch, evaluate

logger = logging.getLogger("exp12")


def create_model(exp_config, device):
    """Create TripleModalityFuseMoE with overridden hyperparameters."""
    model = TripleModalityFuseMoE(
        text_dim=TEXT_DIM,
        smiles_dim=SMILES_DIM,
        hidden_dim=BASE_CONFIG["hidden_dim"],
        num_classes=BASE_CONFIG["num_classes"],
        num_experts=exp_config["num_experts"],
        top_k=exp_config["top_k"],
        num_heads=BASE_CONFIG["num_heads"],
        dropout=BASE_CONFIG["dropout"],
        aux_loss_weight=BASE_CONFIG["aux_loss_weight"],
        eeg_encoder_type=EEG_ENCODER_CONFIG["encoder_type"],
        n_eeg_channels=EEG_ENCODER_CONFIG["n_channels"],
        n_eeg_times=EEG_ENCODER_CONFIG["n_times"],
        eeg_embed_dim=EEG_ENCODER_CONFIG["embed_dim"],
        num_eeg_layers=EEG_ENCODER_CONFIG["num_layers"],
        max_windows=EEG_ENCODER_CONFIG["max_windows"],
        window_chunk_size=EEG_ENCODER_CONFIG["window_chunk_size"],
    )

    # Override temperature decay if needed
    if exp_config["temp_decay"] is None and hasattr(model, 'fuse_moe'):
        # Disable temperature annealing by setting decay to 1.0 (no change)
        model.fuse_moe.temperature_decay = 1.0
    elif exp_config["temp_decay"] is not None and hasattr(model, 'fuse_moe'):
        model.fuse_moe.temperature_decay = exp_config["temp_decay"]

    return model.to(device)


def run_cv(exp_config, device):
    """Run 5-fold CV for a single hyperparameter configuration."""
    logger.info(f"  Preparing data: text={TEXT_MODEL}, smiles={SMILES_MODEL}")
    text_emb, eeg_data, smiles_emb, smiles_idx, df = prepare_data(
        text_model=TEXT_MODEL, smiles_model=SMILES_MODEL
    )
    max_channels = get_max_channels(eeg_data)
    outcomes = df["outcome"].values

    kfold = StratifiedKFold(**CV_CONFIG)
    fold_metrics = {"auc": [], "accuracy": [], "f1": [], "f1_tuned": [], "balanced_acc_tuned": []}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(outcomes)), outcomes)):
        logger.info(f"  Fold {fold + 1}/{CV_CONFIG['n_splits']}")
        train_ds, val_ds = create_datasets(
            text_emb, eeg_data, smiles_emb, smiles_idx, df, train_idx, val_idx, max_channels,
        )

        model = create_model(exp_config, device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"    Parameters: {n_params:,}")

        train_loader = DataLoader(train_ds, batch_size=BASE_CONFIG["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BASE_CONFIG["batch_size"], shuffle=False)

        # Class weights
        train_labels = [train_ds[i][4].item() for i in range(len(train_ds))]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=exp_config["learning_rate"],
            weight_decay=BASE_CONFIG["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Training loop
        best_val_auc = 0.0
        best_metrics = {}
        patience_counter = 0
        global_step = 0

        for epoch in range(BASE_CONFIG["epochs"]):
            train_loss, global_step = train_epoch(
                model, train_loader, optimizer, criterion, device,
                is_moe=True, global_step=global_step,
            )
            val_loss, val_metrics = evaluate(
                model, val_loader, criterion, device, is_moe=True,
            )

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_metrics = val_metrics.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"      Epoch {epoch + 1}: loss={train_loss:.4f}, "
                    f"val_auc={val_metrics['auc']:.4f}"
                )

            if patience_counter >= BASE_CONFIG["patience"]:
                logger.info(f"      Early stopping at epoch {epoch + 1}")
                break

        for key in fold_metrics:
            fold_metrics[key].append(best_metrics.get(key, 0.0))
        logger.info(f"    AUC={best_metrics['auc']:.4f}, BalAcc={best_metrics['balanced_acc_tuned']:.4f}")

    return fold_metrics


def main():
    parser = argparse.ArgumentParser(description="Experiment 12: FuseMoE Hyperparameter Investigation")
    parser.add_argument("--experiments", nargs="+", help="Only run specific experiment names")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--output", type=str, help="Override output path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    experiments = EXPERIMENTS
    if args.experiments:
        experiments = [e for e in experiments if e["name"] in args.experiments]

    logger.info(f"Running {len(experiments)} hyperparameter configurations")

    if args.dry_run:
        for exp in experiments:
            logger.info(f"  {exp['name']}: lr={exp['learning_rate']}, experts={exp['num_experts']}, "
                        f"top_k={exp['top_k']}, temp_decay={exp['temp_decay']}")
        logger.info("Dry run complete.")
        return

    all_results = {}
    for exp in experiments:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {exp['name']}")
        logger.info(f"  lr={exp['learning_rate']}, experts={exp['num_experts']}, "
                     f"top_k={exp['top_k']}, temp_decay={exp['temp_decay']}")
        logger.info(f"{'='*60}")

        fold_metrics = run_cv(exp, device)

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
            "fold_metrics": {k: [float(v) for v in vals] for k, vals in fold_metrics.items()},
            "summary": summary,
        }

        logger.info(
            f"  Result: AUC={summary['auc']['mean']:.4f} +/- {summary['auc']['std']:.4f}"
        )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(RESULTS_DIR / f"results_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Print summary table sorted by AUC
    print(f"\n{'Config':<45} {'LR':>8} {'Exp':>4} {'Temp':>8} {'AUC':>12}")
    print("-" * 85)
    for name, result in sorted(all_results.items(), key=lambda x: x[1]["summary"]["auc"]["mean"], reverse=True):
        s = result["summary"]
        c = result["config"]
        temp_str = str(c["temp_decay"]) if c["temp_decay"] else "None"
        print(
            f"{name:<45} {c['learning_rate']:>8.0e} {c['num_experts']:>4} "
            f"{temp_str:>8} {s['auc']['mean']:.3f}+/-{s['auc']['std']:.3f}"
        )

    # Compare with baseline
    print(f"\nBaseline (exp3b old FuseMoE): AUC 0.753")
    print(f"Baseline (exp3b revised FuseMoE): AUC 0.677")


if __name__ == "__main__":
    main()
