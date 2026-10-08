"""Analyse Experiment 14 Optuna tuning results.

Usage:
    python -m exp14_optuna_tuning.analyse_results               # Comparison table
    python -m exp14_optuna_tuning.analyse_results --rerun-best   # Full metrics for best params
    python -m exp14_optuna_tuning.analyse_results --importance    # Parameter importance
"""

import argparse
import json
import logging
from datetime import datetime

import numpy as np
from exp2_fusion.config import N_CHANNELS
import optuna

from .config import BASELINES, MODEL_NAMES, RESULTS_DIR, STUDY_DB_PATH

logger = logging.getLogger(__name__)

MODEL_DISPLAY_NAMES = {
    "exp7a_mlp": "Exp7a QuadFusionMLP",
    "exp11_quadmlpv2": "Exp11 QuadMLPv2 (EEG2Vec)",
    "exp12_fusemoe": "Exp12 TripleFuseMoE",
}


def load_studies():
    """Load all Optuna studies from SQLite database.

    Returns:
        Dict mapping model_name to Study object (only those with trials).
    """
    studies = {}
    storage = f"sqlite:///{STUDY_DB_PATH}"

    for model_name in MODEL_NAMES:
        study_name = f"exp14_{model_name}"
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage,
            )
            n_complete = len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            if n_complete > 0:
                studies[model_name] = study
                logger.info(
                    f"Loaded {study_name}: {n_complete} complete trials, "
                    f"best AUC={study.best_value:.4f}"
                )
            else:
                logger.warning(f"Study {study_name} has no complete trials")
        except Exception as e:
            logger.warning(f"Could not load study {study_name}: {e}")

    return studies


def print_comparison_table(studies):
    """Print baseline vs tuned AUC comparison table."""
    print("\n" + "=" * 80)
    print("Experiment 14: Optuna HP Tuning Results")
    print("=" * 80)
    print(
        f"{'Model':<30} {'Baseline AUC':>14} {'Tuned AUC':>12} "
        f"{'Delta':>8} {'Trials':>8}"
    )
    print("-" * 80)

    for model_name in MODEL_NAMES:
        display = MODEL_DISPLAY_NAMES[model_name]
        baseline = BASELINES[model_name]["auc_mean"]

        if model_name in studies:
            study = studies[model_name]
            best_auc = study.best_value
            delta = best_auc - baseline
            n_complete = len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            print(
                f"{display:<30} {baseline:>14.4f} {best_auc:>12.4f} "
                f"{delta:>+8.4f} {n_complete:>8}"
            )
        else:
            print(f"{display:<30} {baseline:>14.4f} {'(no data)':>12}")

    print("=" * 80)


def print_best_params(studies):
    """Print best parameters for each model."""
    for model_name, study in studies.items():
        display = MODEL_DISPLAY_NAMES[model_name]
        best = study.best_trial
        print(f"\n{display} - Best trial #{best.number} (AUC={best.value:.4f}):")
        for param, value in sorted(best.params.items()):
            print(f"  {param}: {value}")


def print_param_importance(studies):
    """Print parameter importance rankings using fANOVA."""
    for model_name, study in studies.items():
        display = MODEL_DISPLAY_NAMES[model_name]
        n_complete = len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])

        if n_complete < 10:
            print(f"\n{display}: Need >= 10 complete trials for importance "
                  f"(have {n_complete})")
            continue

        try:
            importances = optuna.importance.get_param_importances(study)
            print(f"\n{display} - Parameter Importance:")
            for param, importance in importances.items():
                bar = "#" * int(importance * 40)
                print(f"  {param:<20} {importance:.3f} {bar}")
        except Exception as e:
            print(f"\n{display}: Could not compute importance: {e}")


def rerun_best_params(studies):
    """Rerun best parameters with full metric collection.

    This produces results in the same format as all other experiments,
    ensuring fair comparison.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, study in studies.items():
        display = MODEL_DISPLAY_NAMES[model_name]
        best_params = study.best_trial.params
        print(f"\nRerunning {display} with best params: {best_params}")

        if model_name == "exp7a_mlp":
            fold_metrics = _rerun_exp7a(best_params, device)
        elif model_name == "exp11_quadmlpv2":
            fold_metrics = _rerun_exp11(best_params, device)
        elif model_name == "exp12_fusemoe":
            fold_metrics = _rerun_exp12(best_params, device)
        else:
            continue

        _save_full_results(model_name, fold_metrics, best_params, study)


def _rerun_exp7a(params, device):
    """Rerun Exp7a with full metric collection."""
    from sklearn.model_selection import StratifiedKFold

    from exp7_all_modalities.config import (
        CLINICAL_DIM,
        EEG_ENCODER_CONFIG,
        SMILES_DIM,
        TEXT_DIM,
    )
    from exp7_all_modalities.data_pipeline import (
        create_quad_modality_datasets,
        prepare_quad_modality_data,
    )
    from exp7_all_modalities.models import QuadFusionMLP
    from exp7_all_modalities.training import evaluate_mlp, train_epoch_mlp

    from .config import CV_CONFIG, TRAINING_FIXED
    from .objectives import _get_exp7a_data

    import torch.nn as nn

    df, smiles_emb, smiles_idx, text_emb, eeg_data = _get_exp7a_data()
    outcomes = df["outcome"].values
    kfold = StratifiedKFold(**CV_CONFIG)

    fold_metrics = {
        "auc": [], "accuracy": [], "f1": [],
        "f1_tuned": [], "balanced_acc_tuned": [],
    }

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(np.zeros(len(outcomes)), outcomes)
    ):
        train_ds, val_ds, _ = create_quad_modality_datasets(
            df, smiles_emb, smiles_idx, text_emb, eeg_data, train_idx, val_idx
        )

        model = QuadFusionMLP(
            clinical_dim=CLINICAL_DIM, text_dim=TEXT_DIM, smiles_dim=SMILES_DIM,
            hidden_dim=params["hidden_dim"], num_classes=2, dropout=params["dropout"],
            eeg_encoder_type=EEG_ENCODER_CONFIG["encoder_type"],
            n_channels=EEG_ENCODER_CONFIG["n_channels"],
            n_times=EEG_ENCODER_CONFIG["n_times"],
            max_windows=EEG_ENCODER_CONFIG["max_windows"],
            window_chunk_size=EEG_ENCODER_CONFIG["window_chunk_size"],
        ).to(device)

        train_loader = __import__("torch.utils.data", fromlist=["DataLoader"]).DataLoader(
            train_ds, batch_size=params["batch_size"], shuffle=True, num_workers=0
        )
        val_loader = __import__("torch.utils.data", fromlist=["DataLoader"]).DataLoader(
            val_ds, batch_size=params["batch_size"], shuffle=False, num_workers=0
        )

        train_labels = [train_ds[i][5].item() for i in range(len(train_ds))]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()
        import torch
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        optimiser = torch.optim.AdamW(
            model.parameters(), lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_auc = 0.0
        best_metrics = {}
        patience_counter = 0

        for epoch in range(TRAINING_FIXED["epochs"]):
            train_epoch_mlp(model, train_loader, optimiser, criterion, device)
            _, val_metrics = evaluate_mlp(model, val_loader, criterion, device)

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_metrics = val_metrics.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= TRAINING_FIXED["patience"]:
                break

        for key in fold_metrics:
            fold_metrics[key].append(best_metrics.get(key, 0.0))

        print(f"  Fold {fold + 1}: AUC={best_metrics['auc']:.4f}, "
              f"BalAcc={best_metrics.get('balanced_acc_tuned', 0):.4f}")

    return fold_metrics


def _rerun_exp11(params, device):
    """Rerun Exp11 with full metric collection."""
    from sklearn.model_selection import StratifiedKFold

    from exp7_all_modalities.config import CLINICAL_DIM, SMILES_DIM, TEXT_DIM
    from exp7_all_modalities.data_pipeline import create_quad_modality_datasets
    from exp7_all_modalities.training import evaluate_mlp, train_epoch_mlp
    from exp11_eeg_upgrade.models import QuadMLPv2

    from .config import CV_CONFIG, TRAINING_FIXED
    from .objectives import _get_exp7a_data

    import torch
    import torch.nn as nn

    df, smiles_emb, smiles_idx, text_emb, eeg_data = _get_exp7a_data()
    outcomes = df["outcome"].values
    kfold = StratifiedKFold(**CV_CONFIG)

    fold_metrics = {
        "auc": [], "accuracy": [], "f1": [],
        "f1_tuned": [], "balanced_acc_tuned": [],
    }

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(np.zeros(len(outcomes)), outcomes)
    ):
        train_ds, val_ds, _ = create_quad_modality_datasets(
            df, smiles_emb, smiles_idx, text_emb, eeg_data, train_idx, val_idx
        )

        model = QuadMLPv2(
            clinical_dim=CLINICAL_DIM, text_dim=TEXT_DIM, smiles_dim=SMILES_DIM,
            hidden_dim=params["hidden_dim"], num_classes=2, dropout=params["dropout"],
            eeg_encoder_type="eeg2vec", eeg_embed_dim=params["eeg_embed_dim"],
            aggregator_type=params["aggregator_type"],
            n_channels=N_CHANNELS, n_times=2000, max_windows=120, window_chunk_size=32,
        ).to(device)

        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_ds, batch_size=params["batch_size"], shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=params["batch_size"], shuffle=False, num_workers=0
        )

        train_labels = [train_ds[i][5].item() for i in range(len(train_ds))]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        optimiser = torch.optim.AdamW(
            model.parameters(), lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_auc = 0.0
        best_metrics = {}
        patience_counter = 0

        for epoch in range(TRAINING_FIXED["epochs"]):
            train_epoch_mlp(model, train_loader, optimiser, criterion, device)
            _, val_metrics = evaluate_mlp(model, val_loader, criterion, device)

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_metrics = val_metrics.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= TRAINING_FIXED["patience"]:
                break

        for key in fold_metrics:
            fold_metrics[key].append(best_metrics.get(key, 0.0))

        print(f"  Fold {fold + 1}: AUC={best_metrics['auc']:.4f}, "
              f"BalAcc={best_metrics.get('balanced_acc_tuned', 0):.4f}")

    return fold_metrics


def _rerun_exp12(params, device):
    """Rerun Exp12 with full metric collection."""
    from sklearn.model_selection import StratifiedKFold

    from exp3_fusion.config import EEG_ENCODER_CONFIG, SMILES_DIMS
    from exp3_fusion.data_pipeline import create_datasets, get_max_channels
    from exp3_fusion.models.triple_fusemoe import TripleModalityFuseMoE
    from exp3_fusion.training import evaluate, train_epoch

    from .config import CV_CONFIG, TRAINING_FIXED
    from .objectives import _get_exp12_data

    import torch
    import torch.nn as nn

    text_emb, eeg_data, smiles_emb, smiles_idx, df = _get_exp12_data()
    outcomes = df["outcome"].values
    max_channels = get_max_channels(eeg_data)
    smiles_dim = SMILES_DIMS["chemberta"]
    kfold = StratifiedKFold(**CV_CONFIG)

    fold_metrics = {
        "auc": [], "accuracy": [], "f1": [],
        "f1_tuned": [], "balanced_acc_tuned": [],
    }

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(np.zeros(len(outcomes)), outcomes)
    ):
        train_ds, val_ds = create_datasets(
            text_emb, eeg_data, smiles_emb, smiles_idx, df,
            train_idx, val_idx, max_channels,
        )

        model = TripleModalityFuseMoE(
            text_dim=768, smiles_dim=smiles_dim, hidden_dim=256, num_classes=2,
            num_experts=params["num_experts"], top_k=params["top_k"],
            num_heads=4, dropout=params["dropout"],
            aux_loss_weight=params["aux_loss_weight"],
            eeg_encoder_type=EEG_ENCODER_CONFIG["encoder_type"],
            n_eeg_channels=EEG_ENCODER_CONFIG["n_channels"],
            n_eeg_times=EEG_ENCODER_CONFIG["n_times"],
            eeg_embed_dim=EEG_ENCODER_CONFIG["embed_dim"],
            num_eeg_layers=EEG_ENCODER_CONFIG["num_layers"],
            max_windows=EEG_ENCODER_CONFIG["max_windows"],
            window_chunk_size=EEG_ENCODER_CONFIG["window_chunk_size"],
        ).to(device)

        temp_decay = params.get("temp_decay")
        if temp_decay is None:
            model.fuse_moe.temperature_decay = 1.0
        else:
            model.fuse_moe.temperature_decay = temp_decay

        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

        train_labels = [train_ds[i][4].item() for i in range(len(train_ds))]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        optimiser = torch.optim.AdamW(
            model.parameters(), lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_auc = 0.0
        best_metrics = {}
        patience_counter = 0
        global_step = 0

        for epoch in range(TRAINING_FIXED["epochs"]):
            _, global_step = train_epoch(
                model, train_loader, optimiser, criterion, device,
                is_moe=True, global_step=global_step,
            )
            _, val_metrics = evaluate(
                model, val_loader, criterion, device, is_moe=True
            )

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_metrics = val_metrics.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= TRAINING_FIXED["patience"]:
                break

        for key in fold_metrics:
            fold_metrics[key].append(best_metrics.get(key, 0.0))

        print(f"  Fold {fold + 1}: AUC={best_metrics['auc']:.4f}, "
              f"BalAcc={best_metrics.get('balanced_acc_tuned', 0):.4f}")

    return fold_metrics


def _save_full_results(model_name, fold_metrics, params, study):
    """Save full rerun results in project-standard JSON format."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "experiment": f"exp14_{model_name}_rerun",
        "model": MODEL_DISPLAY_NAMES[model_name],
        "params": params,
        "n_samples": None,
        "n_folds": 5,
    }

    for key, values in fold_metrics.items():
        results[f"fold_{key}"] = values
        results[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    baseline = BASELINES[model_name]
    results["baseline_auc"] = baseline["auc_mean"]
    results["delta_auc"] = results["auc"]["mean"] - baseline["auc_mean"]
    results["optuna_best_auc"] = study.best_value
    results["timestamp"] = timestamp

    path = RESULTS_DIR / f"exp14_rerun_{model_name}_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{MODEL_DISPLAY_NAMES[model_name]}:")
    print(f"  AUC: {results['auc']['mean']:.4f} +/- {results['auc']['std']:.4f}")
    print(f"  Bal Acc: {results['balanced_acc_tuned']['mean']:.4f} "
          f"+/- {results['balanced_acc_tuned']['std']:.4f}")
    print(f"  Baseline AUC: {baseline['auc_mean']}")
    print(f"  Delta: {results['delta_auc']:+.4f}")
    print(f"  Saved to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse Experiment 14 Optuna results"
    )
    parser.add_argument(
        "--rerun-best",
        action="store_true",
        help="Rerun best params with full metric collection",
    )
    parser.add_argument(
        "--importance",
        action="store_true",
        help="Print parameter importance rankings",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if not STUDY_DB_PATH.exists():
        print(f"No studies found at {STUDY_DB_PATH}")
        print("Run tuning first: python -m exp14_optuna_tuning.run_tuning")
        return

    studies = load_studies()
    if not studies:
        print("No completed studies found")
        return

    print_comparison_table(studies)
    print_best_params(studies)

    if args.importance:
        print_param_importance(studies)

    if args.rerun_best:
        rerun_best_params(studies)


if __name__ == "__main__":
    main()
