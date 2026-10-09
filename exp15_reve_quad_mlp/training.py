"""Training utilities for Experiment 15: REVE-based quad-modal fusion.

Re-uses every helper from exp7_all_modalities.training (which already
handles the 6-tuple/7-tuple dataset, ASM-balancing, and prediction
logging) by passing in the exp15 dataset and model. The only thing
that changes is the model factory and the dataset factory.

The training loops in exp7 don't care about the inner shape of the
``eeg`` tensor passed through the model - they just forward it to
``model(clinical, text, eeg, mask, smiles)``. Since QuadFusionREVE
expects ``eeg`` of shape (B, max_windows, 512) and our dataset yields
exactly that, the exp7 training loop works as-is.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .config import CV_CONFIG, DEFAULT_FEATURE_SET, MLP_CONFIG
from .data_pipeline import (
    create_reve_quad_datasets,
    prepare_quad_modality_data_reve,
)
from .models import get_model as _exp15_get_model

# Reuse exp7's training functions verbatim
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp7_all_modalities.training import (  # noqa: E402
    _DropPidWrapper,
    _predict_with_smiles_override,
    _score_outer_fold,
    compute_metrics,
    evaluate_mlp,
    log_cv_summary,
    train_epoch_mlp,
)
from shared.cv_splits import current_seed, fold_indices, outer_splits, max_epochs  # noqa: E402
from shared.epoch_selection import run_outer_fold

logger = logging.getLogger("exp15")


def train_fold(
    train_dataset,
    val_dataset,
    device: torch.device,
    fold: int = 0,
    asm_balance_mode: str = "none",
    test_dataset=None,
    trace=None,
    fixed_epochs=None,
    refit_choice=None,
) -> Dict[str, float]:
    """Train and evaluate a single fold (MLP-only for exp15).

    ``val_dataset`` is the early-stopping set. ``test_dataset`` (clean runs
    only) is the untouched outer fold, scored once with the best
    early-stopping weights at the early-stopping threshold; None keeps the
    legacy behaviour (report the best epoch's metrics on ``val_dataset``).
    """
    import copy
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from shared.asm_balancing import (
        StratifiedASMBatchSampler,
        WeightedASMDataset,
        compute_asm_sample_weights,
    )

    config = MLP_CONFIG

    asm_weighted = (asm_balance_mode == "weighted")
    asm_stratified = (asm_balance_mode == "stratified_batch")
    train_asm_labels = list(train_dataset.asm_drugs)

    if asm_weighted:
        weights = compute_asm_sample_weights(train_asm_labels)
        logger.info(
            f"  ASM-weighted training: mean={weights.mean():.3f}, "
            f"min={weights.min():.3f}, max={weights.max():.3f}"
        )
        train_dataset = WeightedASMDataset(train_dataset, weights)

    if asm_stratified:
        batch_sampler = StratifiedASMBatchSampler(
            train_asm_labels,
            batch_size=config["batch_size"],
            seed=current_seed(0) + fold,
        )
        logger.info(
            f"  Stratified ASM batch sampler: {len(batch_sampler)} batches, "
            f"ASMs={len(batch_sampler.unique_asms)}"
        )
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=0)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
    val_loader = None if val_dataset is None else DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model = _exp15_get_model(fusion="mlp", device=device, reve_dim=train_dataset.feature_dim)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {n_params:,}")

    # Class weights from training-fold outcomes
    train_labels = [train_dataset[i][5].item() for i in range(len(train_dataset))]
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
    best_metrics: Dict[str, float] = {}
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs(config["epochs"]) if fixed_epochs is None else fixed_epochs):
        train_loss = train_epoch_mlp(
            model, train_loader, optimizer, criterion, device,
            asm_weighted=asm_weighted, class_weights=class_weights,
        )
        if fixed_epochs is not None:
            continue
        val_loss, val_metrics = evaluate_mlp(model, val_loader, criterion, device)
        if trace is not None:
            trace.setdefault("val_probs", []).append(np.asarray(val_metrics["y_prob"], dtype=float))
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_metrics = val_metrics.copy()
            if test_dataset is not None:
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
        # Refit protocol: final weights after exactly fixed_epochs, unthresholded.
        test_loader = DataLoader(
            _DropPidWrapper(test_dataset), batch_size=config["batch_size"],
            shuffle=False, drop_last=False, num_workers=0,
        )
        return evaluate_mlp(model, test_loader, criterion, device)[1]

    if test_dataset is None:
        return best_metrics

    # Clean protocol: restore the early-stopping-best weights and score the
    # outer fold once, at the threshold chosen on the early-stopping set.
    if best_state is not None:
        model.load_state_dict(best_state)
    return _score_outer_fold(
        model, test_dataset, evaluate_mlp, criterion, device,
        config["batch_size"], best_metrics, best_val_auc,
    )


def run_cross_validation(
    text_model: str = "clinicalbert",
    smiles_model: str = "chemberta",
    device: torch.device = None,
    asm_balance_mode: str = "none",
    splitter: str = "legacy",
    inner_val: float = 0.0,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> Dict[str, List[float]]:
    """Run 5-fold CV for exp15 quad-modal REVE.

    ``splitter`` picks the outer folds ('legacy' = outcome-only
    StratifiedKFold); ``inner_val`` is the inner early-stopping fraction
    (0 = legacy protocol, early-stop on the outer fold).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running exp15 CV ({feature_set}, {text_model}, {smiles_model})")

    df, smiles_embeddings, smiles_indices, text_embeddings, reve_data = (
        prepare_quad_modality_data_reve(text_model, smiles_model, feature_set)
    )
    outcomes = df["outcome"].values

    splits = outer_splits(
        df, mode=splitter, n_splits=CV_CONFIG["n_splits"], seed=CV_CONFIG["random_state"],
    )

    fold_metrics: Dict[str, List[float]] = {
        "auc": [], "accuracy": [], "f1": [],
        "f1_tuned": [], "balanced_acc_tuned": [],
    }

    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"Fold {fold + 1}/{CV_CONFIG['n_splits']}")
        # Clinical preprocessor is fitted on the fit set only. Clean runs select
        # on inner data (inner split, or inner folds + refit) and score the outer fold once.
        metrics, selection = run_outer_fold(
            outcomes, train_idx, val_idx, fold, inner_val,
            make_datasets=lambda a, b: create_reve_quad_datasets(
                df, smiles_embeddings, smiles_indices, text_embeddings, reve_data, a, b,
            )[:2],
            train=lambda tr, va, **kw: train_fold(
                tr, va, device=device, fold=fold, asm_balance_mode=asm_balance_mode, **kw),
            log=logger.info,
        )
        for key in fold_metrics:
            fold_metrics[key].append(metrics[key])
        logger.info(
            f"  Fold {fold + 1} results: AUC={metrics['auc']:.4f}, "
            f"BalAcc_tuned={metrics['balanced_acc_tuned']:.4f}"
        )

    log_cv_summary(fold_metrics)
    return fold_metrics


# ----------------------------------------------------------------------------
# Prediction-logging variant (mirrors exp7's train_fold_with_predictions)
# ----------------------------------------------------------------------------

def train_fold_with_predictions(
    train_dataset,
    val_dataset,
    device: torch.device,
    fold: int = 0,
    candidate_smiles: Dict[str, np.ndarray] = None,
    asm_balance_mode: str = "none",
    test_dataset=None,
    trace=None,
    fixed_epochs=None,
    refit_choice=None,
) -> Dict[str, Any]:
    """Train one fold and return per-patient predictions plus ASM-swap predictions.

    Mirrors exp7's train_fold_with_predictions but uses the exp15 model
    factory. ``val_dataset`` is the early-stopping set; with ``test_dataset``
    (clean runs) the predictions, counterfactuals and metrics are reported on
    that untouched outer fold instead, with the restored weights and the
    early-stopping threshold. The reported dataset must have been built with
    return_pid=True.
    """
    import copy
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from shared.asm_balancing import (
        StratifiedASMBatchSampler,
        WeightedASMDataset,
        compute_asm_sample_weights,
    )

    report_dataset = test_dataset if test_dataset is not None else val_dataset
    if not getattr(report_dataset, "return_pid", False):
        raise ValueError("val_dataset/test_dataset must be built with return_pid=True for prediction logging.")

    config = MLP_CONFIG

    val_loader = None if val_dataset is None else DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model = _exp15_get_model(fusion="mlp", device=device, reve_dim=train_dataset.feature_dim)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {n_params:,}")

    train_labels_list = [train_dataset[i][5].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels_list)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    asm_weighted = (asm_balance_mode == "weighted")
    asm_stratified = (asm_balance_mode == "stratified_batch")
    train_asm_labels = list(train_dataset.asm_drugs)

    train_no_pid = _DropPidWrapper(train_dataset)
    if asm_weighted:
        weights = compute_asm_sample_weights(train_asm_labels)
        logger.info(
            f"  ASM-weighted training: mean={weights.mean():.3f}, "
            f"min={weights.min():.3f}, max={weights.max():.3f}"
        )
        train_no_pid = WeightedASMDataset(train_no_pid, weights)

    if asm_stratified:
        batch_sampler = StratifiedASMBatchSampler(
            train_asm_labels,
            batch_size=config["batch_size"],
            seed=current_seed(0) + fold,
        )
        train_loader_for_loss = DataLoader(
            train_no_pid, batch_sampler=batch_sampler, num_workers=0,
        )
    else:
        train_loader_for_loss = DataLoader(
            train_no_pid,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
    val_loader_for_loss = None if val_dataset is None else DataLoader(
        _DropPidWrapper(val_dataset),
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    best_val_auc = -1.0
    best_metrics: Dict[str, float] = {}
    best_state_dict = None
    patience_counter = 0

    for epoch in range(max_epochs(config["epochs"]) if fixed_epochs is None else fixed_epochs):
        train_loss = train_epoch_mlp(
            model, train_loader_for_loss, optimizer, criterion, device,
            asm_weighted=asm_weighted, class_weights=class_weights,
        )
        if fixed_epochs is not None:
            continue
        val_loss, val_metrics = evaluate_mlp(model, val_loader_for_loss, criterion, device)
        if trace is not None:
            trace.setdefault("val_probs", []).append(np.asarray(val_metrics["y_prob"], dtype=float))

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_metrics = val_metrics.copy()
            best_state_dict = copy.deepcopy(model.state_dict())
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
        # Refit protocol: final weights after exactly fixed_epochs; the
        # metrics are unthresholded (run_outer_fold applies the inner threshold).
        best_state_dict = copy.deepcopy(model.state_dict())
        test_loader = DataLoader(
            _DropPidWrapper(test_dataset), batch_size=config["batch_size"],
            shuffle=False, drop_last=False, num_workers=0,
        )
        best_metrics = evaluate_mlp(model, test_loader, criterion, device)[1]
        val_loader = DataLoader(
            test_dataset, batch_size=config["batch_size"],
            shuffle=False, drop_last=False, num_workers=0,
        )
    elif trace is not None:
        return {}  # refit inner run: only the trace is needed
    else:
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # Clean protocol: everything below is reported on the untouched outer
        # fold; legacy runs report the early-stopping fold itself.
        if test_dataset is not None:
            best_metrics = _score_outer_fold(
                model, test_dataset, evaluate_mlp, criterion, device,
                config["batch_size"], best_metrics, best_val_auc,
            )
            val_loader = DataLoader(
                test_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )

    val_pids, val_y_true, val_y_prob = _predict_with_smiles_override(
        model, val_loader, device, fusion="mlp", smiles_override=None,
    )

    val_y_prob_per_asm: Dict[str, List[float]] = {}
    if candidate_smiles is not None:
        for asm_name, smiles_vec in candidate_smiles.items():
            override = torch.from_numpy(np.asarray(smiles_vec, dtype=np.float32))
            _, _, probs = _predict_with_smiles_override(
                model, val_loader, device, fusion="mlp", smiles_override=override,
            )
            val_y_prob_per_asm[asm_name] = probs

    return {
        "metrics": best_metrics,
        "val_pids": val_pids,
        "val_y_true": val_y_true,
        "val_y_prob": val_y_prob,
        "val_y_prob_per_asm": val_y_prob_per_asm,
        "model_state_dict": best_state_dict,
    }
