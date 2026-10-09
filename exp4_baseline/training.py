"""Training utilities for Experiment 4: Clinical features baseline."""

import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from .config import CONFIG_4A, CONFIG_4B, CV_CONFIG
from .data_pipeline import ClinicalDataset, create_datasets, load_clinical_data
from .models import get_model
from shared.asm_balancing import WeightedASMDataset, compute_asm_sample_weights, weighted_cross_entropy
from shared.cv_splits import outer_splits, rethreshold, max_epochs
from shared.epoch_selection import run_outer_fold

logger = logging.getLogger("exp4")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    asm_weighted: bool = False,
) -> float:
    """Train for one epoch.

    Args:
        model: The model to train.
        dataloader: Training data loader.
        optimizer: Optimiser.
        criterion: Loss function (its ``weight`` supplies the outcome-class
            weighting reused under ``asm_weighted``).
        device: Device to use.
        asm_weighted: If True, batches carry a trailing per-sample ASM weight
            and the loss is the ASM-weighted cross-entropy.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        if asm_weighted:
            features, labels, sample_weights = batch
            sample_weights = sample_weights.to(device)
        else:
            features, labels = batch
            sample_weights = None
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(features)
        loss = (
            weighted_cross_entropy(logits, labels, sample_weights, class_weight=criterion.weight)
            if asm_weighted else criterion(logits, labels)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model.

    Args:
        model: The model to evaluate.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device to use.

    Returns:
        Tuple of (average loss, metrics dictionary).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item()
            n_batches += 1

    # Compute metrics
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "y_prob": all_probs.tolist(),
        "y_true": all_labels.tolist(),
    }

    # AUC requires both classes present
    if len(np.unique(all_labels)) > 1:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)

        # Threshold tuning: find optimal threshold for balanced accuracy (Youden's J)
        fpr, tpr, thresholds_roc = roc_curve(all_labels, all_probs)
        youden_j = tpr - fpr  # Maximising J = maximising balanced accuracy
        best_idx = np.argmax(youden_j)
        optimal_threshold = thresholds_roc[best_idx]
        tuned_preds = (all_probs >= optimal_threshold).astype(int)
        metrics["balanced_acc_tuned"] = balanced_accuracy_score(all_labels, tuned_preds)
        metrics["f1_tuned"] = f1_score(all_labels, tuned_preds, zero_division=0)
        metrics["optimal_threshold"] = optimal_threshold
    else:
        metrics["auc"] = 0.5
        metrics["balanced_acc_tuned"] = 0.5
        metrics["f1_tuned"] = 0.0
        metrics["optimal_threshold"] = 0.5

    return total_loss / n_batches, metrics


def train_fold(
    train_dataset: ClinicalDataset,
    val_dataset: ClinicalDataset,
    model_type: str,
    device: torch.device,
    fold: int,
    asm_balance_mode: str = "none",
    test_dataset: Optional[ClinicalDataset] = None,
    trace: Optional[dict] = None,
    fixed_epochs: Optional[int] = None,
    refit_choice=None,
) -> Dict[str, float]:
    """Train and evaluate a single fold.

    Refit protocol (analysis plan B.2): an inner run passes ``trace`` to
    collect the early-stopping set's probabilities after every epoch; the
    refit passes ``fixed_epochs`` (no early-stopping set) and gets the outer
    test metrics of the final weights, unthresholded.

    Args:
        train_dataset: Training dataset.
        val_dataset: Early-stopping dataset (the outer fold in legacy runs,
            the inner split in clean runs).
        model_type: Either 'mlp' or 'attention'.
        device: Device to use.
        fold: Fold number (for logging).
        test_dataset: Clean runs only: the untouched outer fold, scored once
            with the best early-stopping weights at the early-stopping
            threshold. None keeps the legacy behaviour (report the best
            epoch's metrics on ``val_dataset``).

    Returns:
        Dictionary of metrics for the reported fold.
    """
    config = CONFIG_4A if model_type == "mlp" else CONFIG_4B

    # ASM-balancing: wrap the training set so each sample carries an
    # inverse-sqrt weight (threaded into the loss via asm_weighted).
    asm_weighted = asm_balance_mode == "weighted"
    if asm_weighted:
        weights = compute_asm_sample_weights(list(train_dataset.asm_drugs))
        train_dataset = WeightedASMDataset(train_dataset, weights)

    # Create dataloaders
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

    # Create model
    model = get_model(model_type, device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {n_params:,}")

    # Calculate class weights from training data (inverse frequency)
    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.info(f"  Class weights: {class_weights.cpu().numpy()}")

    # Optimiser and criterion
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop with early stopping
    best_val_auc = 0.0
    best_metrics = {}
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs(config["epochs"]) if fixed_epochs is None else fixed_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, asm_weighted=asm_weighted)
        if fixed_epochs is not None:
            continue
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
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
        test_loader = DataLoader(
            test_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False, num_workers=0,
        )
        return evaluate(model, test_loader, criterion, device)[1]

    if test_dataset is None:
        return best_metrics

    # Clean protocol: score the outer fold once with the early-stopping-best
    # weights, at the threshold chosen on the early-stopping set.
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False, num_workers=0,
    )
    _, test_metrics = evaluate(model, test_loader, criterion, device)
    test_metrics = rethreshold(test_metrics, best_metrics.get("optimal_threshold", 0.5))
    test_metrics["es_auc"] = best_val_auc
    return test_metrics


def run_cross_validation(
    model_type: str = "mlp",
    device: torch.device = None,
    prediction_logger=None,
    asm_balance_mode: str = "none",
    splitter: str = "legacy",
    inner_val: float = 0.0,
) -> Dict[str, List[float]]:
    """Run 5-fold stratified cross-validation.

    Args:
        model_type: Either 'mlp' or 'attention'.
        device: Device to use.
        splitter: Outer splitter ('legacy' = outcome-only StratifiedKFold).
        inner_val: Inner early-stopping fraction (0 = legacy protocol).

    Returns:
        Dictionary mapping metric names to lists of per-fold values.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running CV: model={model_type}")

    # Load data
    df = load_clinical_data()

    # Get outcomes for stratified split
    outcomes = df["outcome"].values

    # Cross-validation
    splits = outer_splits(
        df, mode=splitter, n_splits=CV_CONFIG["n_splits"], seed=CV_CONFIG["random_state"],
    )

    fold_metrics = {
        "auc": [],
        "accuracy": [],
        "f1": [],
        "f1_tuned": [],
        "balanced_acc_tuned": [],
    }

    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"Fold {fold + 1}/{CV_CONFIG['n_splits']}")

        # Preprocessor fit on the fit set only. Clean runs select on inner
        # data (inner split, or inner folds + refit) and score the outer fold once.
        metrics, selection = run_outer_fold(
            outcomes, train_idx, val_idx, fold, inner_val,
            make_datasets=lambda a, b: create_datasets(df, a, b)[:2],
            train=lambda tr, va, **kw: train_fold(
                tr, va, model_type, device, fold, asm_balance_mode=asm_balance_mode, **kw),
            log=logger.info,
        )

        if prediction_logger is not None:
            val_pids = df["pid"].iloc[val_idx].tolist()
            prediction_logger.log_fold(
                fold=fold,
                pids=val_pids,
                y_true=metrics["y_true"],
                y_prob=metrics["y_prob"],
                threshold=metrics.get("optimal_threshold"),
                selection=selection,
            )

        for key in fold_metrics:
            fold_metrics[key].append(metrics[key])

        logger.info(
            f"  Fold {fold + 1} results: AUC={metrics['auc']:.4f}, "
            f"Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
            f"BalAcc_tuned={metrics['balanced_acc_tuned']:.4f}, "
            f"F1_tuned={metrics['f1_tuned']:.4f} (thresh={metrics['optimal_threshold']:.3f})"
        )

    # Compute summary statistics
    logger.info("Cross-validation complete:")
    for key in fold_metrics:
        values = fold_metrics[key]
        mean, std = np.mean(values), np.std(values)
        min_val, max_val = np.min(values), np.max(values)
        logger.info(f"  {key}: {mean:.4f} +/- {std:.4f} (min={min_val:.4f}, max={max_val:.4f})")

    return fold_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test training on MLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = run_cross_validation(
        model_type="mlp",
        device=device,
    )
