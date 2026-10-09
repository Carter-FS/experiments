"""Training utilities for Experiment 6: Clinical + SMILES + Third Modality Fusion."""

import copy
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from shared.cv_splits import outer_splits, rethreshold, current_seed, max_epochs
from shared.epoch_selection import run_outer_fold

from .config import CV_CONFIG, TRAINING_CONFIG
from .data_pipeline import (
    ClinicalSMILESEEGDataset,
    ClinicalSMILESTextDataset,
    create_clinical_smiles_eeg_datasets,
    create_clinical_smiles_text_datasets,
    prepare_clinical_smiles_eeg_data,
    prepare_clinical_smiles_text_data,
)
from .models import get_model

logger = logging.getLogger("exp6")


def train_epoch_text(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    asm_weighted: bool = False,
    class_weights: torch.Tensor = None,
) -> float:
    """Train for one epoch (Clinical + SMILES + Text)."""
    from shared.asm_balancing import weighted_cross_entropy
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        if asm_weighted:
            clinical, smiles, text, labels, sample_weights = batch
            sample_weights = sample_weights.to(device)
        else:
            clinical, smiles, text, labels = batch
            sample_weights = None
        clinical = clinical.to(device)
        smiles = smiles.to(device)
        text = text.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(clinical, smiles, text)
        loss = weighted_cross_entropy(logits, labels, sample_weights, class_weight=class_weights) if asm_weighted \
            else criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def train_epoch_eeg(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    asm_weighted: bool = False,
    class_weights: torch.Tensor = None,
) -> float:
    """Train for one epoch (Clinical + SMILES + EEG)."""
    from shared.asm_balancing import weighted_cross_entropy
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        if asm_weighted:
            clinical, smiles, eeg_windows, padding_mask, labels, sample_weights = batch
            sample_weights = sample_weights.to(device)
        else:
            clinical, smiles, eeg_windows, padding_mask, labels = batch
            sample_weights = None
        clinical = clinical.to(device)
        smiles = smiles.to(device)
        eeg_windows = eeg_windows.to(device)
        padding_mask = padding_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(clinical, smiles, eeg_windows, padding_mask)
        loss = weighted_cross_entropy(logits, labels, sample_weights, class_weight=class_weights) if asm_weighted \
            else criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate_text(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model (Clinical + SMILES + Text)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            clinical, smiles, text, labels = batch
            clinical = clinical.to(device)
            smiles = smiles.to(device)
            text = text.to(device)
            labels = labels.to(device)

            logits = model(clinical, smiles, text)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item()
            n_batches += 1

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return total_loss / n_batches, metrics


def evaluate_eeg(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model (Clinical + SMILES + EEG)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            clinical, smiles, eeg_windows, padding_mask, labels = batch
            clinical = clinical.to(device)
            smiles = smiles.to(device)
            eeg_windows = eeg_windows.to(device)
            padding_mask = padding_mask.to(device)
            labels = labels.to(device)

            logits = model(clinical, smiles, eeg_windows, padding_mask)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item()
            n_batches += 1

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return total_loss / n_batches, metrics


def compute_metrics(
    labels: List,
    preds: List,
    probs: List,
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "y_prob": probs.tolist(),
        "y_true": labels.tolist(),
    }

    # AUC requires both classes present
    if len(np.unique(labels)) > 1:
        metrics["auc"] = roc_auc_score(labels, probs)

        # Threshold tuning: find optimal threshold for balanced accuracy (Youden's J)
        fpr, tpr, thresholds_roc = roc_curve(labels, probs)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        optimal_threshold = thresholds_roc[best_idx]
        tuned_preds = (probs >= optimal_threshold).astype(int)
        metrics["balanced_acc_tuned"] = balanced_accuracy_score(labels, tuned_preds)
        metrics["f1_tuned"] = f1_score(labels, tuned_preds, zero_division=0)
        metrics["optimal_threshold"] = optimal_threshold
    else:
        metrics["auc"] = 0.5
        metrics["balanced_acc_tuned"] = 0.5
        metrics["f1_tuned"] = 0.0
        metrics["optimal_threshold"] = 0.5

    return metrics


def train_fold(
    train_dataset,
    val_dataset,
    modality: str,
    smiles_model: str,
    text_model: str = None,
    eeg_model: str = None,
    device: torch.device = None,
    fold: int = 0,
    asm_balance_mode: str = "none",
    test_dataset=None,
    trace=None,
    fixed_epochs=None,
    refit_choice=None,
) -> Dict[str, float]:
    """Train and evaluate a single fold.

    ``val_dataset`` is the early-stopping set: the outer fold in legacy runs,
    the inner split in clean runs. ``test_dataset`` (clean runs only) is the
    untouched outer fold, scored once with the best early-stopping weights at
    the early-stopping threshold. None keeps the legacy behaviour (report the
    best epoch's metrics on ``val_dataset``).
    """
    from shared.asm_balancing import (
        WeightedASMDataset,
        StratifiedASMBatchSampler,
        compute_asm_sample_weights,
    )
    config = TRAINING_CONFIG

    # Get batch size based on modality
    batch_size = config["batch_size_text"] if modality == "text" else config["batch_size_eeg"]

    asm_weighted = (asm_balance_mode == "weighted")
    asm_stratified = (asm_balance_mode == "stratified_batch")
    train_asm_labels = list(train_dataset.asm_drugs)

    if asm_weighted:
        weights = compute_asm_sample_weights(train_asm_labels)
        logger.info(f"  ASM-weighted training: mean={weights.mean():.3f}, min={weights.min():.3f}, max={weights.max():.3f}")
        train_dataset = WeightedASMDataset(train_dataset, weights)

    # Create dataloaders
    if asm_stratified:
        batch_sampler = StratifiedASMBatchSampler(
            train_asm_labels,
            batch_size=batch_size,
            seed=current_seed(0) + fold,
        )
        logger.info(f"  Stratified ASM batch sampler: {len(batch_sampler)} batches, ASMs={len(batch_sampler.unique_asms)}")
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
    val_loader = None if val_dataset is None else DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # Create model
    model = get_model(
        modality=modality,
        smiles_model=smiles_model,
        text_model=text_model,
        eeg_model=eeg_model,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {n_params:,}")

    # Calculate class weights from training data
    if modality == "text":
        train_labels = [train_dataset[i][3].item() for i in range(len(train_dataset))]
    else:  # eeg
        train_labels = [train_dataset[i][4].item() for i in range(len(train_dataset))]

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

    # Select train/eval functions based on modality
    if modality == "text":
        train_fn = train_epoch_text
        eval_fn = evaluate_text
    else:  # eeg
        train_fn = train_epoch_eeg
        eval_fn = evaluate_eeg

    # Training loop with early stopping. When asm_weighted is active,
    # the train_loader yields a 5-tuple (or 6-tuple for eeg) ending in
    # the sample-weight tensor; the per-modality train_epoch functions
    # detect this via the asm_weighted flag.
    best_val_auc = 0.0
    best_metrics = {}
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs(config["epochs"]) if fixed_epochs is None else fixed_epochs):
        train_loss = train_fn(
            model, train_loader, optimizer, criterion, device,
            asm_weighted=asm_weighted, class_weights=class_weights,
        )
        if fixed_epochs is not None:
            continue
        val_loss, val_metrics = eval_fn(model, val_loader, criterion, device)
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
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
        )
        return eval_fn(model, test_loader, criterion, device)[1]

    if test_dataset is None:
        return best_metrics

    # Clean protocol: score the outer fold once with the early-stopping-best
    # weights, at the threshold chosen on the early-stopping set.
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
    )
    _, test_metrics = eval_fn(model, test_loader, criterion, device)
    test_metrics = rethreshold(test_metrics, best_metrics.get("optimal_threshold", 0.5))
    test_metrics["es_auc"] = best_val_auc
    return test_metrics


def run_cross_validation_text(
    smiles_model: str = "chemberta",
    text_model: str = "clinicalbert",
    device: torch.device = None,
    prediction_logger=None,
    asm_balance_mode: str = "none",
    splitter: str = "legacy",
    inner_val: float = 0.0,
) -> Dict[str, List[float]]:
    """Run 5-fold CV for Clinical + SMILES + Text.

    ``splitter`` 'legacy' is the original outcome-only StratifiedKFold;
    ``inner_val`` 0 keeps the legacy protocol (early stopping on the outer fold).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running CV: Clinical + SMILES + Text ({smiles_model}, {text_model})")

    # Prepare data
    df, smiles_embeddings, smiles_indices, text_embeddings = prepare_clinical_smiles_text_data(
        smiles_model, text_model
    )
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

        # Preprocessor fit on the fit set only. Clean runs select on
        # inner data (inner split, or inner folds + refit) and score the outer fold once.
        metrics, selection = run_outer_fold(
            outcomes, train_idx, val_idx, fold, inner_val,
            make_datasets=lambda a, b: create_clinical_smiles_text_datasets(
                df, smiles_embeddings, smiles_indices, text_embeddings, a, b,
            )[:2],
            train=lambda tr, va, **kw: train_fold(
                tr, va, modality="text", smiles_model=smiles_model, text_model=text_model, device=device, fold=fold, asm_balance_mode=asm_balance_mode, **kw),
            log=logger.info,
        )

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

        for key in fold_metrics:
            fold_metrics[key].append(metrics[key])

        logger.info(
            f"  Fold {fold + 1} results: AUC={metrics['auc']:.4f}, "
            f"BalAcc_tuned={metrics['balanced_acc_tuned']:.4f}"
        )

    log_cv_summary(fold_metrics)
    return fold_metrics


def run_cross_validation_eeg(
    smiles_model: str = "chemberta",
    eeg_model: str = "simplecnn",
    device: torch.device = None,
    prediction_logger=None,
    asm_balance_mode: str = "none",
    splitter: str = "legacy",
    inner_val: float = 0.0,
) -> Dict[str, List[float]]:
    """Run 5-fold CV for Clinical + SMILES + EEG.

    ``splitter`` 'legacy' is the original outcome-only StratifiedKFold;
    ``inner_val`` 0 keeps the legacy protocol (early stopping on the outer fold).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running CV: Clinical + SMILES + EEG ({smiles_model}, {eeg_model})")

    # Prepare data
    df, smiles_embeddings, smiles_indices, eeg_data = prepare_clinical_smiles_eeg_data(smiles_model)
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

        # Preprocessor fit on the fit set only. Clean runs select on
        # inner data (inner split, or inner folds + refit) and score the outer fold once.
        metrics, selection = run_outer_fold(
            outcomes, train_idx, val_idx, fold, inner_val,
            make_datasets=lambda a, b: create_clinical_smiles_eeg_datasets(
                df, smiles_embeddings, smiles_indices, eeg_data, a, b,
            )[:2],
            train=lambda tr, va, **kw: train_fold(
                tr, va, modality="eeg", smiles_model=smiles_model, eeg_model=eeg_model, device=device, fold=fold, asm_balance_mode=asm_balance_mode, **kw),
            log=logger.info,
        )

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

        for key in fold_metrics:
            fold_metrics[key].append(metrics[key])

        logger.info(
            f"  Fold {fold + 1} results: AUC={metrics['auc']:.4f}, "
            f"BalAcc_tuned={metrics['balanced_acc_tuned']:.4f}"
        )

    log_cv_summary(fold_metrics)
    return fold_metrics


def log_cv_summary(fold_metrics: Dict[str, List[float]]):
    """Log cross-validation summary."""
    logger.info("Cross-validation complete:")
    for key in fold_metrics:
        values = fold_metrics[key]
        mean, std = np.mean(values), np.std(values)
        min_val, max_val = np.min(values), np.max(values)
        logger.info(f"  {key}: {mean:.4f} +/- {std:.4f} (min={min_val:.4f}, max={max_val:.4f})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test with text
    results = run_cross_validation_text(
        smiles_model="chemberta",
        text_model="clinicalbert",
        device=device,
    )
