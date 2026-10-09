"""Training utilities for Experiment 5: Clinical + Single Modality Fusion."""

import copy
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

# Add parent directory for exp8_stratification import
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp8_stratification.stratified_cv import get_multilabel_splits, get_outcome_only_splits
from shared.cv_splits import outer_splits, rethreshold, current_seed, max_epochs
from shared.epoch_selection import run_outer_fold

from .config import CV_CONFIG, TRAINING_CONFIG
from .data_pipeline import (
    ClinicalEEGDataset,
    ClinicalSMILESDataset,
    ClinicalTextDataset,
    create_clinical_eeg_datasets,
    create_clinical_smiles_datasets,
    create_clinical_text_datasets,
    prepare_clinical_eeg_data,
    prepare_clinical_smiles_data,
    prepare_clinical_text_data,
)
from .models import get_model

logger = logging.getLogger("exp5")


def train_epoch_smiles(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    asm_weighted: bool = False,
    class_weights: torch.Tensor = None,
) -> float:
    """Train for one epoch (Clinical + SMILES)."""
    from shared.asm_balancing import weighted_cross_entropy
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        if asm_weighted:
            clinical, smiles, labels, sample_weights = batch
            sample_weights = sample_weights.to(device)
        else:
            clinical, smiles, labels = batch
            sample_weights = None
        clinical = clinical.to(device)
        smiles = smiles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(clinical, smiles)
        loss = weighted_cross_entropy(logits, labels, sample_weights, class_weight=class_weights) if asm_weighted \
            else criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def train_epoch_text(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    asm_weighted: bool = False,
    class_weights: torch.Tensor = None,
) -> float:
    """Train for one epoch (Clinical + Text)."""
    from shared.asm_balancing import weighted_cross_entropy
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        if asm_weighted:
            clinical, text, labels, sample_weights = batch
            sample_weights = sample_weights.to(device)
        else:
            clinical, text, labels = batch
            sample_weights = None
        clinical = clinical.to(device)
        text = text.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(clinical, text)
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
    """Train for one epoch (Clinical + EEG)."""
    from shared.asm_balancing import weighted_cross_entropy
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        if asm_weighted:
            clinical, eeg_windows, padding_mask, labels, sample_weights = batch
            sample_weights = sample_weights.to(device)
        else:
            clinical, eeg_windows, padding_mask, labels = batch
            sample_weights = None
        clinical = clinical.to(device)
        eeg_windows = eeg_windows.to(device)
        padding_mask = padding_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(clinical, eeg_windows, padding_mask)
        loss = weighted_cross_entropy(logits, labels, sample_weights, class_weight=class_weights) if asm_weighted \
            else criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate_smiles(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model (Clinical + SMILES)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            clinical, smiles, labels = batch
            clinical = clinical.to(device)
            smiles = smiles.to(device)
            labels = labels.to(device)

            logits = model(clinical, smiles)
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


def evaluate_text(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model (Clinical + Text)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            clinical, text, labels = batch
            clinical = clinical.to(device)
            text = text.to(device)
            labels = labels.to(device)

            logits = model(clinical, text)
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
    """Evaluate model (Clinical + EEG)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            clinical, eeg_windows, padding_mask, labels = batch
            clinical = clinical.to(device)
            eeg_windows = eeg_windows.to(device)
            padding_mask = padding_mask.to(device)
            labels = labels.to(device)

            logits = model(clinical, eeg_windows, padding_mask)
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
    smiles_model: str = None,
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

    asm_weighted = (asm_balance_mode == "weighted")
    asm_stratified = (asm_balance_mode == "stratified_batch")
    # asm_drugs is only present on the SMILES dataset; text/EEG datasets omit it,
    # so only require it when a balancing mode actually needs it.
    train_asm_labels = list(getattr(train_dataset, "asm_drugs", []))
    if (asm_weighted or asm_stratified) and not train_asm_labels:
        raise ValueError(
            f"asm_balance_mode={asm_balance_mode!r} needs per-sample ASM labels, but "
            f"{type(train_dataset).__name__} does not expose asm_drugs."
        )

    if asm_weighted:
        weights = compute_asm_sample_weights(train_asm_labels)
        logger.info(f"  ASM-weighted training: mean={weights.mean():.3f}, min={weights.min():.3f}, max={weights.max():.3f}")
        train_dataset = WeightedASMDataset(train_dataset, weights)

    # Create dataloaders
    if asm_stratified:
        batch_sampler = StratifiedASMBatchSampler(
            train_asm_labels,
            batch_size=config["batch_size"],
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
    if modality == "smiles":
        train_labels = [train_dataset[i][2].item() for i in range(len(train_dataset))]
    elif modality == "text":
        train_labels = [train_dataset[i][2].item() for i in range(len(train_dataset))]
    else:  # eeg
        train_labels = [train_dataset[i][3].item() for i in range(len(train_dataset))]

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
    if modality == "smiles":
        train_fn = train_epoch_smiles
        eval_fn = evaluate_smiles
    elif modality == "text":
        train_fn = train_epoch_text
        eval_fn = evaluate_text
    else:  # eeg
        train_fn = train_epoch_eeg
        eval_fn = evaluate_eeg

    # Training loop with early stopping
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
            test_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False, num_workers=0,
        )
        return eval_fn(model, test_loader, criterion, device)[1]

    if test_dataset is None:
        return best_metrics

    # Clean protocol: score the outer fold once with the early-stopping-best
    # weights, at the threshold chosen on the early-stopping set.
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False, num_workers=0,
    )
    _, test_metrics = eval_fn(model, test_loader, criterion, device)
    test_metrics = rethreshold(test_metrics, best_metrics.get("optimal_threshold", 0.5))
    test_metrics["es_auc"] = best_val_auc
    return test_metrics


def _outer_splits(
    df: pd.DataFrame,
    splitter: str,
    use_multilabel_stratification: bool,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Outer CV folds. 'legacy' is exp5's original splitter, unchanged
    (multi-label on outcome + focal + sex, or outcome-only when
    ``use_multilabel_stratification`` is False); 'multilabel' is the shared
    clean-rerun splitter (shared.cv_splits.outer_splits)."""
    if splitter != "legacy":
        return outer_splits(
            df, mode=splitter, n_splits=CV_CONFIG["n_splits"], seed=CV_CONFIG["random_state"],
        )
    if use_multilabel_stratification:
        return list(get_multilabel_splits(
            df,
            stratify_cols=["outcome", "focal", "sex"],
            n_splits=CV_CONFIG["n_splits"],
            shuffle=CV_CONFIG["shuffle"],
            random_state=CV_CONFIG["random_state"],
        ))
    return list(get_outcome_only_splits(
        df,
        n_splits=CV_CONFIG["n_splits"],
        shuffle=CV_CONFIG["shuffle"],
        random_state=CV_CONFIG["random_state"],
    ))


def run_cross_validation_smiles(
    smiles_model: str = "chemberta",
    device: torch.device = None,
    use_multilabel_stratification: bool = True,
    prediction_logger=None,
    asm_balance_mode: str = "none",
    splitter: str = "legacy",
    inner_val: float = 0.0,
) -> Dict[str, List[float]]:
    """Run 5-fold CV for Clinical + SMILES.

    Args:
        smiles_model: Type of SMILES encoder.
        device: Device to use.
        use_multilabel_stratification: Whether to use multi-label stratification
            on outcome + focal + sex (reduces fold variance by 5-8x). Legacy
            splitter only.
        splitter: Outer splitter ('legacy' = exp5's original splitter).
        inner_val: Inner early-stopping fraction (0 = legacy protocol).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    strat_type = "multilabel" if use_multilabel_stratification else "outcome-only"
    if splitter != "legacy":
        strat_type = f"shared {splitter}"
    logger.info(f"Running CV: Clinical + SMILES ({smiles_model}) with {strat_type} stratification")

    # Prepare data
    df, smiles_embeddings, smiles_indices = prepare_clinical_smiles_data(smiles_model)

    # Cross-validation with stratification
    splits = _outer_splits(df, splitter, use_multilabel_stratification)
    outcomes = df["outcome"].values

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
            make_datasets=lambda a, b: create_clinical_smiles_datasets(
                df, smiles_embeddings, smiles_indices, a, b,
            )[:2],
            train=lambda tr, va, **kw: train_fold(
                tr, va, modality="smiles", smiles_model=smiles_model, device=device, fold=fold, asm_balance_mode=asm_balance_mode, **kw),
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


def run_cross_validation_text(
    text_model: str = "clinicalbert",
    device: torch.device = None,
    use_multilabel_stratification: bool = True,
    prediction_logger=None,
    asm_balance_mode: str = "none",
    splitter: str = "legacy",
    inner_val: float = 0.0,
) -> Dict[str, List[float]]:
    """Run 5-fold CV for Clinical + Text.

    Args:
        text_model: Type of text encoder.
        device: Device to use.
        use_multilabel_stratification: Whether to use multi-label stratification
            on outcome + focal + sex (reduces fold variance by 5-8x). Legacy
            splitter only.
        splitter: Outer splitter ('legacy' = exp5's original splitter).
        inner_val: Inner early-stopping fraction (0 = legacy protocol).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    strat_type = "multilabel" if use_multilabel_stratification else "outcome-only"
    if splitter != "legacy":
        strat_type = f"shared {splitter}"
    logger.info(f"Running CV: Clinical + Text ({text_model}) with {strat_type} stratification")

    # Prepare data
    df, text_embeddings = prepare_clinical_text_data(text_model)

    # Cross-validation with stratification
    splits = _outer_splits(df, splitter, use_multilabel_stratification)
    outcomes = df["outcome"].values

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
            make_datasets=lambda a, b: create_clinical_text_datasets(
                df, text_embeddings, a, b,
            )[:2],
            train=lambda tr, va, **kw: train_fold(
                tr, va, modality="text", text_model=text_model, device=device, fold=fold, asm_balance_mode=asm_balance_mode, **kw),
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
    eeg_model: str = "simplecnn",
    device: torch.device = None,
    use_multilabel_stratification: bool = True,
    prediction_logger=None,
    asm_balance_mode: str = "none",
    splitter: str = "legacy",
    inner_val: float = 0.0,
) -> Dict[str, List[float]]:
    """Run 5-fold CV for Clinical + EEG.

    Args:
        eeg_model: Type of EEG encoder.
        device: Device to use.
        use_multilabel_stratification: Whether to use multi-label stratification
            on outcome + focal + sex (reduces fold variance by 5-8x). Legacy
            splitter only.
        splitter: Outer splitter ('legacy' = exp5's original splitter).
        inner_val: Inner early-stopping fraction (0 = legacy protocol).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    strat_type = "multilabel" if use_multilabel_stratification else "outcome-only"
    if splitter != "legacy":
        strat_type = f"shared {splitter}"
    logger.info(f"Running CV: Clinical + EEG ({eeg_model}) with {strat_type} stratification")

    # Prepare data
    df, eeg_data = prepare_clinical_eeg_data()

    # Cross-validation with stratification
    splits = _outer_splits(df, splitter, use_multilabel_stratification)
    outcomes = df["outcome"].values

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
            make_datasets=lambda a, b: create_clinical_eeg_datasets(
                df, eeg_data, a, b,
            )[:2],
            train=lambda tr, va, **kw: train_fold(
                tr, va, modality="eeg", eeg_model=eeg_model, device=device, fold=fold, asm_balance_mode=asm_balance_mode, **kw),
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

    # Test with SMILES
    results = run_cross_validation_smiles(smiles_model="chemberta", device=device)
