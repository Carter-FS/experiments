"""Training loop for Experiment 8: Stratification Analysis.

Compares different stratification methods using the same model architecture.
"""

import logging
from typing import Dict, Generator, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from .config import TRAINING_CONFIG, EEG_ENCODER_CONFIG
from .data_pipeline import create_dataset_from_indices

# Import model from exp7
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp4_baseline.config import CLINICAL_DIM  # noqa: E402
from exp7_all_modalities.models import QuadFusionMLP

logger = logging.getLogger("exp8")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        clinical, text, eeg, mask, smiles, labels = batch
        clinical = clinical.to(device)
        text = text.to(device)
        eeg = eeg.to(device)
        mask = mask.to(device)
        smiles = smiles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(clinical, text, eeg, mask, smiles)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model.

    Returns:
        Tuple of (loss, metrics_dict, all_labels, all_probs).
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            clinical, text, eeg, mask, smiles, labels = batch
            clinical = clinical.to(device)
            text = text.to(device)
            eeg = eeg.to(device)
            mask = mask.to(device)
            smiles = smiles.to(device)
            labels = labels.to(device)

            logits = model(clinical, text, eeg, mask, smiles)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    preds = (all_probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(all_labels, preds),
        "f1": f1_score(all_labels, preds, zero_division=0),
    }

    # AUC (only if both classes present)
    if len(np.unique(all_labels)) > 1:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    else:
        metrics["auc"] = 0.5

    return total_loss / len(loader), metrics, all_labels, all_probs


def train_fold(
    train_dataset,
    val_dataset,
    config: Dict,
    device: torch.device,
) -> Dict[str, float]:
    """Train and evaluate one fold.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        config: Training configuration.
        device: Torch device.

    Returns:
        Dict of fold metrics.
    """
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # Create model
    model = QuadFusionMLP(
        clinical_dim=CLINICAL_DIM,
        text_dim=768,
        smiles_dim=768,
        n_channels=EEG_ENCODER_CONFIG["n_channels"],
        n_times=EEG_ENCODER_CONFIG["n_times"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
        max_windows=EEG_ENCODER_CONFIG["max_windows"],
        window_chunk_size=EEG_ENCODER_CONFIG["window_chunk_size"],
    ).to(device)

    # Class weights
    train_labels = [train_dataset[i][-1].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Training loop with early stopping
    best_val_auc = 0.0
    best_metrics = {}
    best_probs = None
    best_labels = None
    patience_counter = 0

    for epoch in range(config["epochs"]):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_metrics = val_metrics.copy()
            best_probs = val_probs.copy()
            best_labels = val_labels.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            logger.debug(f"Early stopping at epoch {epoch + 1}")
            break

    # Compute tuned metrics using optimal threshold
    if best_probs is not None and len(np.unique(best_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(best_labels, best_probs)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[best_idx]

        tuned_preds = (best_probs >= optimal_threshold).astype(int)
        best_metrics["balanced_acc_tuned"] = balanced_accuracy_score(
            best_labels, tuned_preds
        )
        best_metrics["f1_tuned"] = f1_score(best_labels, tuned_preds, zero_division=0)
        best_metrics["optimal_threshold"] = optimal_threshold
    else:
        best_metrics["balanced_acc_tuned"] = best_metrics.get("accuracy", 0.5)
        best_metrics["f1_tuned"] = best_metrics.get("f1", 0.0)
        best_metrics["optimal_threshold"] = 0.5

    return best_metrics


def run_cv_experiment(
    df,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    smiles_embeddings,
    smiles_indices,
    text_embeddings,
    eeg_data,
    config: Dict = None,
    device: torch.device = None,
) -> Dict[str, List[float]]:
    """Run cross-validation experiment with given splits.

    Args:
        df: Full dataframe.
        splits: List of (train_idx, val_idx) tuples.
        smiles_embeddings: SMILES embedding array.
        smiles_indices: Drug name -> index mapping.
        text_embeddings: PID -> text embedding mapping.
        eeg_data: PID -> EEG data mapping.
        config: Training configuration.
        device: Torch device.

    Returns:
        Dict of metric name -> list of fold values.
    """
    config = config or TRAINING_CONFIG
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_metrics = {
        "auc": [],
        "accuracy": [],
        "f1": [],
        "f1_tuned": [],
        "balanced_acc_tuned": [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"Training fold {fold_idx + 1}/{len(splits)}")

        # Create datasets
        train_dataset = create_dataset_from_indices(
            df, train_idx, smiles_embeddings, smiles_indices, text_embeddings, eeg_data
        )
        val_dataset = create_dataset_from_indices(
            df, val_idx, smiles_embeddings, smiles_indices, text_embeddings, eeg_data
        )

        # Log fold class balance
        train_labels = [train_dataset[i][-1].item() for i in range(len(train_dataset))]
        val_labels = [val_dataset[i][-1].item() for i in range(len(val_dataset))]
        logger.info(
            f"  Fold {fold_idx + 1}: train={len(train_labels)} "
            f"(pos={sum(train_labels)}), val={len(val_labels)} (pos={sum(val_labels)})"
        )

        # Train fold
        metrics = train_fold(train_dataset, val_dataset, config, device)

        # Store metrics
        for key in fold_metrics:
            if key in metrics:
                fold_metrics[key].append(metrics[key])

        logger.info(
            f"  Fold {fold_idx + 1} results: "
            f"AUC={metrics['auc']:.4f}, "
            f"BalAcc={metrics['balanced_acc_tuned']:.4f}, "
            f"F1={metrics['f1_tuned']:.4f}"
        )

    return fold_metrics


def compute_summary(fold_metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Compute summary statistics for fold metrics.

    Args:
        fold_metrics: Dict of metric name -> list of fold values.

    Returns:
        Dict of metric name -> summary stats.
    """
    summary = {}
    for key, values in fold_metrics.items():
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return summary
