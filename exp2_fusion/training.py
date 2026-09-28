"""Training utilities for Experiment 2: EEG + SMILES fusion."""

import copy
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# Add parent directory for exp8_stratification import
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp8_stratification.stratified_cv import get_multilabel_splits, get_outcome_only_splits

from .config import BATCH_SIZE_BY_ENCODER, CHUNK_SIZE_BY_ENCODER, EMBED_DIM_BY_ENCODER, MODEL_CONFIG, TRAIN_CONFIG
from .data_pipeline import EEGSMILESDataset, create_datasets, get_max_channels, prepare_data
from .eeg_pipeline import add_stratification_columns
from shared.asm_balancing import WeightedASMDataset, compute_asm_sample_weights, weighted_cross_entropy
from shared.cv_splits import outer_splits, rethreshold, current_seed
from shared.epoch_selection import run_outer_fold
from .models.fusion import get_fusion_model

logger = logging.getLogger("exp2")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_moe: bool = False,
    global_step: int = 0,
    asm_weighted: bool = False,
) -> Tuple[float, int]:
    """Train for one epoch.

    Args:
        model: Model to train.
        dataloader: Training data loader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to use.
        is_moe: Whether model returns auxiliary loss.
        global_step: Current global step for temperature annealing.

    Returns:
        Tuple of (avg_loss, updated_global_step).
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        if asm_weighted:
            eeg_windows, padding_mask, smiles_emb, labels, sample_weights = batch
            sample_weights = sample_weights.to(device)
        else:
            eeg_windows, padding_mask, smiles_emb, labels = batch
            sample_weights = None
        eeg_windows = eeg_windows.to(device)
        padding_mask = padding_mask.to(device)
        smiles_emb = smiles_emb.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if is_moe:
            logits, aux_loss = model(eeg_windows, padding_mask, smiles_emb)
        else:
            logits = model(eeg_windows, padding_mask, smiles_emb)
            aux_loss = 0.0
        ce = (weighted_cross_entropy(logits, labels, sample_weights, class_weight=criterion.weight)
              if asm_weighted else criterion(logits, labels))
        loss = ce + aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Temperature annealing
        if hasattr(model, 'update_temperature'):
            model.update_temperature(global_step)
        global_step += 1

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches, global_step


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_moe: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model.

    Args:
        model: Model to evaluate.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device to use.
        is_moe: Whether model returns auxiliary loss.

    Returns:
        Tuple of (loss, metrics dict).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            eeg_windows, padding_mask, smiles_emb, labels = batch
            eeg_windows = eeg_windows.to(device)
            padding_mask = padding_mask.to(device)
            smiles_emb = smiles_emb.to(device)
            labels = labels.to(device)

            if is_moe:
                logits, aux_loss = model(eeg_windows, padding_mask, smiles_emb)
                loss = criterion(logits, labels) + aux_loss
            else:
                logits = model(eeg_windows, padding_mask, smiles_emb)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "y_prob": all_probs.tolist(),
        "y_true": all_labels.tolist(),
    }

    # Compute AUC and threshold-tuned metrics if we have both classes
    if len(np.unique(all_labels)) > 1:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)

        # Threshold tuning using Youden's J statistic
        fpr, tpr, thresholds_roc = roc_curve(all_labels, all_probs)
        youden_j = tpr - fpr
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
    train_dataset: EEGSMILESDataset,
    val_dataset: EEGSMILESDataset,
    fusion_type: str,
    eeg_encoder_type: str,
    smiles_embed_dim: int,
    n_eeg_channels: int,
    device: torch.device,
    config: Dict = TRAIN_CONFIG,
    model_config: Dict = MODEL_CONFIG,
    asm_balance_mode: str = "none",
    test_dataset: Optional[EEGSMILESDataset] = None,
    trace: Optional[dict] = None,
    fixed_epochs: Optional[int] = None,
    refit_choice=None,
) -> Tuple[Dict[str, float], nn.Module]:
    """Train model for one fold.

    Refit protocol (analysis plan B.2): an inner run passes ``trace`` to
    collect the early-stopping set's probabilities and the learning rate per
    epoch. The refit passes ``fixed_epochs`` and ``val_dataset=None``; it
    replays the inner runs' median learning rate per epoch
    (``refit_choice.lr_schedule``) in place of ReduceLROnPlateau and returns the
    outer test metrics of the final weights, unthresholded.

    Args:
        train_dataset: Training dataset.
        val_dataset: Early-stopping dataset (the outer fold in legacy runs,
            the inner split in clean runs). It also drives ReduceLROnPlateau.
        fusion_type: Type of fusion model.
        eeg_encoder_type: Type of EEG encoder.
        smiles_embed_dim: SMILES embedding dimension.
        n_eeg_channels: Number of EEG channels.
        device: Device to use.
        config: Training configuration.
        model_config: Model configuration.
        test_dataset: Clean runs only: the untouched outer fold, scored once
            with the best early-stopping weights at the early-stopping
            threshold. None keeps the legacy behaviour (report the best
            epoch's metrics on ``val_dataset``).

    Returns:
        Tuple of (metrics dict for the reported fold, trained model).
    """
    # ASM-balancing: wrap the training set with inverse-sqrt sample weights.
    asm_weighted = asm_balance_mode == "weighted"
    if asm_weighted:
        asm_list = [train_dataset.asm_drugs[pid] for pid in train_dataset.patient_ids]
        train_dataset = WeightedASMDataset(train_dataset, compute_asm_sample_weights(asm_list))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = None if val_dataset is None else DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Create model
    is_moe = fusion_type == "fusemoe"
    window_chunk_size = CHUNK_SIZE_BY_ENCODER.get(eeg_encoder_type, 16)
    # Only encoders listed in EMBED_DIM_BY_ENCODER override the model default.
    embed_kwargs = (
        {"eeg_embed_dim": EMBED_DIM_BY_ENCODER[eeg_encoder_type]}
        if eeg_encoder_type in EMBED_DIM_BY_ENCODER else {}
    )
    model = get_fusion_model(
        fusion_type=fusion_type,
        eeg_encoder_type=eeg_encoder_type,
        n_eeg_channels=n_eeg_channels,
        smiles_embed_dim=smiles_embed_dim,
        hidden_dim=model_config["hidden_dim"],
        num_classes=model_config["num_classes"],
        num_experts=model_config.get("num_experts", 4),
        top_k=model_config.get("top_k", 2),
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        aux_loss_weight=model_config.get("aux_loss_weight", 0.1),
        window_chunk_size=window_chunk_size,
        **embed_kwargs,
    )
    model = model.to(device)

    # Disable temperature annealing for FuseMoE (Exp12 finding)
    if is_moe and hasattr(model, 'fuse_moe'):
        model.fuse_moe.temperature_decay = 1.0

    # Class weighting for imbalanced datasets
    train_labels = [train_dataset[i][3].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.debug(f"  Class weights: {class_weights.cpu().numpy()}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Training loop
    best_auc = 0.0
    best_metrics = {}
    best_state = None
    patience_counter = 0
    global_step = 0

    for epoch in range(config["epochs"] if fixed_epochs is None else fixed_epochs):
        if fixed_epochs is not None and refit_choice is not None and refit_choice.lr_schedule:
            for group in optimizer.param_groups:
                group["lr"] = refit_choice.lr_schedule[epoch]
        if trace is not None:
            trace.setdefault("lr", []).append(optimizer.param_groups[0]["lr"])
        train_loss, global_step = train_epoch(model, train_loader, optimizer, criterion, device, is_moe, global_step, asm_weighted=asm_weighted)
        if fixed_epochs is not None:
            continue
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, is_moe)
        if trace is not None:
            trace.setdefault("val_probs", []).append(np.asarray(val_metrics["y_prob"], dtype=float))

        scheduler.step(val_metrics["auc"])

        # Log progress at intervals
        if epoch % 20 == 0 or epoch == config["epochs"] - 1:
            logger.debug(
                f"Epoch {epoch+1}/{config['epochs']}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_auc={val_metrics['auc']:.3f}"
            )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_metrics = val_metrics.copy()
            if test_dataset is not None:
                best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            logger.debug(f"Early stopping at epoch {epoch+1} (patience={config['patience']})")
            break

    if fixed_epochs is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True,
        )
        return evaluate(model, test_loader, criterion, device, is_moe)[1], model

    if test_dataset is None:
        return best_metrics, model

    # Clean protocol: score the outer fold once with the early-stopping-best
    # weights, at the threshold chosen on the early-stopping set.
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    _, test_metrics = evaluate(model, test_loader, criterion, device, is_moe)
    test_metrics = rethreshold(test_metrics, best_metrics.get("optimal_threshold", 0.5))
    test_metrics["es_auc"] = best_auc
    return test_metrics, model


def run_cross_validation(
    eeg_data: Dict,
    smiles_embeddings: np.ndarray,
    smiles_indices: Dict,
    df,
    fusion_type: str,
    eeg_encoder_type: str,
    smiles_model: str,
    device: torch.device,
    config: Dict = TRAIN_CONFIG,
    verbose: bool = True,
    use_multilabel_stratification: bool = True,
    prediction_logger=None,
    asm_balance_mode: str = "none",
    splitter: str = "legacy",
    inner_val: float = 0.0,
) -> Dict:
    """Run k-fold cross validation.

    Args:
        eeg_data: Dict mapping patient ID to (windows, padding_mask).
        smiles_embeddings: SMILES embeddings array.
        smiles_indices: Dict mapping ASM name to embedding index.
        df: DataFrame with patient info.
        fusion_type: Type of fusion model.
        eeg_encoder_type: Type of EEG encoder.
        smiles_model: Name of SMILES model.
        device: Device to use.
        config: Training configuration.
        verbose: Whether to print progress.
        use_multilabel_stratification: Whether to use multi-label stratification
            on outcome + focal + sex (reduces fold variance by 5-8x). Legacy
            splitter only.
        splitter: Outer splitter ('legacy' = this experiment's original
            splits, 'multilabel' = shared outcome/focal/sex splitter).
        inner_val: Inner early-stopping fraction (0 = legacy protocol).

    Returns:
        Results dictionary with metrics.
    """
    # Set seed (the repeated-CV seed when one is active)
    np.random.seed(current_seed(config["seed"]))
    torch.manual_seed(current_seed(config["seed"]))

    # Adjust batch size based on encoder type
    config = config.copy()
    config["batch_size"] = BATCH_SIZE_BY_ENCODER.get(eeg_encoder_type, config["batch_size"])

    # Get labels and determine dimensions
    n_eeg_channels = get_max_channels(eeg_data)  # Use max channels across all data
    smiles_embed_dim = smiles_embeddings.shape[1]

    # Cross-validation with stratification
    strat_type = "multilabel" if use_multilabel_stratification else "outcome-only"
    if splitter != "legacy":
        strat_type = f"shared {splitter}"
    if verbose:
        logger.info(f"  Using {strat_type} stratification (inner_val={inner_val})")

    if splitter != "legacy":
        # The EEG cohort frame has no focal/sex columns, so join them on for
        # the splitter only (datasets keep using df unchanged).
        splits = outer_splits(
            add_stratification_columns(df), mode=splitter,
            n_splits=config["n_folds"], seed=config["seed"],
        )
    elif use_multilabel_stratification:
        # Use multi-label stratification on outcome + focal + sex. df has no
        # focal/sex columns, so these folds are effectively outcome-only; kept
        # as-is so archived runs reproduce.
        splits = list(get_multilabel_splits(
            df,
            stratify_cols=["outcome", "focal", "sex"],
            n_splits=config["n_folds"],
            shuffle=True,
            random_state=config["seed"],
        ))
    else:
        # Outcome-only stratification (baseline)
        splits = list(get_outcome_only_splits(
            df,
            n_splits=config["n_folds"],
            shuffle=True,
            random_state=config["seed"],
        ))

    fold_metrics = {"accuracy": [], "auc": [], "f1": [], "f1_tuned": [], "balanced_acc_tuned": []}
    outcomes = df["outcome"].values

    for fold, (train_idx, val_idx) in enumerate(splits):
        if verbose:
            logger.info(f"  Fold {fold + 1}/{config['n_folds']}")

        # Clean runs select (early stopping, LR scheduling, threshold) on inner
        # data of the outer training fold and score the outer fold once;
        # legacy runs early-stop on the outer fold itself.
        try:
            metrics, selection = run_outer_fold(
                outcomes, train_idx, val_idx, fold, inner_val,
                # Pass max_channels for consistent padding.
                make_datasets=lambda a, b: create_datasets(
                    eeg_data, smiles_embeddings, smiles_indices, df, a, b,
                    max_channels=n_eeg_channels,
                ),
                train=lambda tr, va, **kw: train_fold(
                    tr, va,
                    fusion_type=fusion_type,
                    eeg_encoder_type=eeg_encoder_type,
                    smiles_embed_dim=smiles_embed_dim,
                    n_eeg_channels=n_eeg_channels,
                    device=device,
                    config=config,
                    asm_balance_mode=asm_balance_mode,
                    **kw,
                )[0],
                log=logger.info if verbose else None,
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

            if verbose:
                logger.info(
                    f"    Fold {fold + 1} results: AUC={metrics['auc']:.4f}, "
                    f"BalAcc_tuned={metrics['balanced_acc_tuned']:.4f}, "
                    f"F1_tuned={metrics['f1_tuned']:.4f} (thresh={metrics['optimal_threshold']:.3f})"
                )

        except Exception as e:
            logger.error(f"    Fold {fold + 1} failed: {type(e).__name__}: {e}")
            raise

    # Aggregate results
    results = {
        "experiment": f"exp2_{eeg_encoder_type}_{smiles_model}_{fusion_type}",
        "eeg_encoder": eeg_encoder_type,
        "smiles_model": smiles_model,
        "fusion_type": fusion_type,
        "n_samples": len(df),
        "n_folds": config["n_folds"],
        "config": {**config, **MODEL_CONFIG},
    }

    for metric in fold_metrics:
        values = np.array(fold_metrics[metric])
        results[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "per_fold": [float(v) for v in values],
        }

    if verbose:
        logger.info("Cross-validation complete:")
        logger.info(f"  AUC: {results['auc']['mean']:.4f} +/- {results['auc']['std']:.4f} (min={results['auc']['min']:.4f}, max={results['auc']['max']:.4f})")
        logger.info(f"  Balanced Acc: {results['balanced_acc_tuned']['mean']:.4f} +/- {results['balanced_acc_tuned']['std']:.4f} (min={results['balanced_acc_tuned']['min']:.4f}, max={results['balanced_acc_tuned']['max']:.4f})")
        logger.info(f"  F1 Tuned: {results['f1_tuned']['mean']:.4f} +/- {results['f1_tuned']['std']:.4f} (min={results['f1_tuned']['min']:.4f}, max={results['f1_tuned']['max']:.4f})")

    return results


if __name__ == "__main__":
    # Quick test with logging
    from .utils.logging_utils import setup_logging, log_environment_info

    logger = setup_logging("exp2_training_test", log_dir="logs")
    log_environment_info(logger)

    logger.info("Testing training pipeline...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Prepare data
    eeg_data, smiles_embeddings, smiles_indices, df = prepare_data(
        smiles_model="chemberta",
        cache_eeg=True,
    )

    # Run single experiment with SimpleCNN (faster for testing)
    results = run_cross_validation(
        eeg_data, smiles_embeddings, smiles_indices, df,
        fusion_type="mlp",
        eeg_encoder_type="simplecnn",
        smiles_model="chemberta",
        device=device,
        verbose=True,
    )

    logger.info(f"Results:")
    logger.info(f"  AUC: {results['auc']['mean']:.4f} +/- {results['auc']['std']:.4f}")
    logger.info(f"  Balanced Acc: {results['balanced_acc_tuned']['mean']:.4f} +/- {results['balanced_acc_tuned']['std']:.4f}")
    logger.info(f"  F1 Tuned: {results['f1_tuned']['mean']:.4f} +/- {results['f1_tuned']['std']:.4f}")
