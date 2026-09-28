"""Training utilities for Experiment 3: LLM + EEG + SMILES triple fusion."""

import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from .config import CONFIG_3A, CONFIG_3B, CV_CONFIG, EEG_ENCODER_CONFIG, SMILES_DIMS
from .data_pipeline import (
    TripleModalityDataset,
    create_datasets,
    get_max_channels,
    prepare_data,
)
from .models import TripleModalityMLP, TripleModalityFuseMoE
from exp2_fusion.eeg_pipeline import add_stratification_columns
from shared.cv_splits import outer_splits, rethreshold, current_seed
from shared.epoch_selection import run_outer_fold

logger = logging.getLogger("exp3")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_moe: bool = False,
    global_step: int = 0,
    asm_weighted: bool = False,
    class_weights: torch.Tensor = None,
) -> Tuple[float, int]:
    """Train for one epoch.

    Returns:
        Tuple of (avg_loss, updated_global_step).
    """
    from shared.asm_balancing import weighted_cross_entropy
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        if asm_weighted:
            text_emb, eeg_windows, padding_mask, smiles_emb, labels, sample_weights = batch
            sample_weights = sample_weights.to(device)
        else:
            text_emb, eeg_windows, padding_mask, smiles_emb, labels = batch
            sample_weights = None
        text_emb = text_emb.to(device)
        eeg_windows = eeg_windows.to(device)
        padding_mask = padding_mask.to(device)
        smiles_emb = smiles_emb.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if is_moe:
            logits, aux_loss = model(text_emb, eeg_windows, padding_mask, smiles_emb)
            if asm_weighted:
                loss = weighted_cross_entropy(logits, labels, sample_weights, class_weight=class_weights) + aux_loss
            else:
                loss = criterion(logits, labels) + aux_loss
        else:
            logits = model(text_emb, eeg_windows, padding_mask, smiles_emb)
            if asm_weighted:
                loss = weighted_cross_entropy(logits, labels, sample_weights, class_weight=class_weights)
            else:
                loss = criterion(logits, labels)

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
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            text_emb, eeg_windows, padding_mask, smiles_emb, labels = batch
            text_emb = text_emb.to(device)
            eeg_windows = eeg_windows.to(device)
            padding_mask = padding_mask.to(device)
            smiles_emb = smiles_emb.to(device)
            labels = labels.to(device)

            if is_moe:
                logits, aux_loss = model(text_emb, eeg_windows, padding_mask, smiles_emb)
                loss = criterion(logits, labels) + aux_loss
            else:
                logits = model(text_emb, eeg_windows, padding_mask, smiles_emb)
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


def get_model(
    fusion_type: str,
    text_dim: int,
    smiles_dim: int,
    device: torch.device,
    eeg_encoder_type: Optional[str] = None,
) -> nn.Module:
    """Create fusion model based on type.

    ``eeg_encoder_type`` overrides EEG_ENCODER_CONFIG["encoder_type"] (None
    keeps the default SimpleCNN); all other EEG settings are shared.
    """
    eeg_encoder_type = eeg_encoder_type or EEG_ENCODER_CONFIG["encoder_type"]
    if fusion_type == "mlp":
        config = CONFIG_3A
        model = TripleModalityMLP(
            text_dim=text_dim,
            smiles_dim=smiles_dim,
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
            eeg_encoder_type=eeg_encoder_type,
            n_eeg_channels=EEG_ENCODER_CONFIG["n_channels"],
            n_eeg_times=EEG_ENCODER_CONFIG["n_times"],
            eeg_embed_dim=EEG_ENCODER_CONFIG["embed_dim"],
            num_heads=EEG_ENCODER_CONFIG["num_heads"],
            num_layers=EEG_ENCODER_CONFIG["num_layers"],
            max_windows=EEG_ENCODER_CONFIG["max_windows"],
            window_chunk_size=EEG_ENCODER_CONFIG["window_chunk_size"],
        )
    elif fusion_type == "fusemoe":
        config = CONFIG_3B
        model = TripleModalityFuseMoE(
            text_dim=text_dim,
            smiles_dim=smiles_dim,
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
            num_experts=config["num_experts"],
            top_k=config["top_k"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
            aux_loss_weight=config["aux_loss_weight"],
            eeg_encoder_type=eeg_encoder_type,
            n_eeg_channels=EEG_ENCODER_CONFIG["n_channels"],
            n_eeg_times=EEG_ENCODER_CONFIG["n_times"],
            eeg_embed_dim=EEG_ENCODER_CONFIG["embed_dim"],
            num_eeg_layers=EEG_ENCODER_CONFIG["num_layers"],
            max_windows=EEG_ENCODER_CONFIG["max_windows"],
            window_chunk_size=EEG_ENCODER_CONFIG["window_chunk_size"],
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")

    return model.to(device)


def train_fold(
    train_dataset: TripleModalityDataset,
    val_dataset: TripleModalityDataset,
    fusion_type: str,
    text_dim: int,
    smiles_dim: int,
    device: torch.device,
    fold: int,
    asm_balance_mode: str = "none",
    test_dataset: Optional[TripleModalityDataset] = None,
    trace=None,
    fixed_epochs=None,
    refit_choice=None,
    eeg_encoder_type: Optional[str] = None,
) -> Dict[str, float]:
    """Train and evaluate a single fold.

    ``val_dataset`` is the early-stopping set (the outer fold in legacy runs,
    the inner split in clean runs). ``test_dataset`` is given in clean runs
    only: the untouched outer fold, scored once with the best early-stopping
    weights at the early-stopping threshold. None keeps the legacy behaviour
    (report the best epoch's metrics on ``val_dataset``).
    """
    from shared.asm_balancing import (
        WeightedASMDataset,
        StratifiedASMBatchSampler,
        compute_asm_sample_weights,
    )
    config = CONFIG_3A if fusion_type == "mlp" else CONFIG_3B
    is_moe = fusion_type == "fusemoe"

    asm_weighted = (asm_balance_mode == "weighted")
    asm_stratified = (asm_balance_mode == "stratified_batch")
    train_asm_labels = [train_dataset.asm_drugs[pid] for pid in train_dataset.patient_ids] \
        if hasattr(train_dataset, "patient_ids") and isinstance(train_dataset.asm_drugs, dict) \
        else list(train_dataset.asm_drugs)

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
    model = get_model(fusion_type, text_dim, smiles_dim, device, eeg_encoder_type=eeg_encoder_type)
    # If we don't have asm_weighted/stratified, the train fold's label
    # extraction below uses train_dataset[i][4]; the WeightedASMDataset
    # wrapper preserves index 4 (label) while appending the weight at
    # index 5, so the existing class_weights computation still works.
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {n_params:,}")

    # Calculate class weights from training data (inverse frequency)
    train_labels = [train_dataset[i][4].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.info(f"  Class weights: {class_weights.cpu().numpy()}")

    # Optimizer and criterion
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Refit protocol: FuseMoE anneals its gating temperature per optimiser
    # step, and the refit takes more steps per epoch than the inner runs, so
    # rescale its step count to the inner runs' (analysis plan B.2).
    if trace is not None:
        trace["steps_per_epoch"] = len(train_loader)
    if (fixed_epochs is not None and refit_choice is not None and refit_choice.steps_per_epoch
            and hasattr(model, "update_temperature")):
        scale = refit_choice.steps_per_epoch / len(train_loader)
        update = model.update_temperature
        model.update_temperature = lambda step: update(int(step * scale))

    # Training loop
    best_val_auc = 0.0
    best_metrics = {}
    best_state = None
    best_step = 0
    patience_counter = 0
    global_step = 0

    for epoch in range(config["epochs"] if fixed_epochs is None else fixed_epochs):
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device, is_moe, global_step,
            asm_weighted=asm_weighted, class_weights=class_weights,
        )
        if fixed_epochs is not None:
            continue
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, is_moe)
        if trace is not None:
            trace.setdefault("val_probs", []).append(np.asarray(val_metrics["y_prob"], dtype=float))

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_metrics = val_metrics.copy()
            if test_dataset is not None:
                best_state = copy.deepcopy(model.state_dict())
                best_step = global_step
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
        return evaluate(model, test_loader, criterion, device, is_moe)[1]

    if test_dataset is None:
        return best_metrics

    # Clean protocol: score the outer fold once with the early-stopping-best
    # weights, at the threshold chosen on the early-stopping set.
    if best_state is not None:
        model.load_state_dict(best_state)
        # FuseMoE's annealed gating temperature is not in the state_dict: set
        # it back to its value at the best epoch (last update used step - 1).
        if hasattr(model, "update_temperature") and best_step > 0:
            model.update_temperature(best_step - 1)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    _, test_metrics = evaluate(model, test_loader, criterion, device, is_moe)
    test_metrics = rethreshold(test_metrics, best_metrics.get("optimal_threshold", 0.5))
    test_metrics["es_auc"] = best_val_auc
    return test_metrics


def run_cross_validation(
    text_model: str,
    smiles_model: str,
    fusion_type: str,
    device: torch.device = None,
    prediction_logger=None,
    asm_balance_mode: str = "none",
    eeg_encoder_type: Optional[str] = None,
    splitter: str = "legacy",
    inner_val: float = 0.0,
) -> Dict[str, List[float]]:
    """Run 5-fold cross-validation for a specific configuration.

    Args:
        eeg_encoder_type: EEG encoder override (None = EEG_ENCODER_CONFIG).
        splitter: Outer splitter ('legacy' = outcome-only StratifiedKFold).
        inner_val: Inner early-stopping fraction (0 = legacy protocol).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        f"Running CV: text={text_model}, smiles={smiles_model}, fusion={fusion_type}, "
        f"eeg={eeg_encoder_type or EEG_ENCODER_CONFIG['encoder_type']}"
    )

    # Prepare data
    text_emb, eeg_data, smiles_emb, smiles_idx, df = prepare_data(
        text_model=text_model,
        smiles_model=smiles_model,
        cache_eeg=True,
    )

    text_dim = 768  # All text models use 768
    smiles_dim = SMILES_DIMS[smiles_model]
    max_channels = get_max_channels(eeg_data)

    # Get outcomes for stratified split
    outcomes = df["outcome"].values

    # Cross-validation. The EEG cohort frame has no focal/sex columns, so the
    # multilabel splitter gets them joined on (datasets keep using df).
    split_df = df if splitter == "legacy" else add_stratification_columns(df)
    splits = outer_splits(
        split_df, mode=splitter, n_splits=CV_CONFIG["n_splits"], seed=CV_CONFIG["random_state"],
    )

    fold_metrics = {"auc": [], "accuracy": [], "f1": [], "f1_tuned": [], "balanced_acc_tuned": []}

    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"Fold {fold + 1}/{CV_CONFIG['n_splits']}")

        # Create datasets. Clean runs select on inner data (inner split, or
        # inner folds + refit) and score the outer fold once.
        metrics, selection = run_outer_fold(
            outcomes, train_idx, val_idx, fold, inner_val,
            make_datasets=lambda a, b: create_datasets(
                text_emb, eeg_data, smiles_emb, smiles_idx, df, a, b, max_channels,
            ),
            train=lambda tr, va, **kw: train_fold(
                tr, va, fusion_type, text_dim, smiles_dim, device, fold,
                asm_balance_mode=asm_balance_mode, eeg_encoder_type=eeg_encoder_type, **kw,
            ),
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

    # Test training on a single config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = run_cross_validation(
        text_model="clinicalbert",
        smiles_model="chemberta",
        fusion_type="mlp",
        device=device,
    )
