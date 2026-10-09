"""Training for Experiment 16 (reduced-capacity quad-modal fusion).

Mirrors exp15's prediction-logging fold trainer but builds a reduced-capacity
QuadMLPv2 from a VARIANT dict. All the heavy lifting (epoch loop, ASM balancing,
per-patient + counterfactual-swap prediction logging) is reused verbatim from
exp7_all_modalities.training.
"""

import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import MLP_CONFIG
from .models import get_model

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.cv_splits import current_seed, max_epochs  # noqa: E402
from exp7_all_modalities.training import (  # noqa: E402
    _DropPidWrapper,
    _predict_with_smiles_override,
    _score_outer_fold,
    evaluate_mlp,
    train_epoch_mlp,
)
from shared.asm_balancing import (  # noqa: E402
    StratifiedASMBatchSampler,
    WeightedASMDataset,
    compute_asm_sample_weights,
)

logger = logging.getLogger("exp16")


def train_fold_with_predictions(
    train_dataset,
    val_dataset,
    variant: dict,
    device: torch.device,
    fold: int = 0,
    candidate_smiles: Dict[str, np.ndarray] = None,
    asm_balance_mode: str = "none",
    test_dataset=None,
    trace=None,
    fixed_epochs=None,
    refit_choice=None,
) -> Dict[str, Any]:
    """Train one fold of a reduced-capacity variant and return per-patient
    predictions plus ASM-swap predictions.

    ``val_dataset`` is the early-stopping set. With ``test_dataset`` (clean
    runs) the predictions, counterfactuals and metrics are reported on that
    untouched outer fold instead, with the restored weights and the
    early-stopping threshold. The reported dataset must have
    ``return_pid=True``."""
    report_dataset = test_dataset if test_dataset is not None else val_dataset
    if not getattr(report_dataset, "return_pid", False):
        raise ValueError("val_dataset/test_dataset must be built with return_pid=True for prediction logging.")

    config = MLP_CONFIG

    val_loader = None if val_dataset is None else DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False,
        drop_last=False, num_workers=0,
    )

    model = get_model(variant, device=device)
    logger.info(f"  [{variant['name']}] parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_labels_list = [train_dataset[i][5].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels_list)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    asm_weighted = (asm_balance_mode == "weighted")
    asm_stratified = (asm_balance_mode == "stratified_batch")
    train_asm_labels = list(train_dataset.asm_drugs)

    train_no_pid = _DropPidWrapper(train_dataset)
    if asm_weighted:
        weights = compute_asm_sample_weights(train_asm_labels)
        train_no_pid = WeightedASMDataset(train_no_pid, weights)

    if asm_stratified:
        batch_sampler = StratifiedASMBatchSampler(
            train_asm_labels, batch_size=config["batch_size"], seed=current_seed(0) + fold,
        )
        train_loader = DataLoader(train_no_pid, batch_sampler=batch_sampler, num_workers=0)
    else:
        train_loader = DataLoader(
            train_no_pid, batch_size=config["batch_size"], shuffle=True,
            drop_last=False, num_workers=0,
        )
    val_loader_for_loss = None if val_dataset is None else DataLoader(
        _DropPidWrapper(val_dataset), batch_size=config["batch_size"], shuffle=False,
        drop_last=False, num_workers=0,
    )

    best_val_auc = -1.0
    best_metrics: Dict[str, float] = {}
    best_state_dict = None
    patience_counter = 0

    for epoch in range(max_epochs(config["epochs"]) if fixed_epochs is None else fixed_epochs):
        train_epoch_mlp(
            model, train_loader, optimizer, criterion, device,
            asm_weighted=asm_weighted, class_weights=class_weights,
        )
        if fixed_epochs is not None:
            continue
        _, val_metrics = evaluate_mlp(model, val_loader_for_loss, criterion, device)
        if trace is not None:
            trace.setdefault("val_probs", []).append(np.asarray(val_metrics["y_prob"], dtype=float))
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_metrics = val_metrics.copy()
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
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
                test_dataset, batch_size=config["batch_size"], shuffle=False,
                drop_last=False, num_workers=0,
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
