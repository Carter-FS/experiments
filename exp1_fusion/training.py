"""Training loop with 5-fold stratified cross-validation."""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Subset

from .config import (
    CONFIG_1A, CONFIG_1B, CV_CONFIG, RESULTS_DIR,
    TEXT_DIM, SMILES_DIMS,
)
from .data_pipeline import get_full_dataset, load_csv_data
from shared.asm_balancing import WeightedASMDataset, compute_asm_sample_weights, weighted_cross_entropy
from shared.cv_splits import outer_splits, rethreshold, max_epochs
from shared.epoch_selection import run_outer_fold
from .models import ConcatMLPClassifier, SimplifiedFuseMoE


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def create_model(
    fusion_type: str,
    text_dim: int,
    smiles_dim: int,
    config: Dict,
) -> nn.Module:
    """Create a model based on fusion type."""
    if fusion_type == 'mlp':
        return ConcatMLPClassifier(
            text_dim=text_dim,
            smiles_dim=smiles_dim,
            hidden_dims=config['hidden_dims'],
            dropout_rates=config['dropout_rates'],
            num_classes=config['num_classes'],
        )
    elif fusion_type == 'fusemoe':
        return SimplifiedFuseMoE(
            text_dim=text_dim,
            smiles_dim=smiles_dim,
            hidden_dim=config['hidden_dim'],
            num_experts=config['num_experts'],
            top_k=config['top_k'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            num_classes=config['num_classes'],
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_aux_loss: bool = False,
    aux_loss_weight: float = 0.1,
    global_step: int = 0,
    asm_weighted: bool = False,
) -> Tuple[float, int]:
    """Train for one epoch.

    Returns:
        Tuple of (avg_loss, updated_global_step).
    """
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        text_emb = batch['text_emb'].to(device)
        smiles_emb = batch['smiles_emb'].to(device)
        labels = batch['label'].to(device)
        sample_weights = batch['asm_weight'].to(device) if asm_weighted else None

        if use_aux_loss:
            logits, aux_loss = model(text_emb, smiles_emb)
        else:
            logits = model(text_emb, smiles_emb)
            aux_loss = 0.0
        ce = (weighted_cross_entropy(logits, labels, sample_weights, class_weight=criterion.weight)
              if asm_weighted else criterion(logits, labels))
        loss = ce + aux_loss_weight * aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Temperature annealing
        if hasattr(model, 'update_temperature'):
            model.update_temperature(global_step)
        global_step += 1

        total_loss += loss.item()

    return total_loss / len(train_loader), global_step


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_aux_loss: bool = False,
) -> Dict[str, float]:
    """
    Evaluate the model.

    Returns:
        Dictionary with metrics: accuracy, auc, f1, balanced_acc_tuned, f1_tuned, optimal_threshold
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            text_emb = batch['text_emb'].to(device)
            smiles_emb = batch['smiles_emb'].to(device)
            labels = batch['label']

            if use_aux_loss:
                logits, _ = model(text_emb, smiles_emb)
            else:
                logits = model(text_emb, smiles_emb)

            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "y_prob": list(all_probs),
        "y_true": list(all_labels),
    }

    # Handle edge case where all predictions are same class
    if len(np.unique(all_labels)) > 1:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)

        # Threshold tuning using Youden's J statistic
        fpr, tpr, thresholds_roc = roc_curve(all_labels, all_probs)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        optimal_threshold = thresholds_roc[best_idx]
        tuned_preds = (np.array(all_probs) >= optimal_threshold).astype(int)
        metrics["balanced_acc_tuned"] = balanced_accuracy_score(all_labels, tuned_preds)
        metrics["f1_tuned"] = f1_score(all_labels, tuned_preds, zero_division=0)
        metrics["optimal_threshold"] = optimal_threshold
    else:
        metrics["auc"] = 0.5
        metrics["balanced_acc_tuned"] = 0.5
        metrics["f1_tuned"] = 0.0
        metrics["optimal_threshold"] = 0.5

    return metrics


def run_experiment(
    experiment_name: str,
    text_model: str,
    smiles_model: str,
    fusion_type: str,
    verbose: bool = True,
    prediction_logger=None,
    asm_balance_mode: str = "none",
    splitter: str = "legacy",
    inner_val: float = 0.0,
) -> Dict:
    """
    Run a single experiment with 5-fold cross-validation.

    Args:
        experiment_name: Name for the experiment
        text_model: 'clinicalbert' or 'pubmedbert'
        smiles_model: 'chemberta' or 'smilestrf'
        fusion_type: 'mlp' or 'fusemoe'
        verbose: Whether to print progress
        splitter: Outer splitter ('legacy' = outcome-only StratifiedKFold).
        inner_val: Inner early-stopping fraction (0 = legacy protocol: early
            stopping, LR scheduling and the threshold use the outer fold).

    Returns:
        Dictionary with results
    """
    device = get_device()
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"Text: {text_model}, SMILES: {smiles_model}, Fusion: {fusion_type}")
        print(f"Device: {device}")
        print(f"{'='*60}")

    # Get config
    config = CONFIG_1A if fusion_type == 'mlp' else CONFIG_1B
    use_aux_loss = fusion_type == 'fusemoe'

    # Load data
    dataset, outcomes, pids = get_full_dataset(text_model, smiles_model)
    smiles_dim = SMILES_DIMS[smiles_model]

    if verbose:
        print(f"Dataset size: {len(dataset)}")
        print(f"Class distribution: {np.bincount(outcomes)}")

    # Set up cross-validation
    splits = outer_splits(
        dataset.cohort_df, mode=splitter,
        n_splits=CV_CONFIG['n_splits'], seed=CV_CONFIG['random_state'],
    )

    # Results storage
    results = {
        'experiment': experiment_name,
        'text_model': text_model,
        'smiles_model': smiles_model,
        'fusion_type': fusion_type,
        'n_samples': len(dataset),
        'n_folds': CV_CONFIG['n_splits'],
        'fold_accuracy': [],
        'fold_auc': [],
        'fold_f1': [],
        'fold_f1_tuned': [],
        'fold_balanced_acc_tuned': [],
        'config': config,
    }

    def fit_fold(fit_idx, es_idx, test_idx=None, trace=None, fixed_epochs=None, refit_choice=None):
        """Train on ``fit_idx`` and return the reported fold's metrics.

        Legacy: early stopping, LR scheduling and metrics on ``es_idx`` (the
        outer fold). Inner split: select on ``es_idx``, score ``test_idx``
        once. Refit inner run: ``trace`` collects ``es_idx`` probabilities and
        the learning rate per epoch. Refit: ``es_idx`` is None; train exactly
        ``fixed_epochs`` epochs replaying the inner runs' median learning rate
        per epoch (``refit_choice.lr_schedule``) in place of ReduceLROnPlateau,
        then score ``test_idx`` unthresholded.
        """
        # Create dataloaders (ASM-weighting wraps the train subset so each
        # sample carries an inverse-sqrt weight, aligned with fit_idx order).
        train_subset = Subset(dataset, fit_idx)
        val_subset = None if es_idx is None else Subset(dataset, es_idx)

        asm_weighted = asm_balance_mode == "weighted"
        if asm_weighted:
            fold_weights = compute_asm_sample_weights([dataset.asm_drugs[i] for i in fit_idx])
            train_subset = WeightedASMDataset(train_subset, fold_weights)

        train_loader = DataLoader(
            train_subset,
            batch_size=config['batch_size'],
            shuffle=True,
        )
        val_loader = None if val_subset is None else DataLoader(
            val_subset,
            batch_size=config['batch_size'],
            shuffle=False,
        )
        test_loader = None
        if test_idx is not None:
            test_loader = DataLoader(
                Subset(dataset, test_idx),
                batch_size=config['batch_size'],
                shuffle=False,
            )
            if verbose and es_idx is not None:
                print(f"  Train: {len(fit_idx)}, Early-stop: {len(es_idx)}, Test: {len(test_idx)}")

        # Create model
        model = create_model(fusion_type, TEXT_DIM, smiles_dim, config)
        model.to(device)

        # Disable temperature annealing (Exp12 finding: no temp decay is optimal)
        if use_aux_loss and hasattr(model, 'fusion_layers'):
            for layer in model.fusion_layers:
                if hasattr(layer, 'fuse_moe'):
                    layer.fuse_moe.temperature_decay = 1.0

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5,
        )

        # Loss function with class weights
        class_counts = np.bincount(outcomes[fit_idx])
        class_weights = torch.FloatTensor([1.0 / c for c in class_counts])
        class_weights = class_weights / class_weights.sum()
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

        # Training loop with early stopping
        best_val_auc = 0.0
        best_es_metrics = {}
        patience_counter = 0
        best_model_state = None
        global_step = 0

        for epoch in range(max_epochs(config['epochs']) if fixed_epochs is None else fixed_epochs):
            if fixed_epochs is not None and refit_choice is not None and refit_choice.lr_schedule:
                for group in optimizer.param_groups:
                    group['lr'] = refit_choice.lr_schedule[epoch]
            if trace is not None:
                trace.setdefault('lr', []).append(optimizer.param_groups[0]['lr'])

            # Train
            train_loss, global_step = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                use_aux_loss=use_aux_loss,
                aux_loss_weight=config.get('aux_loss_weight', 0.1),
                global_step=global_step,
                asm_weighted=asm_weighted,
            )
            if fixed_epochs is not None:
                continue

            # Evaluate
            val_metrics = evaluate(model, val_loader, device, use_aux_loss=use_aux_loss)
            val_auc = val_metrics['auc']
            val_f1 = val_metrics['f1']
            if trace is not None:
                trace.setdefault('val_probs', []).append(np.asarray(val_metrics['y_prob'], dtype=float))

            scheduler.step(val_auc)

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_es_metrics = val_metrics
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config['patience']:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}: Loss={train_loss:.4f}, "
                      f"Val AUC={val_auc:.4f}, Val F1={val_f1:.4f}")

        if fixed_epochs is not None:
            return evaluate(model, test_loader, device, use_aux_loss=use_aux_loss)
        if trace is not None:
            return {}

        # Load best model and get final metrics
        model.load_state_dict(best_model_state)
        model.to(device)

        if test_loader is None:
            return evaluate(model, val_loader, device, use_aux_loss=use_aux_loss)
        # Clean protocol: score the outer fold once with the early-stopping-
        # best weights, at the threshold chosen on the early-stopping set.
        final_metrics = evaluate(model, test_loader, device, use_aux_loss=use_aux_loss)
        final_metrics = rethreshold(final_metrics, best_es_metrics.get('optimal_threshold', 0.5))
        final_metrics['es_auc'] = best_val_auc
        return final_metrics

    # Run cross-validation
    for fold, (train_idx, val_idx) in enumerate(splits):
        if verbose:
            print(f"\n--- Fold {fold + 1}/{CV_CONFIG['n_splits']} ---")

        # Clean runs select (early stopping, LR scheduling, threshold) on
        # inner data of the outer training fold and score the outer fold once.
        final_metrics, selection = run_outer_fold(
            outcomes, train_idx, val_idx, fold, inner_val,
            make_datasets=lambda a, b: (a, b),
            train=lambda a, b, test_dataset=None, **kw: fit_fold(a, b, test_dataset, **kw),
        )

        if prediction_logger is not None:
            prediction_logger.log_fold(
                fold=fold,
                pids=[pids[i] for i in val_idx],
                y_true=final_metrics['y_true'],
                y_prob=final_metrics['y_prob'],
                threshold=final_metrics.get('optimal_threshold'),
                selection=selection,
            )

        results['fold_accuracy'].append(final_metrics['accuracy'])
        results['fold_auc'].append(final_metrics['auc'])
        results['fold_f1'].append(final_metrics['f1'])
        results['fold_f1_tuned'].append(final_metrics['f1_tuned'])
        results['fold_balanced_acc_tuned'].append(final_metrics['balanced_acc_tuned'])

        if verbose:
            print(f"  Fold {fold + 1} Results: AUC={final_metrics['auc']:.4f}, "
                  f"BalAcc_tuned={final_metrics['balanced_acc_tuned']:.4f}, "
                  f"F1_tuned={final_metrics['f1_tuned']:.4f} (thresh={final_metrics['optimal_threshold']:.3f})")

    # Aggregate results
    results['accuracy'] = {
        'mean': float(np.mean(results['fold_accuracy'])),
        'std': float(np.std(results['fold_accuracy'])),
        'min': float(np.min(results['fold_accuracy'])),
        'max': float(np.max(results['fold_accuracy'])),
        'per_fold': results['fold_accuracy'],
    }
    results['auc'] = {
        'mean': float(np.mean(results['fold_auc'])),
        'std': float(np.std(results['fold_auc'])),
        'min': float(np.min(results['fold_auc'])),
        'max': float(np.max(results['fold_auc'])),
        'per_fold': results['fold_auc'],
    }
    results['f1'] = {
        'mean': float(np.mean(results['fold_f1'])),
        'std': float(np.std(results['fold_f1'])),
        'min': float(np.min(results['fold_f1'])),
        'max': float(np.max(results['fold_f1'])),
        'per_fold': results['fold_f1'],
    }
    results['f1_tuned'] = {
        'mean': float(np.mean(results['fold_f1_tuned'])),
        'std': float(np.std(results['fold_f1_tuned'])),
        'min': float(np.min(results['fold_f1_tuned'])),
        'max': float(np.max(results['fold_f1_tuned'])),
        'per_fold': results['fold_f1_tuned'],
    }
    results['balanced_acc_tuned'] = {
        'mean': float(np.mean(results['fold_balanced_acc_tuned'])),
        'std': float(np.std(results['fold_balanced_acc_tuned'])),
        'min': float(np.min(results['fold_balanced_acc_tuned'])),
        'max': float(np.max(results['fold_balanced_acc_tuned'])),
        'per_fold': results['fold_balanced_acc_tuned'],
    }

    # Clean up fold lists (now in nested dict)
    del results['fold_accuracy']
    del results['fold_auc']
    del results['fold_f1']
    del results['fold_f1_tuned']
    del results['fold_balanced_acc_tuned']

    if verbose:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS: {experiment_name}")
        print(f"{'='*60}")
        print(f"AUC:              {results['auc']['mean']:.4f} +/- {results['auc']['std']:.4f} (min={results['auc']['min']:.4f}, max={results['auc']['max']:.4f})")
        print(f"Balanced Acc:     {results['balanced_acc_tuned']['mean']:.4f} +/- {results['balanced_acc_tuned']['std']:.4f} (min={results['balanced_acc_tuned']['min']:.4f}, max={results['balanced_acc_tuned']['max']:.4f})")
        print(f"F1 Tuned:         {results['f1_tuned']['mean']:.4f} +/- {results['f1_tuned']['std']:.4f} (min={results['f1_tuned']['min']:.4f}, max={results['f1_tuned']['max']:.4f})")

    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()

    return results


def save_results(results: Dict, filename: Optional[str] = None) -> str:
    """Save results to JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if filename is None:
        filename = f"{results['experiment']}.json"

    filepath = os.path.join(RESULTS_DIR, filename)

    # Convert config to serializable format
    results_copy = results.copy()
    results_copy['config'] = {k: v for k, v in results['config'].items()}

    with open(filepath, 'w') as f:
        json.dump(results_copy, f, indent=2)

    print(f"Results saved to: {filepath}")
    return filepath


if __name__ == '__main__':
    # Test training with a single experiment
    print("Testing training pipeline...")

    results = run_experiment(
        experiment_name='test_exp1a_pubmedbert_chemberta',
        text_model='pubmedbert',
        smiles_model='chemberta',
        fusion_type='mlp',
        verbose=True,
    )

    save_results(results, 'test_results.json')
    print("\nTraining test complete!")
