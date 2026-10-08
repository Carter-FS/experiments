"""Optuna objective functions for Experiment 14: HP tuning of top 3 models.

Each objective function wraps an existing model's training loop, sampling
hyperparameters from Optuna and reporting per-fold AUC for pruning.

Data is cached at module level to avoid reloading across trials.
"""

import logging

import numpy as np
import optuna
import torch
import torch.nn as nn
from exp2_fusion.config import N_CHANNELS
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .config import CV_CONFIG, SEARCH_SPACES, TRAINING_FIXED

logger = logging.getLogger(__name__)

# Module-level data cache - loaded once per process, reused across trials
_data_cache = {}


def sample_params(trial, search_space):
    """Sample hyperparameters from a search space definition.

    Args:
        trial: Optuna trial object.
        search_space: Dict mapping param names to (type, *args) tuples.

    Returns:
        Dict of sampled hyperparameter values.
    """
    params = {}
    for name, spec in search_space.items():
        if spec[0] == "float":
            _, low, high, log = spec
            params[name] = trial.suggest_float(name, low, high, log=log)
        elif spec[0] == "categorical":
            _, choices = spec
            params[name] = trial.suggest_categorical(name, choices)
    return params


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

def _get_exp7a_data():
    """Load and cache quad modality data (exp7a / exp11)."""
    if "exp7a" not in _data_cache:
        from exp7_all_modalities.data_pipeline import prepare_quad_modality_data

        logger.info("Loading quad modality data (first time)...")
        _data_cache["exp7a"] = prepare_quad_modality_data(
            "clinicalbert", "chemberta"
        )
    return _data_cache["exp7a"]


def _get_exp12_data():
    """Load and cache triple modality data (exp12 / exp3b)."""
    if "exp12" not in _data_cache:
        from exp3_fusion.data_pipeline import prepare_data

        logger.info("Loading triple modality data (first time)...")
        _data_cache["exp12"] = prepare_data(
            text_model="clinicalbert",
            smiles_model="chemberta"
        )
    return _data_cache["exp12"]


# ---------------------------------------------------------------------------
# Objective 1: Exp7a QuadFusionMLP (AUC 0.798)
# ---------------------------------------------------------------------------

def objective_exp7a_mlp(trial: optuna.Trial) -> float:
    """Optuna objective for Exp7a QuadFusionMLP.

    Tunes: learning_rate, weight_decay, dropout, hidden_dim, batch_size.
    Fixed: EEG2Vec encoder, ClinicalBERT text, ChemBERTa SMILES.
    """
    from exp7_all_modalities.config import (
        CLINICAL_DIM,
        EEG_ENCODER_CONFIG,
        SMILES_DIM,
        TEXT_DIM,
    )
    from exp7_all_modalities.data_pipeline import create_quad_modality_datasets
    from exp7_all_modalities.models import QuadFusionMLP
    from exp7_all_modalities.training import evaluate_mlp, train_epoch_mlp

    params = sample_params(trial, SEARCH_SPACES["exp7a_mlp"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load cached data
    df, smiles_emb, smiles_idx, text_emb, eeg_data = _get_exp7a_data()
    outcomes = df["outcome"].values

    kfold = StratifiedKFold(**CV_CONFIG)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(np.zeros(len(outcomes)), outcomes)
    ):
        train_ds, val_ds, _ = create_quad_modality_datasets(
            df, smiles_emb, smiles_idx, text_emb, eeg_data, train_idx, val_idx
        )

        model = QuadFusionMLP(
            clinical_dim=CLINICAL_DIM,
            text_dim=TEXT_DIM,
            smiles_dim=SMILES_DIM,
            hidden_dim=params["hidden_dim"],
            num_classes=2,
            dropout=params["dropout"],
            eeg_encoder_type=EEG_ENCODER_CONFIG["encoder_type"],
            n_channels=EEG_ENCODER_CONFIG["n_channels"],
            n_times=EEG_ENCODER_CONFIG["n_times"],
            max_windows=EEG_ENCODER_CONFIG["max_windows"],
            window_chunk_size=EEG_ENCODER_CONFIG["window_chunk_size"],
        ).to(device)

        train_loader = DataLoader(
            train_ds, batch_size=params["batch_size"], shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=params["batch_size"], shuffle=False, num_workers=0
        )

        # Class weights (inverse frequency)
        train_labels = [train_ds[i][5].item() for i in range(len(train_ds))]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_auc = 0.0
        patience_counter = 0

        for epoch in range(TRAINING_FIXED["epochs"]):
            train_epoch_mlp(model, train_loader, optimiser, criterion, device)
            _, val_metrics = evaluate_mlp(model, val_loader, criterion, device)

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= TRAINING_FIXED["patience"]:
                break

        fold_aucs.append(best_val_auc)

        # Report intermediate value for pruning
        trial.report(np.mean(fold_aucs), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_aucs))


# ---------------------------------------------------------------------------
# Objective 2: Exp11 QuadMLPv2 with EEG2Vec (AUC 0.791)
# ---------------------------------------------------------------------------

def objective_exp11_quadmlpv2(trial: optuna.Trial) -> float:
    """Optuna objective for Exp11 QuadMLPv2 (EEG2Vec 128D).

    Tunes: learning_rate, weight_decay, dropout, hidden_dim, batch_size,
           aggregator_type, eeg_embed_dim.
    Fixed: ClinicalBERT text, ChemBERTa SMILES, EEG2Vec encoder.
    """
    from exp7_all_modalities.config import CLINICAL_DIM, SMILES_DIM, TEXT_DIM
    from exp7_all_modalities.data_pipeline import create_quad_modality_datasets
    from exp7_all_modalities.training import evaluate_mlp, train_epoch_mlp
    from exp11_eeg_upgrade.models import QuadMLPv2

    params = sample_params(trial, SEARCH_SPACES["exp11_quadmlpv2"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reuse same cached data as exp7a
    df, smiles_emb, smiles_idx, text_emb, eeg_data = _get_exp7a_data()
    outcomes = df["outcome"].values

    kfold = StratifiedKFold(**CV_CONFIG)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(np.zeros(len(outcomes)), outcomes)
    ):
        train_ds, val_ds, _ = create_quad_modality_datasets(
            df, smiles_emb, smiles_idx, text_emb, eeg_data, train_idx, val_idx
        )

        model = QuadMLPv2(
            clinical_dim=CLINICAL_DIM,
            text_dim=TEXT_DIM,
            smiles_dim=SMILES_DIM,
            hidden_dim=params["hidden_dim"],
            num_classes=2,
            dropout=params["dropout"],
            eeg_encoder_type="eeg2vec",
            eeg_embed_dim=params["eeg_embed_dim"],
            aggregator_type=params["aggregator_type"],
            n_channels=N_CHANNELS,
            n_times=2000,
            max_windows=120,
            window_chunk_size=32,
        ).to(device)

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
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_auc = 0.0
        patience_counter = 0

        for epoch in range(TRAINING_FIXED["epochs"]):
            train_epoch_mlp(model, train_loader, optimiser, criterion, device)
            _, val_metrics = evaluate_mlp(model, val_loader, criterion, device)

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= TRAINING_FIXED["patience"]:
                break

        fold_aucs.append(best_val_auc)

        trial.report(np.mean(fold_aucs), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_aucs))


# ---------------------------------------------------------------------------
# Objective 3: Exp12 TripleModalityFuseMoE (AUC 0.760)
# ---------------------------------------------------------------------------

def objective_exp12_fusemoe(trial: optuna.Trial) -> float:
    """Optuna objective for Exp12 TripleModalityFuseMoE.

    Tunes: learning_rate, weight_decay, dropout, num_experts, top_k,
           aux_loss_weight, temp_decay.
    Fixed: ClinicalBERT text, ChemBERTa SMILES, SimpleCNN EEG encoder.
    """
    from exp3_fusion.config import EEG_ENCODER_CONFIG, SMILES_DIMS
    from exp3_fusion.data_pipeline import create_datasets, get_max_channels
    from exp3_fusion.models.triple_fusemoe import TripleModalityFuseMoE
    from exp3_fusion.training import evaluate, train_epoch

    params = sample_params(trial, SEARCH_SPACES["exp12_fusemoe"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load cached data
    text_emb, eeg_data, smiles_emb, smiles_idx, df = _get_exp12_data()
    outcomes = df["outcome"].values
    max_channels = get_max_channels(eeg_data)
    smiles_dim = SMILES_DIMS["chemberta"]

    kfold = StratifiedKFold(**CV_CONFIG)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(np.zeros(len(outcomes)), outcomes)
    ):
        train_ds, val_ds = create_datasets(
            text_emb, eeg_data, smiles_emb, smiles_idx, df,
            train_idx, val_idx, max_channels,
        )

        model = TripleModalityFuseMoE(
            text_dim=768,
            smiles_dim=smiles_dim,
            hidden_dim=256,
            num_classes=2,
            num_experts=params["num_experts"],
            top_k=params["top_k"],
            num_heads=4,
            dropout=params["dropout"],
            aux_loss_weight=params["aux_loss_weight"],
            eeg_encoder_type=EEG_ENCODER_CONFIG["encoder_type"],
            n_eeg_channels=EEG_ENCODER_CONFIG["n_channels"],
            n_eeg_times=EEG_ENCODER_CONFIG["n_times"],
            eeg_embed_dim=EEG_ENCODER_CONFIG["embed_dim"],
            num_eeg_layers=EEG_ENCODER_CONFIG["num_layers"],
            max_windows=EEG_ENCODER_CONFIG["max_windows"],
            window_chunk_size=EEG_ENCODER_CONFIG["window_chunk_size"],
        ).to(device)

        # Configure temperature annealing
        temp_decay = params["temp_decay"]
        if temp_decay is None:
            model.fuse_moe.temperature_decay = 1.0
        else:
            model.fuse_moe.temperature_decay = temp_decay

        train_loader = DataLoader(
            train_ds, batch_size=8, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=8, shuffle=False, num_workers=0
        )

        # Class weights (label is index 4 for triple modality dataset)
        train_labels = [train_ds[i][4].item() for i in range(len(train_ds))]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_auc = 0.0
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
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= TRAINING_FIXED["patience"]:
                break

        fold_aucs.append(best_val_auc)

        trial.report(np.mean(fold_aucs), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_aucs))
