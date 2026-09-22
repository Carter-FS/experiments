"""Model builders and train/predict loops for the cohort-portable configurations.

The six HEP1 Table 3 configurations (Exp4a, Exp5a, Exp5b without EEG; Exp5c,
Exp6b, Exp7a on the 19-channel montage) plus the reduced-capacity Exp16_tiny,
with the training loops the external-validation scripts use. Shared by
thesisStandalone/analysis/hep_*.py and exp18_mixed_cohort so both train with
one implementation. Moved unchanged from hep_external_validation.py (non-EEG)
and hep_external_validation_eeg.py (EEG); the EEG loop's hyperparameters carry
an ``EEG_`` prefix here because the two scripts used the same names.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

from shared.cv_splits import current_seed

CV_SEED = 42

# Non-EEG configurations (Exp4a / Exp5a / Exp5b).
N_EPOCHS_MAX = 80
EARLY_STOP_PATIENCE = 15
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

# EEG configurations (Exp5c / Exp6b / Exp7a / Exp16_tiny), 19-channel montage.
EEG_N_EPOCHS_MAX = 60
EEG_EARLY_STOP_PATIENCE = 10
EEG_BATCH_SIZE = 4  # EEG memory; matches Stage A typical
N_CHANNELS = 19
N_TIMES = 2000
MAX_WINDOWS = 120

EEG_CONFIGS = ("Exp5c", "Exp6b", "Exp6b_eeg2vec", "Exp7a", "Exp16_tiny")


# -----------------------------------------------------------------------
# Models (reuse the exp4-7 / exp11 architectures)
# -----------------------------------------------------------------------

def build_model(config: str, device: torch.device) -> nn.Module:
    if config == "Exp4a":
        from exp4_baseline.models import get_model
        return get_model("mlp", device)
    if config == "Exp5a":
        from exp5_clinical_fusion.models import ClinicalSMILESFusion
        return ClinicalSMILESFusion(smiles_dim=768).to(device)
    if config == "Exp5b":
        from exp5_clinical_fusion.models import ClinicalTextFusion
        return ClinicalTextFusion().to(device)
    if config == "Exp5c":
        from exp5_clinical_fusion.models import ClinicalEEGFusion
        return ClinicalEEGFusion(n_channels=N_CHANNELS, n_times=N_TIMES, max_windows=MAX_WINDOWS).to(device)
    if config in ("Exp6b", "Exp6b_eeg2vec"):
        # Exp6b is the original SimpleCNN model the published HEP1 table used;
        # the clean protocol pre-specifies EEG2Vec for every EEG configuration.
        from exp6_clinical_triple.models import ClinicalSMILESEEGFusion
        encoder = "eeg2vec" if config == "Exp6b_eeg2vec" else "simplecnn"
        return ClinicalSMILESEEGFusion(n_channels=N_CHANNELS, n_times=N_TIMES, max_windows=MAX_WINDOWS,
                                       eeg_encoder_type=encoder).to(device)
    if config == "Exp7a":
        from exp7_all_modalities.models import QuadFusionMLP
        return QuadFusionMLP(n_channels=N_CHANNELS, n_times=N_TIMES, max_windows=MAX_WINDOWS).to(device)
    if config == "Exp16_tiny":
        # Reduced-capacity quad model (exp16 "tiny": hidden_dim 16, eeg_embed_dim 64,
        # MeanMax pooling; ~157k params vs ~2M). Same forward signature as Exp7a.
        from exp11_eeg_upgrade.models import QuadMLPv2
        return QuadMLPv2(
            hidden_dim=16, eeg_embed_dim=64, aggregator_type="meanmax",
            eeg_encoder_type="eeg2vec",
            n_channels=N_CHANNELS, n_times=N_TIMES, max_windows=MAX_WINDOWS,
        ).to(device)
    raise ValueError(f"Unknown config: {config}")


# Configurations named "LF:<anything>" use LateFusionMLP built by a caller-supplied
# factory (exp19): each input tensor is one modality, in the order given.
LATE_FUSION_PREFIX = "LF:"


class LateFusionMLP(nn.Module):
    """Generic late fusion: each input -> Linear(d, 64), ReLU, LayerNorm, dropout;
    concatenate; Linear -> 64, ReLU, dropout, Linear -> 2. The block the paper
    describes for its late-fusion MLP, parameterised by the input dimensions so
    serialised-text and tabular inputs share one architecture."""

    def __init__(self, input_dims: list[int], hidden: int = 64, dropout: float = DROPOUT):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout))
            for d in input_dims
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * len(input_dims), hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2),
        )

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([b(x) for b, x in zip(self.branches, inputs)], dim=1))


def forward_pass(model: nn.Module, batch: tuple, config: str) -> torch.Tensor:
    if config.startswith(LATE_FUSION_PREFIX):
        return model(*batch)
    if config == "Exp4a":
        clinical = batch[0]
        return model(clinical)
    if config in ("Exp5a", "Exp5b"):
        clinical, modality = batch
        return model(clinical, modality)
    raise ValueError(config)


def train_fold(
    config: str,
    train_tensors: list[torch.Tensor],
    val_tensors: list[torch.Tensor],
    train_labels: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    model_factory=None,
) -> nn.Module:
    """Train a model with early stopping on a held-out val split.

    ``model_factory`` (a no-argument callable returning an nn.Module) replaces
    build_model for "LF:" late-fusion configurations.
    """
    model = model_factory().to(device) if model_factory is not None else build_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    class_counts = np.bincount(train_labels.numpy())
    cw = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32)
    cw = cw / cw.sum()
    criterion = nn.CrossEntropyLoss(weight=cw.to(device))
    train_tensors = [t.to(device) for t in train_tensors]
    val_tensors = [t.to(device) for t in val_tensors]
    train_labels = train_labels.to(device)
    val_labels = val_labels.to(device)
    train_ds = TensorDataset(*train_tensors, train_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    for epoch in range(N_EPOCHS_MAX):
        model.train()
        for batch in train_loader:
            *features, labels = batch
            optimizer.zero_grad()
            logits = forward_pass(model, features, config)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = forward_pass(model, val_tensors, config)
            val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
        if len(np.unique(val_labels.cpu().numpy())) > 1:
            val_auc = roc_auc_score(val_labels.cpu().numpy(), val_probs)
        else:
            val_auc = 0.5
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model: nn.Module, tensors: list[torch.Tensor], config: str, device: torch.device) -> np.ndarray:
    model.eval()
    tensors = [t.to(device) for t in tensors]
    with torch.no_grad():
        logits = forward_pass(model, tensors, config)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    if len(np.unique(y_true)) < 2 or len(y_true) < 2:
        return {"auc": float("nan"), "sens": float("nan"), "spec": float("nan"),
                "bal_acc": float("nan"), "n": int(len(y_true)),
                "n_responder": int(y_true.sum()) if len(y_true) else 0}
    auc = float(roc_auc_score(y_true, y_prob))
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_idx = int(np.argmax(tpr - fpr))
    thr = float(thresholds[j_idx])
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {"auc": auc, "sens": float(sens), "spec": float(spec),
            "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
            "threshold": thr, "n": int(len(y_true)),
            "n_responder": int(y_true.sum())}


def refit_clinical(train_df: pd.DataFrame, apply_df: pd.DataFrame, fit_idx: np.ndarray):
    """Clinical tensors for both cohorts with the preprocessor fitted on the
    training cohort's fit rows only (clean protocol; imputation statistics
    otherwise leak from the outer test fold)."""
    from exp4_baseline.data_pipeline import ClinicalFeaturePreprocessor
    pre = ClinicalFeaturePreprocessor().fit(train_df.iloc[fit_idx])
    to_t = lambda d: torch.from_numpy(pre.transform(d)).float()  # noqa: E731
    return to_t(train_df), to_t(apply_df)


# -----------------------------------------------------------------------
# EEG configurations
# -----------------------------------------------------------------------

def load_eeg_cache(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def stack_eeg_for_pids(eeg_cache: dict, pids: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack per-patient (windows, padding_mask) into batched tensors.

    Returns:
        windows: (n_patients, MAX_WINDOWS, N_CHANNELS, N_TIMES) float32
        padding_mask: (n_patients, MAX_WINDOWS) bool
    """
    n = len(pids)
    windows = np.zeros((n, MAX_WINDOWS, N_CHANNELS, N_TIMES), dtype=np.float32)
    pad = np.ones((n, MAX_WINDOWS), dtype=bool)  # True = padded
    for i, pid in enumerate(pids):
        w, m = eeg_cache[pid]
        nw = min(w.shape[0], MAX_WINDOWS)
        windows[i, :nw] = w[:nw]
        pad[i, :nw] = m[:nw] if hasattr(m, "shape") else False
    return torch.from_numpy(windows), torch.from_numpy(pad)


def model_forward(model: nn.Module, batch: dict, config: str) -> torch.Tensor:
    if config == "Exp5c":
        return model(batch["clinical"], batch["eeg_windows"], batch["eeg_mask"])
    if config in ("Exp6b", "Exp6b_eeg2vec"):
        return model(batch["clinical"], batch["smiles"], batch["eeg_windows"], batch["eeg_mask"])
    if config in ("Exp7a", "Exp16_tiny"):
        return model(batch["clinical"], batch["text"], batch["eeg_windows"], batch["eeg_mask"], batch["smiles"])
    raise ValueError(config)


def index_modalities(d: dict, idx) -> dict:
    return {k: v[idx] for k, v in d.items()}


def to_device(d: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in d.items()}


def iterate_minibatches(modalities: dict, labels: torch.Tensor, batch_size: int, shuffle: bool, rng: np.random.Generator | None):
    n = len(labels)
    order = np.arange(n)
    if shuffle:
        order = rng.permutation(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield {k: v[idx] for k, v in modalities.items()}, labels[idx]


def train_fold_eeg(
    config: str,
    train_modalities: dict,
    val_modalities: dict,
    train_labels: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
) -> nn.Module:
    model = build_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    class_counts = np.bincount(train_labels.numpy())
    cw = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32)
    cw = cw / cw.sum()
    criterion = nn.CrossEntropyLoss(weight=cw.to(device))
    rng = np.random.default_rng(current_seed(CV_SEED))
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    for epoch in range(EEG_N_EPOCHS_MAX):
        model.train()
        for batch_mod, batch_labels in iterate_minibatches(train_modalities, train_labels, EEG_BATCH_SIZE, True, rng):
            batch_mod = to_device(batch_mod, device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model_forward(model, batch_mod, config)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        # Validation
        model.eval()
        val_probs = predict_eeg(model, val_modalities, config, device)
        if len(np.unique(val_labels.numpy())) > 1:
            val_auc = roc_auc_score(val_labels.numpy(), val_probs)
        else:
            val_auc = 0.5
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= EEG_EARLY_STOP_PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_eeg(model: nn.Module, modalities: dict, config: str, device: torch.device) -> np.ndarray:
    model.eval()
    probs_list = []
    with torch.no_grad():
        for batch_mod, _ in iterate_minibatches(modalities, torch.zeros(len(next(iter(modalities.values())))), EEG_BATCH_SIZE, False, None):
            batch_mod = to_device(batch_mod, device)
            logits = model_forward(model, batch_mod, config)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs_list.append(probs)
    return np.concatenate(probs_list)


def filter_to_intersection(df: pd.DataFrame, *required_lookups: set | dict) -> pd.DataFrame:
    """Filter df to patients present in every required lookup (set or dict of pids)."""
    mask = pd.Series(True, index=df.index)
    for lookup in required_lookups:
        pids = set(lookup.keys()) if isinstance(lookup, dict) else set(lookup)
        mask &= df["pid"].astype(str).isin(pids)
    return df[mask].reset_index(drop=True)
