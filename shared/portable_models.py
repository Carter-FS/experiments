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

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from shared.eeg_cache import load_cache
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

def build_model(config: str, device: torch.device, clinical_dim: int | None = None) -> nn.Module:
    """``clinical_dim`` is the clinical input width (the preprocessor's
    ``output_dim``); None means the full 16-feature encoding. Cross-cohort
    callers pass the reduced width (analysis plan B.4)."""
    from exp4_baseline.config import CLINICAL_DIM
    cd = CLINICAL_DIM if clinical_dim is None else int(clinical_dim)
    if config == "Exp4a":
        from exp4_baseline.config import CONFIG_4A
        from exp4_baseline.models import ClinicalMLP
        return ClinicalMLP(input_dim=cd, hidden_dims=CONFIG_4A["hidden_dims"],
                           num_classes=CONFIG_4A["num_classes"], dropout=CONFIG_4A["dropout"]).to(device)
    if config == "Exp5a":
        from exp5_clinical_fusion.models import ClinicalSMILESFusion
        return ClinicalSMILESFusion(clinical_dim=cd, smiles_dim=768).to(device)
    if config == "Exp5b":
        from exp5_clinical_fusion.models import ClinicalTextFusion
        return ClinicalTextFusion(clinical_dim=cd).to(device)
    if config == "Exp5c":
        from exp5_clinical_fusion.models import ClinicalEEGFusion
        return ClinicalEEGFusion(clinical_dim=cd, n_channels=N_CHANNELS, n_times=N_TIMES,
                                 max_windows=MAX_WINDOWS).to(device)
    if config in ("Exp6b", "Exp6b_eeg2vec"):
        # Exp6b is the original SimpleCNN model the published HEP1 table used;
        # the clean protocol pre-specifies EEG2Vec for every EEG configuration.
        from exp6_clinical_triple.models import ClinicalSMILESEEGFusion
        encoder = "eeg2vec" if config == "Exp6b_eeg2vec" else "simplecnn"
        return ClinicalSMILESEEGFusion(clinical_dim=cd, n_channels=N_CHANNELS, n_times=N_TIMES,
                                       max_windows=MAX_WINDOWS, eeg_encoder_type=encoder).to(device)
    if config == "Exp7a":
        from exp7_all_modalities.models import QuadFusionMLP
        return QuadFusionMLP(clinical_dim=cd, n_channels=N_CHANNELS, n_times=N_TIMES,
                             max_windows=MAX_WINDOWS).to(device)
    if config == "Exp16_tiny":
        # Reduced-capacity quad model (exp16 "tiny": hidden_dim 16, eeg_embed_dim 64,
        # MeanMax pooling; ~157k params vs ~2M). Same forward signature as Exp7a.
        from exp11_eeg_upgrade.models import QuadMLPv2
        return QuadMLPv2(
            clinical_dim=cd, hidden_dim=16, eeg_embed_dim=64, aggregator_type="meanmax",
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
    trace: dict | None = None,
    fixed_epochs: int | None = None,
    val_cohorts=None,
) -> nn.Module:
    """Train a model with early stopping on a held-out val split.

    ``model_factory`` (a no-argument callable returning an nn.Module) replaces
    build_model for "LF:" late-fusion configurations; otherwise the clinical
    input width is taken from ``train_tensors[0]``.

    Refit protocol (analysis plan B.2): ``trace`` collects the validation
    probabilities after every epoch; ``fixed_epochs`` trains exactly that
    many epochs with no validation set (``val_tensors`` None) and returns the
    final model. ``val_cohorts`` makes the early-stopping criterion the
    cohort-stratified AUC (exp18 mixed arms, plan B.6).
    """
    model = (model_factory().to(device) if model_factory is not None
             else build_model(config, device, clinical_dim=train_tensors[0].shape[1]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    class_counts = np.bincount(train_labels.numpy())
    cw = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32)
    cw = cw / cw.sum()
    criterion = nn.CrossEntropyLoss(weight=cw.to(device))
    train_tensors = [t.to(device) for t in train_tensors]
    if val_tensors is not None:
        val_tensors = [t.to(device) for t in val_tensors]
        val_labels = val_labels.to(device)
    train_labels = train_labels.to(device)
    train_ds = TensorDataset(*train_tensors, train_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    for epoch in range(N_EPOCHS_MAX if fixed_epochs is None else fixed_epochs):
        model.train()
        for batch in train_loader:
            *features, labels = batch
            optimizer.zero_grad()
            logits = forward_pass(model, features, config)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        if fixed_epochs is not None:
            continue
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = forward_pass(model, val_tensors, config)
            val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
        if trace is not None:
            trace.setdefault("val_probs", []).append(val_probs.astype(float))
        val_auc = _val_auc(val_labels.cpu().numpy(), val_probs, val_cohorts)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            break
    if fixed_epochs is None and best_state is not None:
        model.load_state_dict(best_state)
    return model


def _val_auc(y, probs, cohorts=None) -> float:
    """Early-stopping criterion: AUC, or the cohort-stratified AUC when
    ``cohorts`` is given; 0.5 when undefined."""
    if cohorts is not None:
        from shared.stats_util import cohort_stratified_auc
        v = cohort_stratified_auc(y, probs, cohorts)
        return 0.5 if not np.isfinite(v) else v
    return roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.5


def predict(model: nn.Module, tensors: list[torch.Tensor], config: str, device: torch.device) -> np.ndarray:
    model.eval()
    tensors = [t.to(device) for t in tensors]
    with torch.no_grad():
        logits = forward_pass(model, tensors, config)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float | None = None) -> dict:
    """AUC plus sensitivity/specificity at ``threshold`` (a transported one,
    chosen on the training cohort), or at the scored cohort's own Youden point
    when ``threshold`` is None (an oracle, labelled ``_ownthr`` in clean runs)."""
    if len(np.unique(y_true)) < 2 or len(y_true) < 2:
        return {"auc": float("nan"), "sens": float("nan"), "spec": float("nan"),
                "bal_acc": float("nan"), "n": int(len(y_true)),
                "n_responder": int(y_true.sum()) if len(y_true) else 0}
    auc = float(roc_auc_score(y_true, y_prob))
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_idx = int(np.argmax(tpr - fpr))
        thr = float(thresholds[j_idx])
    else:
        thr = float(threshold)
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


def refit_clinical(train_df: pd.DataFrame, apply_df: pd.DataFrame, fit_idx: np.ndarray,
                   drop: tuple[str, ...] = ()):
    """Clinical tensors for both cohorts with the preprocessor fitted on the
    training cohort's fit rows only (clean protocol; imputation statistics
    otherwise leak from the outer test fold). ``drop`` removes binary
    features (cross-cohort runs pass hep_cohort.CROSS_COHORT_DROP)."""
    from exp4_baseline.data_pipeline import ClinicalFeaturePreprocessor
    pre = ClinicalFeaturePreprocessor(drop=drop).fit(train_df.iloc[fit_idx])
    to_t = lambda d: torch.from_numpy(pre.transform(d)).float()  # noqa: E731
    return to_t(train_df), to_t(apply_df)


# -----------------------------------------------------------------------
# EEG configurations
# -----------------------------------------------------------------------

EEG_CONVENTION = "zscore_window"   # per-window, per-channel z-score in microvolts


def load_eeg_cache(path: Path) -> dict:
    """``{pid: (windows, padding_mask)}`` from a version-2 cache (shared.eeg_cache)."""
    return load_cache(path, EEG_CONVENTION)


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
    trace: dict | None = None,
    fixed_epochs: int | None = None,
    val_cohorts=None,
) -> nn.Module:
    """EEG counterpart of train_fold (same refit-protocol arguments)."""
    model = build_model(config, device, clinical_dim=train_modalities["clinical"].shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    class_counts = np.bincount(train_labels.numpy())
    cw = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32)
    cw = cw / cw.sum()
    criterion = nn.CrossEntropyLoss(weight=cw.to(device))
    rng = np.random.default_rng(current_seed(CV_SEED))
    best_val_auc = 0.0
    best_state = None
    patience_counter = 0
    for epoch in range(EEG_N_EPOCHS_MAX if fixed_epochs is None else fixed_epochs):
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
        if fixed_epochs is not None:
            continue
        # Validation
        model.eval()
        val_probs = predict_eeg(model, val_modalities, config, device)
        if trace is not None:
            trace.setdefault("val_probs", []).append(np.asarray(val_probs, dtype=float))
        val_auc = _val_auc(val_labels.numpy(), val_probs, val_cohorts)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= EEG_EARLY_STOP_PATIENCE:
            break
    if fixed_epochs is None and best_state is not None:
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
