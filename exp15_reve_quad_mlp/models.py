"""Models for Experiment 15: REVE-based quad-modal fusion.

QuadFusionREVE is structurally identical to QuadFusionMLP from
exp7_all_modalities/models.py except for the EEG branch:

  - exp7's EEG branch:   raw windows (B, 120, 19, 2000)
                         -> EEG2Vec window encoder (B, 120, 256)
                         -> EEGWindowTransformer aggregator (B, 64)
  - exp15's EEG branch:  pre-computed REVE features (B, 120, 512)
                         -> Linear(512 -> 256) projection (B, 120, 256)
                         -> EEGWindowTransformer aggregator (B, 64)

REVE itself is run upstream (analysis/reve_extract_features.py in the
thesisStandalone repo) and is not part of this model. The projection,
aggregator, classifier, and all other modality encoders ARE trained
per fold.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

from .config import (
    AGGREGATOR_CONFIG,
    CLINICAL_DIM,
    MLP_CONFIG,
    REVE_DIM,
    SMILES_DIM,
    TEXT_DIM,
)

# Reuse exp2's transformer aggregator and exp7's ModalityEncoder + FusionClassifier
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp2_fusion.models.eeg_transformer import EEGWindowTransformer  # noqa: E402
from exp7_all_modalities.models import (  # noqa: E402
    FusionClassifier,
    ModalityEncoder,
)


class QuadFusionREVE(nn.Module):
    """Late-fusion MLP model for all four modalities using REVE EEG features.

    Forward arguments mirror exp7's QuadFusionMLP so the existing
    training/eval loops can be reused with minimal changes. ``eeg_windows``
    now carries (batch, max_windows, REVE_DIM=512) instead of
    (batch, max_windows, channels, time).
    """

    def __init__(
        self,
        clinical_dim: int = CLINICAL_DIM,
        text_dim: int = TEXT_DIM,
        smiles_dim: int = SMILES_DIM,
        reve_dim: int = REVE_DIM,
        embed_dim: int = AGGREGATOR_CONFIG["embed_dim"],
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
        max_windows: int = AGGREGATOR_CONFIG["max_windows"],
        num_heads: int = AGGREGATOR_CONFIG["num_heads"],
        num_layers: int = AGGREGATOR_CONFIG["num_layers"],
    ):
        super().__init__()

        # Modality encoders (clinical / text / SMILES identical to exp7)
        self.clinical_encoder = ModalityEncoder(clinical_dim, hidden_dim, dropout)
        self.text_encoder = ModalityEncoder(text_dim, hidden_dim, dropout)
        self.smiles_encoder = ModalityEncoder(smiles_dim, hidden_dim, dropout)

        # EEG branch: project REVE per-window features (512) -> aggregator
        # embed_dim (256), then use the same EEGWindowTransformer as exp7.
        self.reve_projection = nn.Linear(reve_dim, embed_dim)
        self.aggregator = EEGWindowTransformer(
            embed_dim=embed_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_windows=max_windows,
        )

        # Classifier (4 modalities * hidden_dim)
        self.classifier = FusionClassifier(hidden_dim * 4, hidden_dim, num_classes, dropout)

    def forward(
        self,
        clinical: torch.Tensor,
        text: torch.Tensor,
        eeg_windows: torch.Tensor,
        padding_mask: torch.Tensor,
        smiles: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            clinical: Clinical features (batch, 19).
            text: Text embedding (batch, 768).
            eeg_windows: REVE per-window features (batch, max_windows, 512).
            padding_mask: Padding mask (batch, max_windows) bool, True = padded.
            smiles: SMILES embedding (batch, 768).

        Returns:
            Logits (batch, num_classes).
        """
        clinical_feat = self.clinical_encoder(clinical)
        text_feat = self.text_encoder(text)
        smiles_feat = self.smiles_encoder(smiles)
        reve_projected = self.reve_projection(eeg_windows)         # (B, 120, 256)
        eeg_feat = self.aggregator(reve_projected, padding_mask)   # (B, 64)
        fused = torch.cat([clinical_feat, text_feat, eeg_feat, smiles_feat], dim=1)
        return self.classifier(fused)


def get_model(
    fusion: str = "mlp",
    text_model: str = None,
    smiles_model: str = None,
    device: torch.device = None,
    reve_dim: int = REVE_DIM,
) -> nn.Module:
    """Create QuadFusionREVE. The fusion arg is kept for signature parity with
    exp7.get_model; only "mlp" is supported in exp15. ``reve_dim`` is the width of
    the stored per-window features (512 for REVE-base, 200 for LaBraM-base).
    """
    if fusion != "mlp":
        raise ValueError(
            f"exp15 only supports 'mlp' fusion; got {fusion!r}. "
            "MoE fusion would need a parallel exp15b directory if required."
        )
    config = MLP_CONFIG
    model = QuadFusionREVE(
        reve_dim=reve_dim,
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    )
    if device is not None:
        model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
