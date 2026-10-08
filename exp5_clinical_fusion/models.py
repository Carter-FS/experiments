"""Models for Experiment 5: Clinical + Single Modality Fusion.

All models use late fusion: each modality is encoded separately,
then concatenated and passed through a classifier.
"""

from typing import Optional

import torch
import torch.nn as nn
from exp2_fusion.config import N_CHANNELS

from .config import (
    CLINICAL_DIM,
    EEG_DIM,
    EEG_ENCODER_CONFIG,
    SMILES_DIMS,
    TEXT_DIM,
    TRAINING_CONFIG,
)

# Import EEG components from exp2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp2_fusion.models.eeg_encoders import get_eeg_encoder
from exp2_fusion.models.eeg_transformer import EEGWindowTransformer


class ModalityEncoder(nn.Module):
    """Generic MLP encoder for a single modality."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FusionClassifier(nn.Module):
    """MLP classifier for fused representations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class ClinicalSMILESFusion(nn.Module):
    """Fusion model for Clinical + SMILES (Experiment 5a).

    Architecture:
        Clinical (20D) -> Encoder -> 64D
        SMILES (768/256D) -> Encoder -> 64D
        Concatenate -> 128D -> Classifier -> 2 classes
    """

    def __init__(
        self,
        clinical_dim: int = CLINICAL_DIM,
        smiles_dim: int = 768,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.clinical_encoder = ModalityEncoder(clinical_dim, hidden_dim, dropout)
        self.smiles_encoder = ModalityEncoder(smiles_dim, hidden_dim, dropout)
        self.classifier = FusionClassifier(hidden_dim * 2, hidden_dim, num_classes, dropout)

    def forward(
        self,
        clinical: torch.Tensor,
        smiles: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            clinical: Clinical features (batch, 20).
            smiles: SMILES embedding (batch, smiles_dim).

        Returns:
            Logits (batch, num_classes).
        """
        clinical_feat = self.clinical_encoder(clinical)
        smiles_feat = self.smiles_encoder(smiles)
        fused = torch.cat([clinical_feat, smiles_feat], dim=1)
        return self.classifier(fused)


class ClinicalTextFusion(nn.Module):
    """Fusion model for Clinical + Text/LLM (Experiment 5b).

    Architecture:
        Clinical (20D) -> Encoder -> 64D
        Text (768D) -> Encoder -> 64D
        Concatenate -> 128D -> Classifier -> 2 classes
    """

    def __init__(
        self,
        clinical_dim: int = CLINICAL_DIM,
        text_dim: int = TEXT_DIM,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.clinical_encoder = ModalityEncoder(clinical_dim, hidden_dim, dropout)
        self.text_encoder = ModalityEncoder(text_dim, hidden_dim, dropout)
        self.classifier = FusionClassifier(hidden_dim * 2, hidden_dim, num_classes, dropout)

    def forward(
        self,
        clinical: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            clinical: Clinical features (batch, 20).
            text: Text embedding (batch, 768).

        Returns:
            Logits (batch, num_classes).
        """
        clinical_feat = self.clinical_encoder(clinical)
        text_feat = self.text_encoder(text)
        fused = torch.cat([clinical_feat, text_feat], dim=1)
        return self.classifier(fused)


class ClinicalEEGFusion(nn.Module):
    """Fusion model for Clinical + EEG (Experiment 5c).

    Architecture:
        Clinical (20D) -> Encoder -> 64D
        EEG (num_windows, 27, 2000) -> EEG2Vec -> Aggregator -> 64D
        Concatenate -> 128D -> Classifier -> 2 classes
    """

    def __init__(
        self,
        clinical_dim: int = CLINICAL_DIM,
        eeg_encoder_type: str = "eeg2vec",
        n_channels: int = N_CHANNELS,
        n_times: int = 2000,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
        max_windows: int = 120,
        window_chunk_size: int = 32,
    ):
        super().__init__()

        self.window_chunk_size = window_chunk_size

        # Clinical encoder
        self.clinical_encoder = ModalityEncoder(clinical_dim, hidden_dim, dropout)

        # EEG window encoder (via factory)
        self.window_encoder = get_eeg_encoder(
            encoder_type=eeg_encoder_type,
            n_channels=n_channels,
            n_times=n_times,
            emb_size=256,
        )

        # EEG window aggregator
        self.aggregator = EEGWindowTransformer(
            embed_dim=256,
            output_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout=dropout,
            max_windows=max_windows,
        )

        # Classifier
        self.classifier = FusionClassifier(hidden_dim * 2, hidden_dim, num_classes, dropout)

    def encode_eeg_windows(
        self,
        windows: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode EEG windows with chunking for memory efficiency.

        Args:
            windows: (batch, num_windows, channels, time)
            padding_mask: (batch, num_windows)

        Returns:
            Aggregated EEG embedding (batch, hidden_dim)
        """
        batch_size, num_windows, n_channels, n_times = windows.shape

        # Encode windows in chunks
        all_embeddings = []
        for i in range(0, num_windows, self.window_chunk_size):
            chunk = windows[:, i:i + self.window_chunk_size]
            chunk_size = chunk.shape[1]

            # Flatten for encoding
            chunk_flat = chunk.reshape(batch_size * chunk_size, n_channels, n_times)

            # Encode
            chunk_emb = self.window_encoder(chunk_flat)

            # Reshape back
            chunk_emb = chunk_emb.reshape(batch_size, chunk_size, -1)
            all_embeddings.append(chunk_emb)

        # Concatenate all chunks
        window_embeddings = torch.cat(all_embeddings, dim=1)

        # Aggregate
        eeg_feat = self.aggregator(window_embeddings, padding_mask)

        return eeg_feat

    def forward(
        self,
        clinical: torch.Tensor,
        eeg_windows: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            clinical: Clinical features (batch, 20).
            eeg_windows: EEG windows (batch, num_windows, channels, time).
            padding_mask: Padding mask (batch, num_windows).

        Returns:
            Logits (batch, num_classes).
        """
        clinical_feat = self.clinical_encoder(clinical)
        eeg_feat = self.encode_eeg_windows(eeg_windows, padding_mask)
        fused = torch.cat([clinical_feat, eeg_feat], dim=1)
        return self.classifier(fused)


def get_model(
    modality: str,
    smiles_model: Optional[str] = None,
    text_model: Optional[str] = None,
    eeg_model: Optional[str] = None,
    device: torch.device = None,
) -> nn.Module:
    """Create model based on experiment configuration.

    Args:
        modality: 'smiles', 'text', or 'eeg'
        smiles_model: 'chemberta' or 'smilestrf' (for smiles modality)
        text_model: 'clinicalbert' or 'pubmedbert' (for text modality)
        eeg_model: EEG encoder type string (for eeg modality)
        device: Device to place model on.

    Returns:
        Initialised model.
    """
    dropout = TRAINING_CONFIG["dropout"]
    num_classes = TRAINING_CONFIG["num_classes"]

    if modality == "smiles":
        smiles_dim = SMILES_DIMS.get(smiles_model, 768)
        model = ClinicalSMILESFusion(
            clinical_dim=CLINICAL_DIM,
            smiles_dim=smiles_dim,
            hidden_dim=64,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif modality == "text":
        model = ClinicalTextFusion(
            clinical_dim=CLINICAL_DIM,
            text_dim=TEXT_DIM,
            hidden_dim=64,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif modality == "eeg":
        model = ClinicalEEGFusion(
            clinical_dim=CLINICAL_DIM,
            eeg_encoder_type=EEG_ENCODER_CONFIG["encoder_type"],
            n_channels=EEG_ENCODER_CONFIG["n_channels"],
            n_times=EEG_ENCODER_CONFIG["n_times"],
            hidden_dim=64,
            num_classes=num_classes,
            dropout=dropout,
            max_windows=EEG_ENCODER_CONFIG["max_windows"],
            window_chunk_size=EEG_ENCODER_CONFIG["window_chunk_size"],
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")

    if device is not None:
        model = model.to(device)

    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_models():
    """Test model forward passes."""
    print("Testing Exp5 models...")

    device = torch.device("cpu")
    batch_size = 4

    # Test Clinical + SMILES
    print("\nTesting ClinicalSMILESFusion:")
    clinical = torch.randn(batch_size, CLINICAL_DIM)
    smiles = torch.randn(batch_size, 768)

    model = get_model("smiles", smiles_model="chemberta", device=device)
    print(f"  Parameters: {count_parameters(model):,}")
    output = model(clinical, smiles)
    print(f"  Input: clinical={clinical.shape}, smiles={smiles.shape}")
    print(f"  Output: {output.shape}")

    # Test Clinical + Text
    print("\nTesting ClinicalTextFusion:")
    text = torch.randn(batch_size, TEXT_DIM)

    model = get_model("text", text_model="clinicalbert", device=device)
    print(f"  Parameters: {count_parameters(model):,}")
    output = model(clinical, text)
    print(f"  Input: clinical={clinical.shape}, text={text.shape}")
    print(f"  Output: {output.shape}")

    # Test Clinical + EEG
    print("\nTesting ClinicalEEGFusion:")
    eeg_windows = torch.randn(batch_size, 120, 27, 2000)
    padding_mask = torch.zeros(batch_size, 120, dtype=torch.bool)
    padding_mask[:, 90:] = True  # Last 30 windows are padded

    model = get_model("eeg", eeg_model="eeg2vec", device=device)
    print(f"  Parameters: {count_parameters(model):,}")
    output = model(clinical, eeg_windows, padding_mask)
    print(f"  Input: clinical={clinical.shape}, eeg={eeg_windows.shape}")
    print(f"  Output: {output.shape}")

    print("\nAll model tests passed!")


if __name__ == "__main__":
    test_models()
