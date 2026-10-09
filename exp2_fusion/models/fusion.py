"""Fusion models for EEG + SMILES embeddings."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from exp2_fusion.config import N_CHANNELS
import torch.nn.functional as F

from .eeg_encoders import get_eeg_encoder
from .eeg_transformer import EEGWindowTransformer
from shared.fuse_moe import FuseMoE


class EEGSMILESMLPFusion(nn.Module):
    """MLP fusion model for EEG + SMILES embeddings.

    Concatenates EEG and SMILES embeddings and passes through MLP classifier.
    """

    def __init__(
        self,
        eeg_encoder_type: str = "labram",
        n_eeg_channels: int = N_CHANNELS,
        n_eeg_times: int = 2000,
        eeg_embed_dim: int = 200,
        smiles_embed_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_windows: int = 120,
        window_chunk_size: int = 16,
    ):
        super().__init__()

        self.window_chunk_size = window_chunk_size

        # EEG window encoder
        self.window_encoder = get_eeg_encoder(
            encoder_type=eeg_encoder_type,
            n_channels=n_eeg_channels,
            n_times=n_eeg_times,
            emb_size=eeg_embed_dim,
        )

        # Window aggregation transformer
        self.aggregator = EEGWindowTransformer(
            embed_dim=eeg_embed_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_windows=max_windows,
        )

        # SMILES projection
        self.smiles_proj = nn.Linear(smiles_embed_dim, hidden_dim)

        # Fusion MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def encode_windows_chunked(
        self,
        windows: torch.Tensor,
        chunk_size: int = 16,
    ) -> torch.Tensor:
        """Encode windows in chunks to save memory.

        Args:
            windows: (batch * num_windows, channels, time)
            chunk_size: Number of windows to process at once.

        Returns:
            Window embeddings of shape (batch * num_windows, embed_dim)
        """
        n_total = windows.shape[0]
        embeddings = []

        for i in range(0, n_total, chunk_size):
            chunk = windows[i:i + chunk_size]
            with torch.no_grad() if not self.training else torch.enable_grad():
                emb = self.window_encoder(chunk)
            embeddings.append(emb)

        return torch.cat(embeddings, dim=0)

    def forward(
        self,
        eeg_windows: torch.Tensor,
        padding_mask: torch.Tensor,
        smiles_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            eeg_windows: (batch, num_windows, channels, time)
            padding_mask: (batch, num_windows) boolean, True for padded
            smiles_emb: (batch, smiles_embed_dim)

        Returns:
            Logits of shape (batch, num_classes)
        """
        batch_size, num_windows = eeg_windows.shape[:2]  # per-window shape: (channels, time) or (dim,)

        # Encode EEG windows (in chunks to save memory)
        windows_flat = eeg_windows.reshape(batch_size * num_windows, *eeg_windows.shape[2:])
        window_embeddings = self.encode_windows_chunked(windows_flat, chunk_size=self.window_chunk_size)
        embed_dim = window_embeddings.shape[-1]
        window_embeddings = window_embeddings.view(batch_size, num_windows, embed_dim)

        # Aggregate windows
        eeg_emb = self.aggregator(window_embeddings, padding_mask)

        # Project SMILES
        smiles_proj = self.smiles_proj(smiles_emb)

        # Concatenate and classify
        fused = torch.cat([eeg_emb, smiles_proj], dim=-1)
        logits = self.classifier(fused)

        return logits


class EEGSMILESFuseMoE(nn.Module):
    """FuseMoE model for EEG + SMILES fusion.

    Uses Mixture of Experts for multimodal fusion.
    """

    def __init__(
        self,
        eeg_encoder_type: str = "labram",
        n_eeg_channels: int = N_CHANNELS,
        n_eeg_times: int = 2000,
        eeg_embed_dim: int = 200,
        smiles_embed_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_experts: int = 4,
        top_k: int = 2,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        aux_loss_weight: float = 0.1,
        max_windows: int = 120,
        window_chunk_size: int = 16,
    ):
        super().__init__()

        self.aux_loss_weight = aux_loss_weight
        self.window_chunk_size = window_chunk_size

        # EEG window encoder
        self.window_encoder = get_eeg_encoder(
            encoder_type=eeg_encoder_type,
            n_channels=n_eeg_channels,
            n_times=n_eeg_times,
            emb_size=eeg_embed_dim,
        )

        # Window aggregation transformer
        self.aggregator = EEGWindowTransformer(
            embed_dim=eeg_embed_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_windows=max_windows,
        )

        # SMILES projection
        self.smiles_proj = nn.Linear(smiles_embed_dim, hidden_dim)

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # MoE fusion layer (shared reference implementation)
        self.fuse_moe = FuseMoE(
            strategy="joint",
            input_dims=hidden_dim * 2,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_experts=num_experts,
            k=top_k,
            num_modalities=1,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def update_temperature(self, global_step: int):
        """Update FuseMoE temperature for annealing."""
        self.fuse_moe.update_temperature(global_step)

    def encode_windows_chunked(
        self,
        windows: torch.Tensor,
        chunk_size: int = 16,
    ) -> torch.Tensor:
        """Encode windows in chunks to save memory."""
        n_total = windows.shape[0]
        embeddings = []

        for i in range(0, n_total, chunk_size):
            chunk = windows[i:i + chunk_size]
            with torch.no_grad() if not self.training else torch.enable_grad():
                emb = self.window_encoder(chunk)
            embeddings.append(emb)

        return torch.cat(embeddings, dim=0)

    def forward(
        self,
        eeg_windows: torch.Tensor,
        padding_mask: torch.Tensor,
        smiles_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            eeg_windows: (batch, num_windows, channels, time)
            padding_mask: (batch, num_windows) boolean, True for padded
            smiles_emb: (batch, smiles_embed_dim)

        Returns:
            Tuple of (logits, aux_loss)
        """
        batch_size, num_windows = eeg_windows.shape[:2]  # per-window shape: (channels, time) or (dim,)

        # Encode EEG windows (in chunks to save memory)
        windows_flat = eeg_windows.reshape(batch_size * num_windows, *eeg_windows.shape[2:])
        window_embeddings = self.encode_windows_chunked(windows_flat, chunk_size=self.window_chunk_size)
        embed_dim = window_embeddings.shape[-1]
        window_embeddings = window_embeddings.view(batch_size, num_windows, embed_dim)

        # Aggregate windows
        eeg_emb = self.aggregator(window_embeddings, padding_mask)

        # Project SMILES
        smiles_proj = self.smiles_proj(smiles_emb)

        # Cross-attention (EEG attends to SMILES)
        eeg_query = eeg_emb.unsqueeze(1)
        smiles_kv = smiles_proj.unsqueeze(1)
        attended_eeg, _ = self.cross_attention(eeg_query, smiles_kv, smiles_kv)
        attended_eeg = attended_eeg.squeeze(1)

        # Concatenate for MoE
        fused_input = torch.cat([attended_eeg, smiles_proj], dim=-1)

        # MoE layer
        fused, aux_loss = self.fuse_moe(fused_input)

        # Classify
        logits = self.classifier(fused)

        return logits, aux_loss * self.aux_loss_weight


def get_fusion_model(
    fusion_type: str = "mlp",
    eeg_encoder_type: str = "labram",
    n_eeg_channels: int = N_CHANNELS,
    smiles_embed_dim: int = 768,
    **kwargs,
) -> nn.Module:
    """Factory function to create fusion model.

    Args:
        fusion_type: 'mlp' or 'fusemoe'
        eeg_encoder_type: 'labram' or 'simplecnn'
        n_eeg_channels: Number of EEG channels
        smiles_embed_dim: SMILES embedding dimension
        **kwargs: Additional model arguments

    Returns:
        Fusion model
    """
    # Common arguments for both models
    common_keys = ["hidden_dim", "num_classes", "num_heads", "num_layers", "dropout", "window_chunk_size",
                   "eeg_embed_dim"]
    common_kwargs = {k: v for k, v in kwargs.items() if k in common_keys}

    if fusion_type == "mlp":
        return EEGSMILESMLPFusion(
            eeg_encoder_type=eeg_encoder_type,
            n_eeg_channels=n_eeg_channels,
            smiles_embed_dim=smiles_embed_dim,
            **common_kwargs,
        )
    elif fusion_type == "fusemoe":
        # FuseMoE has additional arguments
        moe_keys = ["num_experts", "top_k", "aux_loss_weight"]
        moe_kwargs = {k: v for k, v in kwargs.items() if k in moe_keys}
        return EEGSMILESFuseMoE(
            eeg_encoder_type=eeg_encoder_type,
            n_eeg_channels=n_eeg_channels,
            smiles_embed_dim=smiles_embed_dim,
            **common_kwargs,
            **moe_kwargs,
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


def test_fusion_models():
    """Test fusion model implementations."""
    print("Testing fusion models...")

    batch_size = 2
    num_windows = 120
    n_channels = N_CHANNELS
    n_times = 2000
    smiles_dim = 768

    # Create inputs
    eeg = torch.randn(batch_size, num_windows, n_channels, n_times)
    mask = torch.zeros(batch_size, num_windows, dtype=torch.bool)
    mask[:, 90:] = True
    smiles = torch.randn(batch_size, smiles_dim)

    # Test MLP fusion
    print("\nTesting MLP fusion:")
    mlp_model = EEGSMILESMLPFusion(
        eeg_encoder_type="simplecnn",  # Use simpler encoder for testing
        n_eeg_channels=n_channels,
        smiles_embed_dim=smiles_dim,
    )
    logits = mlp_model(eeg, mask, smiles)
    print(f"  Output shape: {logits.shape}")

    # Test FuseMoE
    print("\nTesting FuseMoE:")
    moe_model = EEGSMILESFuseMoE(
        eeg_encoder_type="simplecnn",
        n_eeg_channels=n_channels,
        smiles_embed_dim=smiles_dim,
    )
    logits, aux_loss = moe_model(eeg, mask, smiles)
    print(f"  Output shape: {logits.shape}")
    print(f"  Aux loss: {aux_loss.item():.4f}")


if __name__ == "__main__":
    test_fusion_models()
