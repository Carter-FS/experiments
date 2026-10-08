"""Triple modality FuseMoE model (Experiment 3b)."""

from typing import Tuple

import torch
import torch.nn as nn
from exp2_fusion.config import N_CHANNELS
import torch.nn.functional as F

# Import EEG components from exp2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from exp2_fusion.models.eeg_encoders import get_eeg_encoder
from exp2_fusion.models.eeg_transformer import EEGWindowTransformer
from shared.fuse_moe import FuseMoE


class TripleModalityFuseMoE(nn.Module):
    """FuseMoE model for Text + EEG + SMILES triple modality fusion.

    Architecture:
    - Each modality -> Projection(dim->256) + Learnable modality token
    - Self-attention across 3 modality tokens
    - Per-modality FuseMoE routing (Laplace gating, MI loss, 3-layer residual experts)
    - Classifier
    """

    def __init__(
        self,
        text_dim: int = 768,
        smiles_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_experts: int = 4,
        top_k: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        aux_loss_weight: float = 0.1,
        # EEG encoder config
        eeg_encoder_type: str = "simplecnn",
        n_eeg_channels: int = N_CHANNELS,
        n_eeg_times: int = 2000,
        eeg_embed_dim: int = 256,
        num_eeg_layers: int = 2,
        max_windows: int = 120,
        window_chunk_size: int = 32,
    ):
        super().__init__()

        self.aux_loss_weight = aux_loss_weight
        self.window_chunk_size = window_chunk_size

        # Text projection
        self.text_proj = nn.Linear(text_dim, hidden_dim)

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
            num_layers=num_eeg_layers,
            dropout=dropout,
            max_windows=max_windows,
        )

        # SMILES projection
        self.smiles_proj = nn.Linear(smiles_dim, hidden_dim)

        # Learnable modality tokens
        self.text_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.eeg_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.smiles_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Self-attention for cross-modal interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # MoE fusion (shared reference implementation)
        self.fuse_moe = FuseMoE(
            strategy="permodality",
            input_dims=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_experts=num_experts,
            k=top_k,
            num_modalities=3,
        )
        self.fuse_norm = nn.LayerNorm(hidden_dim)

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
        chunk_size: int = 32,
    ) -> torch.Tensor:
        """Encode windows in chunks to save memory."""
        n_total = windows.shape[0]
        embeddings = []

        for i in range(0, n_total, chunk_size):
            chunk = windows[i : i + chunk_size]
            with torch.no_grad() if not self.training else torch.enable_grad():
                emb = self.window_encoder(chunk)
            embeddings.append(emb)

        return torch.cat(embeddings, dim=0)

    def forward(
        self,
        text_emb: torch.Tensor,
        eeg_windows: torch.Tensor,
        padding_mask: torch.Tensor,
        smiles_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            text_emb: (batch, text_dim)
            eeg_windows: (batch, num_windows, channels, time)
            padding_mask: (batch, num_windows) boolean, True for padded
            smiles_emb: (batch, smiles_dim)

        Returns:
            Tuple of (logits, aux_loss)
        """
        batch_size, num_windows, n_channels, n_times = eeg_windows.shape

        # Project text
        text_proj = self.text_proj(text_emb)  # (batch, hidden_dim)

        # Encode EEG windows
        windows_flat = eeg_windows.view(batch_size * num_windows, n_channels, n_times)
        window_embeddings = self.encode_windows_chunked(windows_flat, self.window_chunk_size)
        embed_dim = window_embeddings.shape[-1]
        window_embeddings = window_embeddings.view(batch_size, num_windows, embed_dim)

        # Aggregate windows
        eeg_emb = self.aggregator(window_embeddings, padding_mask)  # (batch, hidden_dim)

        # Project SMILES
        smiles_proj = self.smiles_proj(smiles_emb)  # (batch, hidden_dim)

        # Add modality tokens to embeddings
        text_with_token = text_proj.unsqueeze(1) + self.text_token.expand(batch_size, -1, -1)
        eeg_with_token = eeg_emb.unsqueeze(1) + self.eeg_token.expand(batch_size, -1, -1)
        smiles_with_token = smiles_proj.unsqueeze(1) + self.smiles_token.expand(batch_size, -1, -1)

        # Concatenate modality tokens: (batch, 3, hidden_dim)
        modality_tokens = torch.cat([text_with_token, eeg_with_token, smiles_with_token], dim=1)

        # Cross-modal self-attention
        attn_out, _ = self.cross_attention(modality_tokens, modality_tokens, modality_tokens)
        modality_tokens = self.attn_norm(modality_tokens + attn_out)

        # Extract individual modality vectors for per-modality routing
        text_vec = modality_tokens[:, 0, :]    # (batch, hidden_dim)
        eeg_vec = modality_tokens[:, 1, :]     # (batch, hidden_dim)
        smiles_vec = modality_tokens[:, 2, :]  # (batch, hidden_dim)

        fused, total_aux_loss = self.fuse_moe(text_vec, eeg_vec, smiles_vec)
        fused = self.fuse_norm(fused)

        # Classify
        logits = self.classifier(fused)

        return logits, total_aux_loss * self.aux_loss_weight


def test_triple_fusemoe():
    """Test TripleModalityFuseMoE model."""
    print("Testing TripleModalityFuseMoE...")

    batch_size = 2
    text_dim = 768
    smiles_dim = 768
    num_windows = 120
    n_channels = N_CHANNELS
    n_times = 2000

    # Create inputs
    text = torch.randn(batch_size, text_dim)
    eeg = torch.randn(batch_size, num_windows, n_channels, n_times)
    mask = torch.zeros(batch_size, num_windows, dtype=torch.bool)
    mask[:, 90:] = True  # Last 30 windows are padding
    smiles = torch.randn(batch_size, smiles_dim)

    # Create model
    model = TripleModalityFuseMoE(
        text_dim=text_dim,
        smiles_dim=smiles_dim,
        hidden_dim=256,
        num_classes=2,
        eeg_encoder_type="simplecnn",
        n_eeg_channels=n_channels,
    )

    # Forward pass
    logits, aux_loss = model(text, eeg, mask, smiles)
    print(f"Output shape: {logits.shape}")
    print(f"Aux loss: {aux_loss.item():.4f}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")


if __name__ == "__main__":
    test_triple_fusemoe()
