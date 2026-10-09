"""Window aggregation transformer for EEG embeddings.

Aggregates embeddings from multiple 10-second windows into a single
patient-level EEG representation using transformer attention.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position information."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class EEGWindowTransformer(nn.Module):
    """Transformer for aggregating EEG window embeddings.

    Takes embeddings from multiple 10-second windows and produces
    a single patient-level embedding using transformer attention.
    Uses key_padding_mask to handle variable-length sequences.
    """

    def __init__(
        self,
        embed_dim: int = 200,
        output_dim: Optional[int] = None,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
        max_windows: int = 120,
    ):
        """Initialize window aggregation transformer.

        Args:
            embed_dim: Dimension of input window embeddings.
            output_dim: Dimension of output (defaults to embed_dim).
            num_heads: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            ff_dim: Feed-forward hidden dimension.
            dropout: Dropout probability.
            max_windows: Maximum number of windows.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim or embed_dim

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_windows, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Optional projection to different output dim
        if self.output_dim != embed_dim:
            self.projection = nn.Linear(embed_dim, self.output_dim)
        else:
            self.projection = nn.Identity()

    def forward(
        self,
        window_embeddings: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate window embeddings into single patient embedding.

        Args:
            window_embeddings: Window embeddings of shape (batch, num_windows, embed_dim).
            padding_mask: Boolean mask of shape (batch, num_windows).
                          True indicates padded (invalid) positions.

        Returns:
            Aggregated embedding of shape (batch, output_dim).
        """
        # Add positional encoding
        x = self.pos_encoder(window_embeddings)

        # Apply transformer with padding mask
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Apply layer norm
        x = self.norm(x)

        # Mean pooling over valid (non-padded) windows
        if padding_mask is not None:
            # Invert mask: True for valid positions
            valid_mask = ~padding_mask
            # Expand mask for broadcasting
            valid_mask = valid_mask.unsqueeze(-1)  # (batch, num_windows, 1)
            # Masked mean
            x = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            # Simple mean pooling
            x = x.mean(dim=1)

        # Project to output dimension
        x = self.projection(x)

        return x


class EEGEncoder(nn.Module):
    """Full EEG encoder: window encoder + aggregation transformer.

    Processes raw EEG windows through a per-window encoder and then
    aggregates the window embeddings using a transformer.
    """

    def __init__(
        self,
        window_encoder: nn.Module,
        window_embed_dim: int = 200,
        output_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_windows: int = 120,
    ):
        """Initialize full EEG encoder.

        Args:
            window_encoder: Module that encodes individual windows.
            window_embed_dim: Embedding dimension from window encoder.
            output_dim: Final output embedding dimension.
            num_heads: Number of attention heads in aggregator.
            num_layers: Number of transformer layers in aggregator.
            dropout: Dropout probability.
            max_windows: Maximum number of windows.
        """
        super().__init__()

        self.window_encoder = window_encoder
        self.aggregator = EEGWindowTransformer(
            embed_dim=window_embed_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_windows=max_windows,
        )

    def forward(
        self,
        windows: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode EEG windows to patient-level embedding.

        Args:
            windows: EEG windows of shape (batch, num_windows, channels, time).
            padding_mask: Boolean mask of shape (batch, num_windows).
                          True indicates padded windows.

        Returns:
            Patient embedding of shape (batch, output_dim).
        """
        batch_size, num_windows = windows.shape[:2]  # per-window shape: (channels, time) or (dim,)

        # Flatten batch and windows for encoder
        windows_flat = windows.reshape(batch_size * num_windows, *windows.shape[2:])

        # Encode all windows
        window_embeddings = self.window_encoder(windows_flat)

        # Reshape back to (batch, num_windows, embed_dim)
        embed_dim = window_embeddings.shape[-1]
        window_embeddings = window_embeddings.view(batch_size, num_windows, embed_dim)

        # Aggregate windows
        patient_embedding = self.aggregator(window_embeddings, padding_mask)

        return patient_embedding


def test_transformer():
    """Test window aggregation transformer."""
    print("Testing EEG Window Transformer...")

    batch_size = 4
    num_windows = 120
    embed_dim = 200

    # Create random window embeddings
    window_embeddings = torch.randn(batch_size, num_windows, embed_dim)

    # Create padding mask (last 30 windows are padded)
    padding_mask = torch.zeros(batch_size, num_windows, dtype=torch.bool)
    padding_mask[:, 90:] = True

    # Test transformer
    transformer = EEGWindowTransformer(
        embed_dim=embed_dim,
        output_dim=256,
        num_heads=4,
        num_layers=2,
    )

    output = transformer(window_embeddings, padding_mask)

    print(f"Input window embeddings shape: {window_embeddings.shape}")
    print(f"Padding mask shape: {padding_mask.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    test_transformer()
