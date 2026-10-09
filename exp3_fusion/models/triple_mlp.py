"""Triple modality MLP fusion model (Experiment 3a)."""

import torch
import torch.nn as nn
from exp2_fusion.config import N_CHANNELS

# Import EEG components from exp2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from exp2_fusion.models.eeg_encoders import flatten_windows, get_eeg_encoder
from exp2_fusion.models.eeg_transformer import EEGWindowTransformer


class TripleModalityMLP(nn.Module):
    """MLP fusion model for Text + EEG + SMILES embeddings.

    Architecture:
    - Text (768D) -> Linear(768->256) -> LayerNorm
    - EEG (256D) -> Identity (already 256D from SimpleCNN+Transformer)
    - SMILES (768D or 256D) -> Linear(dim->256) -> LayerNorm
    - Concat (768D) -> MLP classifier
    """

    def __init__(
        self,
        text_dim: int = 768,
        smiles_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        # EEG encoder config
        eeg_encoder_type: str = "simplecnn",
        n_eeg_channels: int = N_CHANNELS,
        n_eeg_times: int = 2000,
        eeg_embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        max_windows: int = 120,
        window_chunk_size: int = 32,
    ):
        super().__init__()

        self.window_chunk_size = window_chunk_size

        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

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
            dropout=0.1,
            max_windows=max_windows,
        )

        # SMILES projection
        self.smiles_proj = nn.Sequential(
            nn.Linear(smiles_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Fusion MLP classifier (3 * hidden_dim -> num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),  # 0.2 if dropout=0.3
            nn.Linear(hidden_dim // 2, num_classes),
        )

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
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            text_emb: (batch, text_dim)
            eeg_windows: (batch, num_windows, channels, time)
            padding_mask: (batch, num_windows) boolean, True for padded
            smiles_emb: (batch, smiles_dim)

        Returns:
            Logits of shape (batch, num_classes)
        """
        windows_flat, batch_size, num_windows = flatten_windows(eeg_windows)

        # Project text
        text_proj = self.text_proj(text_emb)

        # Encode EEG windows
        window_embeddings = self.encode_windows_chunked(windows_flat, self.window_chunk_size)
        embed_dim = window_embeddings.shape[-1]
        window_embeddings = window_embeddings.view(batch_size, num_windows, embed_dim)

        # Aggregate windows
        eeg_emb = self.aggregator(window_embeddings, padding_mask)

        # Project SMILES
        smiles_proj = self.smiles_proj(smiles_emb)

        # Concatenate all modalities
        fused = torch.cat([text_proj, eeg_emb, smiles_proj], dim=-1)

        # Classify
        logits = self.classifier(fused)

        return logits


def test_triple_mlp():
    """Test TripleModalityMLP model."""
    print("Testing TripleModalityMLP...")

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
    model = TripleModalityMLP(
        text_dim=text_dim,
        smiles_dim=smiles_dim,
        hidden_dim=256,
        num_classes=2,
        eeg_encoder_type="simplecnn",
        n_eeg_channels=n_channels,
    )

    # Forward pass
    logits = model(text, eeg, mask, smiles)
    print(f"Output shape: {logits.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")


if __name__ == "__main__":
    test_triple_mlp()
