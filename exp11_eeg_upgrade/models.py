"""Models for Experiment 11: EEG2Vec 128D re-runs with aggregator variants.

Adapted from exp3a (TripleModalityMLP), exp6b (ClinicalSMILESEEGFusion),
and exp7a (QuadFusionMLP) with configurable EEG embed_dim and aggregator.
"""

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp4_baseline.config import CLINICAL_DIM
from exp2_fusion.models.eeg_encoders import get_eeg_encoder
from exp2_fusion.models.eeg_transformer import EEGWindowTransformer
from exp2_fusion.models.aggregators import get_aggregator


def _build_aggregator(aggregator_type, embed_dim, output_dim, num_heads=4, num_layers=2, dropout=0.1, max_windows=120):
    """Create EEG window aggregator by type."""
    if aggregator_type == "transformer":
        return EEGWindowTransformer(
            embed_dim=embed_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_windows=max_windows,
        )
    return get_aggregator(
        aggregator_type=aggregator_type,
        embed_dim=embed_dim,
        output_dim=output_dim,
    )


def _encode_eeg_chunked(window_encoder, windows, padding_mask, chunk_size=32):
    """Encode EEG windows in chunks then aggregate. Shared by all models."""
    batch_size, num_windows, n_channels, n_times = windows.shape
    all_embeddings = []
    for i in range(0, num_windows, chunk_size):
        chunk = windows[:, i:i + chunk_size]
        cs = chunk.shape[1]
        chunk_flat = chunk.reshape(batch_size * cs, n_channels, n_times)
        chunk_emb = window_encoder(chunk_flat)
        chunk_emb = chunk_emb.reshape(batch_size, cs, -1)
        all_embeddings.append(chunk_emb)
    return torch.cat(all_embeddings, dim=1)


class ModalityEncoder(nn.Module):
    """Generic MLP encoder for a single modality (from exp6/exp7)."""

    def __init__(self, input_dim, output_dim=64, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.encoder(x)


class FusionClassifier(nn.Module):
    """MLP classifier for fused representations (from exp6/exp7)."""

    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class TripleMLPv2(nn.Module):
    """Triple modality MLP with configurable EEG pipeline (based on exp3a).

    Architecture: Text(256D) || EEG(256D) || SMILES(256D) -> 768D -> MLP -> 2

    Changes from TripleModalityMLP:
    - eeg_embed_dim parameter (was hardcoded to match encoder default)
    - aggregator_type parameter (was hardcoded as EEGWindowTransformer)
    """

    def __init__(
        self,
        text_dim=768,
        smiles_dim=768,
        hidden_dim=256,
        num_classes=2,
        dropout=0.3,
        eeg_encoder_type="eeg2vec",
        eeg_embed_dim=128,
        aggregator_type="transformer",
        n_eeg_channels=27,
        n_eeg_times=2000,
        max_windows=120,
        window_chunk_size=32,
    ):
        super().__init__()
        self.window_chunk_size = window_chunk_size

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.window_encoder = get_eeg_encoder(
            encoder_type=eeg_encoder_type,
            n_channels=n_eeg_channels,
            n_times=n_eeg_times,
            emb_size=eeg_embed_dim,
        )

        self.aggregator = _build_aggregator(
            aggregator_type, eeg_embed_dim, hidden_dim,
            max_windows=max_windows,
        )

        self.smiles_proj = nn.Sequential(
            nn.Linear(smiles_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, text_emb, eeg_windows, padding_mask, smiles_emb):
        text_proj = self.text_proj(text_emb)
        window_embeddings = _encode_eeg_chunked(
            self.window_encoder, eeg_windows, padding_mask, self.window_chunk_size
        )
        eeg_emb = self.aggregator(window_embeddings, padding_mask)
        smiles_proj = self.smiles_proj(smiles_emb)
        fused = torch.cat([text_proj, eeg_emb, smiles_proj], dim=-1)
        return self.classifier(fused)


class ClinicalEEGFusionv2(nn.Module):
    """Clinical + SMILES + EEG late fusion with configurable EEG pipeline (based on exp6b).

    Architecture: Clinical(64D) || SMILES(64D) || EEG(64D) -> 192D -> MLP -> 2

    Changes from ClinicalSMILESEEGFusion:
    - Uses get_eeg_encoder factory (was hardcoded SimpleCNNEncoder)
    - eeg_embed_dim parameter (was hardcoded 256)
    - aggregator_type parameter (was hardcoded EEGWindowTransformer)
    """

    def __init__(
        self,
        clinical_dim=CLINICAL_DIM,
        smiles_dim=768,
        hidden_dim=64,
        num_classes=2,
        dropout=0.3,
        eeg_encoder_type="eeg2vec",
        eeg_embed_dim=128,
        aggregator_type="transformer",
        n_channels=27,
        n_times=2000,
        max_windows=120,
        window_chunk_size=32,
    ):
        super().__init__()
        self.window_chunk_size = window_chunk_size

        self.clinical_encoder = ModalityEncoder(clinical_dim, hidden_dim, dropout)
        self.smiles_encoder = ModalityEncoder(smiles_dim, hidden_dim, dropout)

        self.window_encoder = get_eeg_encoder(
            encoder_type=eeg_encoder_type,
            n_channels=n_channels,
            n_times=n_times,
            emb_size=eeg_embed_dim,
        )

        self.aggregator = _build_aggregator(
            aggregator_type, eeg_embed_dim, hidden_dim,
            dropout=dropout, max_windows=max_windows,
        )

        self.classifier = FusionClassifier(hidden_dim * 3, hidden_dim, num_classes, dropout)

    def forward(self, clinical, smiles, eeg_windows, padding_mask):
        clinical_feat = self.clinical_encoder(clinical)
        smiles_feat = self.smiles_encoder(smiles)
        window_embeddings = _encode_eeg_chunked(
            self.window_encoder, eeg_windows, padding_mask, self.window_chunk_size
        )
        eeg_feat = self.aggregator(window_embeddings, padding_mask)
        fused = torch.cat([clinical_feat, smiles_feat, eeg_feat], dim=1)
        return self.classifier(fused)


class QuadMLPv2(nn.Module):
    """Quad modality MLP with configurable EEG pipeline (based on exp7a).

    Architecture: Clinical(64D) || Text(64D) || EEG(64D) || SMILES(64D) -> 256D -> MLP -> 2

    Changes from QuadFusionMLP:
    - eeg_embed_dim parameter (was hardcoded 256)
    - aggregator_type parameter (was hardcoded EEGWindowTransformer)
    """

    def __init__(
        self,
        clinical_dim=CLINICAL_DIM,
        text_dim=768,
        smiles_dim=768,
        hidden_dim=64,
        num_classes=2,
        dropout=0.3,
        eeg_encoder_type="eeg2vec",
        eeg_embed_dim=128,
        aggregator_type="transformer",
        n_channels=27,
        n_times=2000,
        max_windows=120,
        window_chunk_size=32,
    ):
        super().__init__()
        self.window_chunk_size = window_chunk_size

        self.clinical_encoder = ModalityEncoder(clinical_dim, hidden_dim, dropout)
        self.text_encoder = ModalityEncoder(text_dim, hidden_dim, dropout)
        self.smiles_encoder = ModalityEncoder(smiles_dim, hidden_dim, dropout)

        self.window_encoder = get_eeg_encoder(
            encoder_type=eeg_encoder_type,
            n_channels=n_channels,
            n_times=n_times,
            emb_size=eeg_embed_dim,
        )

        self.aggregator = _build_aggregator(
            aggregator_type, eeg_embed_dim, hidden_dim,
            dropout=dropout, max_windows=max_windows,
        )

        self.classifier = FusionClassifier(hidden_dim * 4, hidden_dim, num_classes, dropout)

    def forward(self, clinical, text, eeg, mask, smiles):
        clinical_feat = self.clinical_encoder(clinical)
        text_feat = self.text_encoder(text)
        smiles_feat = self.smiles_encoder(smiles)
        window_embeddings = _encode_eeg_chunked(
            self.window_encoder, eeg, mask, self.window_chunk_size
        )
        eeg_feat = self.aggregator(window_embeddings, mask)
        fused = torch.cat([clinical_feat, text_feat, eeg_feat, smiles_feat], dim=1)
        return self.classifier(fused)


def test_models():
    """Smoke test all model variants."""
    print("Testing exp11 models...")

    batch_size = 2
    n_windows = 10  # Small for testing
    n_channels = 27
    n_times = 2000

    eeg = torch.randn(batch_size, n_windows, n_channels, n_times)
    mask = torch.zeros(batch_size, n_windows, dtype=torch.bool)
    mask[:, 8:] = True

    for agg in ["transformer", "meanmax"]:
        print(f"\n--- Aggregator: {agg} ---")

        # TripleMLPv2
        model = TripleMLPv2(
            text_dim=768, smiles_dim=768, eeg_embed_dim=128,
            aggregator_type=agg,
        )
        text = torch.randn(batch_size, 768)
        smiles = torch.randn(batch_size, 768)
        out = model(text, eeg, mask, smiles)
        print(f"TripleMLPv2: {out.shape}, params={sum(p.numel() for p in model.parameters()):,}")

        # ClinicalEEGFusionv2
        model = ClinicalEEGFusionv2(
            smiles_dim=768, eeg_embed_dim=128,
            aggregator_type=agg,
        )
        clinical = torch.randn(batch_size, CLINICAL_DIM)
        out = model(clinical, smiles, eeg, mask)
        print(f"ClinicalEEGFusionv2: {out.shape}, params={sum(p.numel() for p in model.parameters()):,}")

        # QuadMLPv2
        model = QuadMLPv2(
            smiles_dim=768, eeg_embed_dim=128,
            aggregator_type=agg,
        )
        clinical = torch.randn(batch_size, CLINICAL_DIM)
        out = model(clinical, text, eeg, mask, smiles)
        print(f"QuadMLPv2: {out.shape}, params={sum(p.numel() for p in model.parameters()):,}")

    print("\nAll exp11 model tests passed!")


if __name__ == "__main__":
    test_models()
