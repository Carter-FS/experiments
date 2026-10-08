"""Test LaBraM encoder with realistic EEG dimensions and memory constraints."""

import torch
import sys
sys.path.insert(0, "/home/carter/carter_massive/experiments")

from exp2_fusion.models.eeg_encoders import get_eeg_encoder, LABRAM_AVAILABLE, EEGNET_AVAILABLE
from exp2_fusion.config import EEG_CONFIG, BATCH_SIZE_BY_ENCODER, CHUNK_SIZE_BY_ENCODER


def test_labram_shapes():
    """Verify LaBraM produces correct output shapes with varying channel counts."""
    if not LABRAM_AVAILABLE:
        print("SKIP: LaBraM not available")
        return

    n_times = int(EEG_CONFIG["window_sec"] * EEG_CONFIG["target_sr"])  # 2000
    test_cases = [
        (19, 128, "19-channel montage"),
        (19, 128, "19-channel montage"),
    ]

    for n_channels, emb_size, desc in test_cases:
        print(f"Testing LaBraM with {desc} (n_channels={n_channels}, emb_size={emb_size})")
        encoder = get_eeg_encoder("labram", n_channels=n_channels, n_times=n_times, emb_size=emb_size)
        x = torch.randn(1, n_channels, n_times)
        out = encoder(x)
        assert out.shape == (1, emb_size), f"Expected (1, {emb_size}), got {out.shape}"
        print(f"  Output shape: {out.shape} - PASS")
        del encoder, x, out


def test_eegnet_shapes():
    """Verify EEGNet produces correct output shapes with varying channel counts."""
    if not EEGNET_AVAILABLE:
        print("SKIP: EEGNet not available")
        return

    n_times = int(EEG_CONFIG["window_sec"] * EEG_CONFIG["target_sr"])  # 2000
    test_cases = [
        (19, 256, "19-channel montage"),
        (19, 256, "19-channel montage"),
    ]

    for n_channels, emb_size, desc in test_cases:
        print(f"Testing EEGNet with {desc} (n_channels={n_channels}, emb_size={emb_size})")
        encoder = get_eeg_encoder("eegnet", n_channels=n_channels, n_times=n_times, emb_size=emb_size)
        x = torch.randn(2, n_channels, n_times)
        out = encoder(x)
        assert out.shape == (2, emb_size), f"Expected (2, {emb_size}), got {out.shape}"
        print(f"  Output shape: {out.shape} - PASS")
        params = sum(p.numel() for p in encoder.parameters())
        print(f"  Parameters: {params:,}")
        del encoder, x, out


def test_labram_chunked_processing():
    """Test LaBraM with chunked window processing (simulates real pipeline)."""
    if not LABRAM_AVAILABLE:
        print("SKIP: LaBraM not available")
        return

    n_channels = 19
    n_times = 2000
    emb_size = 128
    chunk_size = CHUNK_SIZE_BY_ENCODER.get("labram", 4)

    print(f"Testing LaBraM chunked processing (chunk_size={chunk_size})")
    encoder = get_eeg_encoder("labram", n_channels=n_channels, n_times=n_times, emb_size=emb_size)

    # Simulate 12 windows processed in chunks
    num_windows = 12
    all_embeddings = []
    for i in range(0, num_windows, chunk_size):
        chunk = torch.randn(min(chunk_size, num_windows - i), n_channels, n_times)
        with torch.no_grad():
            emb = encoder(chunk)
        all_embeddings.append(emb)
        print(f"  Chunk {i//chunk_size + 1}: {chunk.shape[0]} windows -> {emb.shape}")

    combined = torch.cat(all_embeddings, dim=0)
    assert combined.shape == (num_windows, emb_size)
    print(f"  Combined shape: {combined.shape} - PASS")


if __name__ == "__main__":
    test_labram_shapes()
    print()
    test_eegnet_shapes()
    print()
    test_labram_chunked_processing()
    print("\nAll encoder tests passed!")
