"""EEG encoder wrappers for extracting embeddings from EEG windows."""

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from exp2_fusion.config import N_CHANNELS
import torch.nn.functional as F

# Import guard for braindecode (may fail due to CUDA library issues)
LABRAM_AVAILABLE = False
LABRAM_IMPORT_ERROR = None
EEGNET_AVAILABLE = False
EEGNET_IMPORT_ERROR = None

try:
    from braindecode.models import Labram
    LABRAM_AVAILABLE = True
except ImportError as e:
    LABRAM_IMPORT_ERROR = str(e)
except Exception as e:
    # Catch other errors like OSError for missing CUDA libraries
    LABRAM_IMPORT_ERROR = f"{type(e).__name__}: {e}"

try:
    # Try newer name first, fall back to deprecated EEGNetv4
    try:
        from braindecode.models import EEGNet as BraindecodeEEGNet
    except ImportError:
        from braindecode.models import EEGNetv4 as BraindecodeEEGNet
    EEGNET_AVAILABLE = True
except ImportError as e:
    EEGNET_IMPORT_ERROR = str(e)
except Exception as e:
    EEGNET_IMPORT_ERROR = f"{type(e).__name__}: {e}"

logger = logging.getLogger(__name__)


class LaBraMEncoder(nn.Module):
    """Wrapper for LaBraM model to extract EEG embeddings.

    Takes individual 10-second windows and produces embeddings.
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_times: int = 2000,  # 10s at 200Hz
        sfreq: float = 200,
        emb_size: int = 128,  # Reduced for memory
        n_layers: int = 2,  # Reduced for memory efficiency
        patch_size: int = 200,
        att_num_heads: int = 4,  # Reduced for memory
        dropout: float = 0.1,
    ):
        """Initialize LaBraM encoder.

        Args:
            n_channels: Number of EEG channels.
            n_times: Number of time samples per window.
            sfreq: Sampling frequency.
            emb_size: Embedding dimension.
            n_layers: Number of transformer layers.
            patch_size: Size of temporal patches.
            att_num_heads: Number of attention heads.
            dropout: Dropout probability.

        Raises:
            ImportError: If braindecode is not available.
        """
        if not LABRAM_AVAILABLE:
            error_msg = (
                f"LaBraM encoder requires braindecode, but it failed to import.\n"
                f"Error: {LABRAM_IMPORT_ERROR}\n"
                f"Try: pip install braindecode\n"
                f"Or use --eeg-encoder simplecnn instead."
            )
            logger.error(error_msg)
            raise ImportError(error_msg)

        super().__init__()

        self.n_channels = n_channels
        self.n_times = n_times
        self.emb_size = emb_size

        # Create LaBraM model
        self.model = Labram(
            n_chans=n_channels,
            n_times=n_times,
            sfreq=sfreq,
            n_outputs=emb_size,  # Use as embedding dim
            emb_size=emb_size,
            n_layers=n_layers,
            patch_size=patch_size,
            att_num_heads=att_num_heads,
            drop_prob=dropout,
            neural_tokenizer=True,
        )

        # Keep the final_layer as Linear(internal_dim, emb_size)
        # The braindecode Labram internal dimension (tied to patch_size)
        # differs from emb_size, so final_layer provides the projection.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from EEG windows.

        Args:
            x: EEG windows of shape (batch, channels, time).

        Returns:
            Embeddings of shape (batch, emb_size).
        """
        return self.model(x)


class EEGNetEncoder(nn.Module):
    """Wrapper for EEGNetv4 model to extract EEG embeddings.

    EEGNet is a compact CNN designed specifically for EEG classification.
    It uses depthwise and separable convolutions to reduce parameters
    while capturing spatial and temporal features.

    Original paper: Lawhern et al. 2018 "EEGNet: A Compact Convolutional
    Neural Network for EEG-based Brain-Computer Interfaces"
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_times: int = 2000,
        sfreq: float = 200,
        emb_size: int = 256,
        F1: int = 8,  # Number of temporal filters
        F2: int = 16,  # Number of pointwise filters
        D: int = 2,  # Depth multiplier for depthwise convolution
        dropout: float = 0.25,
    ):
        """Initialise EEGNet encoder.

        Args:
            n_channels: Number of EEG channels.
            n_times: Number of time samples per window.
            sfreq: Sampling frequency.
            emb_size: Output embedding dimension.
            F1: Number of temporal filters in first conv layer.
            F2: Number of pointwise filters.
            D: Depth multiplier (number of spatial filters per temporal filter).
            dropout: Dropout probability.

        Raises:
            ImportError: If braindecode EEGNet is not available.
        """
        if not EEGNET_AVAILABLE:
            error_msg = (
                f"EEGNet encoder requires braindecode, but it failed to import.\n"
                f"Error: {EEGNET_IMPORT_ERROR}\n"
                f"Try: pip install braindecode\n"
                f"Or use --eeg-encoder simplecnn instead."
            )
            logger.error(error_msg)
            raise ImportError(error_msg)

        super().__init__()

        self.n_channels = n_channels
        self.n_times = n_times
        self.emb_size = emb_size

        # Create EEGNet model
        # Use n_outputs for the number of classes; we'll replace final layer
        self.model = BraindecodeEEGNet(
            n_chans=n_channels,
            n_outputs=2,  # Dummy, will be replaced
            n_times=n_times,
            final_conv_length='auto',
            pool_mode='mean',
            F1=F1,
            D=D,
            F2=F2,
            drop_prob=dropout,
        )

        # Calculate the feature dimension before final layer
        # by doing a forward pass with the original model
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channels, n_times)
            # Get output after all conv layers but before final classification
            x = dummy_input
            for name, module in self.model.named_children():
                if name == 'final_layer':
                    break
                x = module(x)
            # Flatten to get feature dim
            feature_dim = x.numel()

        # Replace final layer with our own projection
        self.model.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, emb_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from EEG windows.

        Args:
            x: EEG windows of shape (batch, channels, time).

        Returns:
            Embeddings of shape (batch, emb_size).
        """
        return self.model(x)


class SimpleCNNEncoder(nn.Module):
    """Simple CNN-based EEG encoder as a baseline/fallback.

    Uses 1D convolutions along time axis to extract features.
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_times: int = 2000,
        emb_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_times = n_times
        self.emb_size = emb_size

        # Temporal convolutions
        self.conv_layers = nn.Sequential(
            # Layer 1: Extract basic temporal features
            nn.Conv1d(n_channels, 64, kernel_size=25, stride=5, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Conv1d(64, 128, kernel_size=15, stride=3, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3
            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 4
            nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
        )

        # Final projection
        self.projection = nn.Linear(256, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from EEG windows.

        Args:
            x: EEG windows of shape (batch, channels, time).

        Returns:
            Embeddings of shape (batch, emb_size).
        """
        # Conv layers
        x = self.conv_layers(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)

        # Project to embedding
        x = self.projection(x)  # (batch, emb_size)

        return x


class EEG2VecEncoder(nn.Module):
    """EEG2Vec encoder using CVAE with EEGNet backbone.

    Based on arxiv 2207.08002. Uses EEGNet-style convolutional layers as
    the feature encoder, then projects to a VAE latent space. Returns mu
    (mean) as the deterministic embedding at inference time.

    The logvar head is retained so that KL divergence can optionally be
    used as an auxiliary training loss.
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_times: int = 2000,
        emb_size: int = 256,
        F1: int = 8,
        F2: int = 16,
        D: int = 2,
        dropout: float = 0.25,
    ):
        """Initialise EEG2Vec encoder.

        Args:
            n_channels: Number of EEG channels.
            n_times: Number of time samples per window.
            emb_size: Output embedding dimension (latent space size).
            F1: Number of temporal filters in first conv layer.
            F2: Number of pointwise filters.
            D: Depth multiplier for depthwise convolution.
            dropout: Dropout probability.
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_times = n_times
        self.emb_size = emb_size

        # EEGNet backbone (conv layers for temporal + spatial features)
        # Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        # Depthwise spatial convolution
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        # Separable convolution (depthwise + pointwise)
        self.conv3_dw = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                                  groups=F1 * D, bias=False)
        self.conv3_pw = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Calculate flattened feature size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            dummy = self.pool1(F.elu(self.bn2(self.conv2(self.bn1(self.conv1(dummy))))))
            dummy = self.pool2(F.elu(self.bn3(self.conv3_pw(self.conv3_dw(dummy)))))
            feat_size = dummy.numel()

        # VAE projection heads
        self.fc_mu = nn.Linear(feat_size, emb_size)
        self.fc_logvar = nn.Linear(feat_size, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from EEG windows.

        Args:
            x: EEG windows of shape (batch, channels, time).

        Returns:
            Mu embeddings of shape (batch, emb_size).
        """
        # x: (batch, channels, time) -> (batch, 1, channels, time)
        x = x.unsqueeze(1)
        x = self.drop1(self.pool1(F.elu(self.bn2(self.conv2(self.bn1(self.conv1(x)))))))
        x = self.drop2(self.pool2(F.elu(self.bn3(self.conv3_pw(self.conv3_dw(x))))))
        x = x.flatten(1)
        mu = self.fc_mu(x)
        # logvar available via self.fc_logvar(x) for KL loss if needed
        return mu


class PretrainedLaBraMEncoder(nn.Module):
    """LaBraM-base with the published pretrained weights (vendored architecture).

    Input windows must be in the ``labram`` cache convention (microvolts / 100) with the
    19 standard channels in cache order; the channel names the weights were exported
    with are passed on every forward call, so the pretrained channel embeddings are the
    ones selected. ``pooling="mean"`` is the official fine-tuning head input (mean over
    the patch tokens through a LayerNorm that starts at identity); ``pooling="cls"`` is
    the [CLS] token after the pretrained final norm. With ``frozen=True`` no parameter
    trains and the model stays in evaluation mode whatever the parent's mode, so the
    features equal those written by ``shared.labram_pretrained.extract_features``.

    The weights file is written by ``python -m shared.labram_pretrained export-weights``
    (run in ``.venv-reve``) and lives outside git in ``outputs/``.
    """

    INPUT_CONVENTION = "labram"

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_times: int = 2000,
        emb_size: int = 200,
        pooling: str = "mean",
        frozen: bool = True,
        dropout: float = 0.0,
        drop_path_prob: float = 0.0,
        weights_path: Optional[Path] = None,
    ):
        super().__init__()
        from shared.labram_pretrained import EMBED_DIM, WEIGHTS_19CH_PATH  # constants only
        from shared.vendor.labram import Labram

        if pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', not {pooling!r}")
        if emb_size != EMBED_DIM:
            raise ValueError(f"LaBraM-base features are {EMBED_DIM}-dimensional; emb_size={emb_size} is not supported")
        path = Path(weights_path or WEIGHTS_19CH_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m shared.labram_pretrained export-weights` in .venv-reve first."
            )
        payload = torch.load(path, map_location="cpu", weights_only=True)  # tensors, dicts, lists and scalars only
        model_kwargs = dict(payload["model_kwargs"])
        if (model_kwargs["n_chans"], model_kwargs["n_times"]) != (n_channels, n_times):
            raise ValueError(
                f"weights are for {model_kwargs['n_chans']} channels x {model_kwargs['n_times']} samples, "
                f"not {n_channels} x {n_times}"
            )
        model_kwargs.update(drop_prob=dropout, drop_path_prob=drop_path_prob)
        with warnings.catch_warnings():
            # the vendored model notes that 19 channels are not its 128-channel layout;
            # the channel names are passed to every forward call, which is the supported path
            warnings.filterwarnings("ignore", message="Labram chs_info does not match")
            self.model = Labram(
                **model_kwargs,
                use_mean_pooling=(pooling == "mean"),
                chs_info=[{"ch_name": c} for c in payload["labram_ch_names"]],
            )
        conventions = payload["pooling"][pooling]
        state = {k: v for k, v in payload["state_dict"].items() if k not in conventions["drop_keys"]}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if set(missing) != set(conventions["fresh_keys"]) or unexpected:
            raise RuntimeError(f"unexpected weight mismatch: missing {missing}, unexpected {unexpected}")

        self.n_channels = n_channels
        self.n_times = n_times
        self.emb_size = emb_size
        self.pooling = pooling
        self.frozen = frozen
        self.ch_names = list(payload["labram_ch_names"])
        self.input_chans = list(payload["input_chans"])
        self.provenance = {k: payload[k] for k in ("hub_repo", "hub_revision", "hub_safetensors_sha256", "braindecode_version")}
        if frozen:
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True) -> "PretrainedLaBraMEncoder":
        super().train(mode)
        if self.frozen:
            self.model.eval()  # no dropout or drop path on frozen features
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, channels, time) in the labram convention -> (batch, emb_size)."""
        if x.ndim != 3 or x.shape[1:] != (self.n_channels, self.n_times):
            raise ValueError(f"expected (batch, {self.n_channels}, {self.n_times}), got {tuple(x.shape)}")
        if self.frozen:
            with torch.no_grad():
                return self.model(x, ch_names=self.ch_names)
        return self.model(x, ch_names=self.ch_names)


class PrecomputedFeatureEncoder(nn.Module):
    """Identity over per-window features computed outside training (REVE or the
    pretrained LaBraM feature files), so the window aggregator and classifier are
    shared with the raw-EEG encoders. Input (batch, emb_size); no parameters.
    """

    INPUT_CONVENTION = "precomputed"

    def __init__(self, emb_size: int, n_channels: int = N_CHANNELS, n_times: int = 2000):
        super().__init__()
        self.emb_size = emb_size
        # kept so callers can treat every encoder alike; a feature window has no time axis
        self.n_channels = n_channels
        self.n_times = n_times

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.emb_size:
            raise ValueError(f"expected (batch, {self.emb_size}) precomputed features, got {tuple(x.shape)}")
        return x


def flatten_windows(windows: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """``(batch, num_windows, *per_window)`` -> ``(batch * num_windows, *per_window)``
    plus the batch and window counts. Raw EEG windows are ``(channels, time)``;
    precomputed features are ``(dim,)``."""
    batch_size, num_windows = windows.shape[:2]
    return windows.reshape(batch_size * num_windows, *windows.shape[2:]), batch_size, num_windows


ENCODER_TYPES = ("labram", "eegnet", "simplecnn", "eeg2vec", "labram_pretrained", "precomputed")


def get_eeg_encoder(
    encoder_type: str = "labram",
    n_channels: int = N_CHANNELS,
    n_times: int = 2000,
    emb_size: int = 200,
    **kwargs,
) -> nn.Module:
    """Factory function to get EEG encoder by type.

    Args:
        encoder_type: One of ``ENCODER_TYPES``: 'labram' (architecture trained from
            scratch), 'eegnet', 'simplecnn', 'eeg2vec', 'labram_pretrained' (published
            weights, vendored architecture) or 'precomputed' (identity over features).
        n_channels: Number of EEG channels.
        n_times: Number of time samples per window.
        emb_size: Embedding dimension.
        **kwargs: Additional arguments for specific encoders.

    Returns:
        EEG encoder module.

    Raises:
        ValueError: If encoder_type is unknown.
        ImportError: If encoder dependencies are not available.
    """
    logger.info(f"Creating EEG encoder: {encoder_type}")

    if encoder_type == "labram_pretrained":
        return PretrainedLaBraMEncoder(
            n_channels=n_channels,
            n_times=n_times,
            emb_size=emb_size,
            **kwargs,
        )
    elif encoder_type == "precomputed":
        return PrecomputedFeatureEncoder(
            emb_size=emb_size,
            n_channels=n_channels,
            n_times=n_times,
            **kwargs,
        )
    elif encoder_type == "labram":
        if not LABRAM_AVAILABLE:
            logger.error(f"LaBraM requested but braindecode not available: {LABRAM_IMPORT_ERROR}")
            raise ImportError(
                f"LaBraM encoder requires braindecode.\n"
                f"Import error: {LABRAM_IMPORT_ERROR}\n"
                f"Use --eeg-encoder simplecnn as an alternative."
            )
        return LaBraMEncoder(
            n_channels=n_channels,
            n_times=n_times,
            emb_size=emb_size,
            **kwargs,
        )
    elif encoder_type == "eegnet":
        if not EEGNET_AVAILABLE:
            logger.error(f"EEGNet requested but braindecode not available: {EEGNET_IMPORT_ERROR}")
            raise ImportError(
                f"EEGNet encoder requires braindecode.\n"
                f"Import error: {EEGNET_IMPORT_ERROR}\n"
                f"Use --eeg-encoder simplecnn as an alternative."
            )
        return EEGNetEncoder(
            n_channels=n_channels,
            n_times=n_times,
            emb_size=emb_size,
            **kwargs,
        )
    elif encoder_type == "simplecnn":
        return SimpleCNNEncoder(
            n_channels=n_channels,
            n_times=n_times,
            emb_size=emb_size,
            **kwargs,
        )
    elif encoder_type == "eeg2vec":
        return EEG2VecEncoder(
            n_channels=n_channels,
            n_times=n_times,
            emb_size=emb_size,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown encoder type: {encoder_type}. "
            f"Available: {', '.join(ENCODER_TYPES)}"
        )


def is_labram_available() -> bool:
    """Check if LaBraM encoder is available."""
    return LABRAM_AVAILABLE


def get_labram_import_error() -> Optional[str]:
    """Get the LaBraM import error message if it failed to import."""
    return LABRAM_IMPORT_ERROR


def test_encoders():
    """Test EEG encoder implementations."""
    print("Testing EEG encoders...")
    print(f"LaBraM available: {LABRAM_AVAILABLE}")
    if not LABRAM_AVAILABLE:
        print(f"LaBraM import error: {LABRAM_IMPORT_ERROR}")
    print(f"EEGNet available: {EEGNET_AVAILABLE}")
    if not EEGNET_AVAILABLE:
        print(f"EEGNet import error: {EEGNET_IMPORT_ERROR}")

    n_channels = N_CHANNELS
    n_times = 2000
    batch_size = 4

    x = torch.randn(batch_size, n_channels, n_times)

    # Test LaBraM encoder (if available)
    if LABRAM_AVAILABLE:
        print("\nTesting LaBraM encoder:")
        try:
            labram = LaBraMEncoder(n_channels=n_channels, n_times=n_times, emb_size=200)
            out = labram(x)
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {out.shape}")
            print(f"  Parameters: {sum(p.numel() for p in labram.parameters()):,}")
        except Exception as e:
            print(f"  LaBraM test failed: {e}")
    else:
        print("\nSkipping LaBraM encoder test (not available)")

    # Test EEGNet encoder (if available)
    if EEGNET_AVAILABLE:
        print("\nTesting EEGNet encoder:")
        try:
            eegnet = EEGNetEncoder(n_channels=n_channels, n_times=n_times, emb_size=256)
            out = eegnet(x)
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {out.shape}")
            print(f"  Parameters: {sum(p.numel() for p in eegnet.parameters()):,}")
        except Exception as e:
            print(f"  EEGNet test failed: {e}")
    else:
        print("\nSkipping EEGNet encoder test (not available)")

    # Test SimpleCNN encoder
    print("\nTesting SimpleCNN encoder:")
    cnn = SimpleCNNEncoder(n_channels=n_channels, n_times=n_times, emb_size=256)
    out = cnn(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in cnn.parameters()):,}")

    # Test EEG2Vec encoder
    print("\nTesting EEG2Vec encoder:")
    eeg2vec = EEG2VecEncoder(n_channels=n_channels, n_times=n_times, emb_size=256)
    out = eeg2vec(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in eeg2vec.parameters()):,}")

    print("\nEncoder tests complete.")


if __name__ == "__main__":
    test_encoders()
