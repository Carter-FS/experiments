"""EEG data loading, preprocessing, and windowing pipeline."""

import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd

from .config import (
    CSV_PATH,
    EEG_CONFIG,
    EEG_DIR,
    MAX_WINDOWS,
)

# Suppress MNE verbose output
mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger("exp2")


# ============================================================================
# EEG Quality Metrics
# ============================================================================

def compute_snr(data: np.ndarray, sr: float, noise_band: Tuple[float, float] = (55.0, 65.0)) -> np.ndarray:
    """Estimate signal-to-noise ratio per channel.

    Uses power spectral density to estimate signal (0.5-40Hz) vs noise (55-65Hz).

    Args:
        data: EEG data [channels x samples].
        sr: Sample rate.
        noise_band: Frequency band for noise estimation.

    Returns:
        SNR values per channel (in dB).
    """
    from scipy import signal as scipy_signal

    n_channels = data.shape[0]
    snr_values = np.zeros(n_channels)

    for ch in range(n_channels):
        freqs, psd = scipy_signal.welch(data[ch], sr, nperseg=min(1024, len(data[ch])))

        # Signal power: 0.5-40 Hz (typical EEG range)
        signal_mask = (freqs >= 0.5) & (freqs <= 40.0)
        signal_power = np.mean(psd[signal_mask]) if signal_mask.any() else 1e-10

        # Noise power: 55-65 Hz (high-frequency noise)
        noise_mask = (freqs >= noise_band[0]) & (freqs <= noise_band[1])
        noise_power = np.mean(psd[noise_mask]) if noise_mask.any() else 1e-10

        # SNR in dB
        snr_values[ch] = 10 * np.log10(signal_power / max(noise_power, 1e-10))

    return snr_values


def detect_artifacts(data: np.ndarray, threshold_uv: float = 500.0) -> Dict:
    """Detect high-amplitude artifacts in EEG data.

    Args:
        data: EEG data [channels x samples] in Volts.
        threshold_uv: Threshold in microvolts for artifact detection.

    Returns:
        Dict with artifact statistics per channel.
    """
    # Convert to microvolts (EDF is typically in Volts)
    data_uv = data * 1e6

    n_channels, n_samples = data.shape
    artifact_stats = {
        "per_channel_artifact_ratio": [],
        "overall_artifact_ratio": 0.0,
        "max_amplitude_uv": 0.0,
        "channels_with_artifacts": 0,
    }

    total_artifacts = 0
    for ch in range(n_channels):
        ch_data = np.abs(data_uv[ch])
        artifact_mask = ch_data > threshold_uv
        artifact_ratio = artifact_mask.mean()
        artifact_stats["per_channel_artifact_ratio"].append(float(artifact_ratio))

        if artifact_ratio > 0.01:  # >1% artifacts
            artifact_stats["channels_with_artifacts"] += 1

        total_artifacts += artifact_mask.sum()

    artifact_stats["overall_artifact_ratio"] = total_artifacts / (n_channels * n_samples)
    artifact_stats["max_amplitude_uv"] = float(np.abs(data_uv).max())

    return artifact_stats


def detect_flatlines(data: np.ndarray, window_samples: int = 200, std_threshold: float = 0.1) -> Dict:
    """Detect flatline segments (near-zero variance) in EEG data.

    Args:
        data: EEG data [channels x samples].
        window_samples: Window size for variance computation.
        std_threshold: Threshold for flatline detection (in microvolts).

    Returns:
        Dict with flatline statistics.
    """
    data_uv = data * 1e6  # Convert to microvolts
    n_channels, n_samples = data.shape

    flatline_stats = {
        "per_channel_flatline_ratio": [],
        "overall_flatline_ratio": 0.0,
        "channels_with_flatlines": 0,
    }

    total_flatline_samples = 0

    for ch in range(n_channels):
        ch_data = data_uv[ch]
        n_windows = n_samples // window_samples

        flatline_windows = 0
        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            window_std = np.std(ch_data[start:end])
            if window_std < std_threshold:
                flatline_windows += 1

        flatline_ratio = flatline_windows / max(n_windows, 1)
        flatline_stats["per_channel_flatline_ratio"].append(float(flatline_ratio))

        if flatline_ratio > 0.05:  # >5% flatlines
            flatline_stats["channels_with_flatlines"] += 1

        total_flatline_samples += flatline_windows * window_samples

    flatline_stats["overall_flatline_ratio"] = total_flatline_samples / (n_channels * n_samples)

    return flatline_stats


def compute_channel_correlation(data: np.ndarray) -> Dict:
    """Compute inter-channel correlation statistics.

    High correlation may indicate referencing issues or global artifacts.
    Low correlation may indicate noisy/disconnected channels.

    Args:
        data: EEG data [channels x samples].

    Returns:
        Dict with correlation statistics.
    """
    n_channels = data.shape[0]

    if n_channels < 2:
        return {"mean_correlation": 0.0, "low_correlation_channels": 0}

    # Compute correlation matrix
    corr_matrix = np.corrcoef(data)

    # Mask diagonal
    np.fill_diagonal(corr_matrix, np.nan)

    # Mean absolute correlation per channel
    mean_corr_per_channel = np.nanmean(np.abs(corr_matrix), axis=1)

    # Low correlation channels (may be noisy or disconnected)
    low_corr_threshold = 0.1
    low_corr_channels = (mean_corr_per_channel < low_corr_threshold).sum()

    # Overall statistics
    overall_mean = np.nanmean(np.abs(corr_matrix))

    return {
        "mean_correlation": float(overall_mean),
        "per_channel_mean_correlation": mean_corr_per_channel.tolist(),
        "low_correlation_channels": int(low_corr_channels),
    }


def compute_quality_score(
    snr_values: np.ndarray,
    artifact_stats: Dict,
    flatline_stats: Dict,
    correlation_stats: Dict,
) -> Dict:
    """Compute overall quality score from individual metrics.

    Args:
        snr_values: SNR per channel.
        artifact_stats: Artifact detection results.
        flatline_stats: Flatline detection results.
        correlation_stats: Correlation statistics.

    Returns:
        Dict with quality scores and overall score.
    """
    # SNR score (0-1, higher is better)
    mean_snr = np.mean(snr_values)
    snr_score = np.clip((mean_snr + 10) / 30, 0, 1)  # Map -10dB to 20dB -> 0 to 1

    # Artifact score (0-1, lower artifact ratio is better)
    artifact_score = 1.0 - min(artifact_stats["overall_artifact_ratio"] * 10, 1.0)

    # Flatline score (0-1, lower flatline ratio is better)
    flatline_score = 1.0 - min(flatline_stats["overall_flatline_ratio"] * 5, 1.0)

    # Correlation score (moderate correlation is best)
    mean_corr = correlation_stats["mean_correlation"]
    # Penalise very low (<0.1) or very high (>0.8) correlation
    if mean_corr < 0.1:
        corr_score = mean_corr / 0.1
    elif mean_corr > 0.8:
        corr_score = (1.0 - mean_corr) / 0.2
    else:
        corr_score = 1.0

    # Overall quality score (weighted average)
    weights = {"snr": 0.3, "artifact": 0.3, "flatline": 0.2, "correlation": 0.2}
    overall_score = (
        weights["snr"] * snr_score +
        weights["artifact"] * artifact_score +
        weights["flatline"] * flatline_score +
        weights["correlation"] * corr_score
    )

    return {
        "snr_score": float(snr_score),
        "artifact_score": float(artifact_score),
        "flatline_score": float(flatline_score),
        "correlation_score": float(corr_score),
        "overall_quality_score": float(overall_score),
        "mean_snr_db": float(mean_snr),
    }


# ============================================================================
# EEG Normalisation Functions
# ============================================================================

def zscore_normalise_global(
    data: np.ndarray,
    mean: np.ndarray = None,
    std: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply global z-score normalisation per channel.

    Args:
        data: EEG data [channels x samples].
        mean: Pre-computed channel means (for test data).
        std: Pre-computed channel stds (for test data).

    Returns:
        Tuple of (normalised_data, mean, std).
    """
    if mean is None:
        mean = data.mean(axis=1, keepdims=True)
    if std is None:
        std = data.std(axis=1, keepdims=True)

    # Avoid division by zero
    std = np.where(std < 1e-10, 1.0, std)

    normalised = (data - mean) / std
    return normalised, mean.squeeze(), std.squeeze()


def zscore_normalise_window(data: np.ndarray) -> np.ndarray:
    """Apply per-window z-score normalisation.

    Removes DC offset and scales each window independently.
    Useful for making the model invariant to absolute amplitude.

    Args:
        data: EEG data [channels x samples] or windows [n_windows, channels, samples].

    Returns:
        Normalised data with same shape.
    """
    if data.ndim == 2:
        # Single window or full signal
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)
        return (data - mean) / std
    elif data.ndim == 3:
        # Multiple windows: [n_windows, channels, samples]
        normalised = np.zeros_like(data)
        for i in range(data.shape[0]):
            normalised[i] = zscore_normalise_window(data[i])
        return normalised
    else:
        raise ValueError(f"Expected 2D or 3D data, got {data.ndim}D")


def robust_normalise(
    data: np.ndarray,
    median: np.ndarray = None,
    iqr: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply robust normalisation using median and IQR.

    More robust to artifacts than z-score normalisation.

    Args:
        data: EEG data [channels x samples].
        median: Pre-computed channel medians (for test data).
        iqr: Pre-computed channel IQRs (for test data).

    Returns:
        Tuple of (normalised_data, median, iqr).
    """
    if median is None:
        median = np.median(data, axis=1, keepdims=True)
    if iqr is None:
        q75 = np.percentile(data, 75, axis=1, keepdims=True)
        q25 = np.percentile(data, 25, axis=1, keepdims=True)
        iqr = q75 - q25

    # Avoid division by zero
    iqr = np.where(iqr < 1e-10, 1.0, iqr)

    normalised = (data - median) / iqr
    return normalised, median.squeeze(), iqr.squeeze()


def clip_amplitude(data: np.ndarray, n_std: float = 5.0) -> np.ndarray:
    """Clip extreme values to reduce artifact impact.

    Args:
        data: EEG data [channels x samples].
        n_std: Number of standard deviations to clip to.

    Returns:
        Clipped data.
    """
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)

    lower = mean - n_std * std
    upper = mean + n_std * std

    return np.clip(data, lower, upper)


class EEGNormaliser:
    """EEG normalisation with fit/transform interface.

    Supports global z-score, per-window z-score, and robust normalisation.
    """

    def __init__(
        self,
        method: str = "zscore",
        clip_std: float = None,
    ):
        """Initialise normaliser.

        Args:
            method: Normalisation method ('zscore', 'window_zscore', 'robust', 'none').
            clip_std: If set, clip values beyond n standard deviations.
        """
        self.method = method
        self.clip_std = clip_std

        # Fitted statistics (for zscore/robust)
        self.mean_ = None
        self.std_ = None
        self.median_ = None
        self.iqr_ = None
        self.is_fitted = False

    def fit(self, data: np.ndarray):
        """Fit normaliser on training data.

        Args:
            data: EEG data [channels x samples] or stacked windows.
        """
        if data.ndim == 3:
            # Flatten windows to [channels, total_samples]
            n_windows, n_channels, n_samples = data.shape
            data = data.transpose(1, 0, 2).reshape(n_channels, -1)

        if self.method == "zscore":
            self.mean_ = data.mean(axis=1)
            self.std_ = data.std(axis=1)
            self.std_ = np.where(self.std_ < 1e-10, 1.0, self.std_)
        elif self.method == "robust":
            self.median_ = np.median(data, axis=1)
            q75 = np.percentile(data, 75, axis=1)
            q25 = np.percentile(data, 25, axis=1)
            self.iqr_ = q75 - q25
            self.iqr_ = np.where(self.iqr_ < 1e-10, 1.0, self.iqr_)

        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted statistics.

        Args:
            data: EEG data [channels x samples] or windows [n_windows, channels, samples].

        Returns:
            Normalised data.
        """
        original_shape = data.shape
        is_windows = data.ndim == 3

        if is_windows:
            # Process each window
            n_windows, n_channels, n_samples = data.shape
            normalised = np.zeros_like(data)
            for i in range(n_windows):
                normalised[i] = self._transform_single(data[i])
            return normalised
        else:
            return self._transform_single(data)

    def _transform_single(self, data: np.ndarray) -> np.ndarray:
        """Transform a single 2D array [channels x samples]."""
        # Optional clipping first
        if self.clip_std is not None:
            data = clip_amplitude(data, self.clip_std)

        if self.method == "none":
            return data
        elif self.method == "window_zscore":
            return zscore_normalise_window(data)
        elif self.method == "zscore":
            if not self.is_fitted:
                raise ValueError("Normaliser not fitted. Call fit() first.")
            mean = self.mean_.reshape(-1, 1)
            std = self.std_.reshape(-1, 1)
            return (data - mean) / std
        elif self.method == "robust":
            if not self.is_fitted:
                raise ValueError("Normaliser not fitted. Call fit() first.")
            median = self.median_.reshape(-1, 1)
            iqr = self.iqr_.reshape(-1, 1)
            return (data - median) / iqr
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)


# ============================================================================
# File Operations
# ============================================================================

def build_patient_eeg_map(eeg_dir: Path = EEG_DIR) -> Dict[str, Path]:
    """Build mapping from patient ID to EEG file path.

    Superseded for the experiments by ``shared.eeg_cache.map_patients_to_edf``
    (same exact-id rule, recursive scan, all file-name forms); kept for the
    exploratory quality analysis.

    Returns:
        Dictionary mapping patient ID (string) to EEG file path.
    """
    patient_map = {}

    for edf_file in eeg_dir.glob("*.edf"):
        pid = extract_patient_id(edf_file.name)
        if pid:
            # If multiple files per patient, keep the first one
            if pid not in patient_map:
                patient_map[pid] = edf_file

    return patient_map


from .channels import (  # noqa: E402,F401  re-exported; the constants live in a module without MNE
    N_CHANNELS, OLD_TO_MODERN_NAME, STD_19_TEN_TWENTY, extract_patient_id, normalise_channel_name,
)


def filter_to_standard_19(raw) -> "mne.io.BaseRaw":
    """Keep only the 19 standard 10-20 EEG channels, in canonical order.

    Step 1: rename every channel to its canonical uppercase modern name
    (applies OLD->MODERN MCN renaming: T3->T7, T4->T8, T5->P7, T6->P8).
    Step 2: pick the 19 standard channels in canonical order, dropping
    A1/A2/ECG/EMG/PG/sop and any other non-standard channel.

    Raises a ValueError if fewer than 19 of the standard channels are
    present after renaming (the recording is unusable for the
    19-channel encoder).
    """
    name_map: Dict[str, str] = {}
    seen_targets: set = set()
    for ch in raw.ch_names:
        canonical = normalise_channel_name(ch)
        if canonical in seen_targets:
            # Defensive: if a duplicate target name appears (e.g. both
            # 't3' and 'T7' present), keep the first and leave this
            # channel unrenamed (it will be dropped at pick time).
            name_map[ch] = ch
        else:
            name_map[ch] = canonical
            seen_targets.add(canonical)
    raw.rename_channels(name_map)
    present_canonical = set(raw.ch_names)
    missing = [std for std in STD_19_TEN_TWENTY if std not in present_canonical]
    if missing:
        raise ValueError(
            f"Recording is missing {len(missing)} of the 19 standard 10-20 channels: "
            f"{missing}. Present after renaming: {sorted(present_canonical)}"
        )
    # pick + reorder in one call: passing an ordered list to .pick()
    # both restricts and reorders.
    raw.pick(list(STD_19_TEN_TWENTY))
    return raw


def read_edf(filepath: Path) -> "mne.io.BaseRaw":
    """Read an EDF file with preload, trying the annotation encodings in turn.

    Raises:
        ValueError: If the file cannot be read with any encoding.
    """
    filepath = Path(filepath)
    raw = None
    successful_encoding = None
    for encoding in ("utf-8", "latin1", "iso-8859-1"):
        try:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False, encoding=encoding)
            successful_encoding = encoding
            break
        except Exception as e:
            logger.debug(f"Failed to load {filepath.name} with encoding {encoding}: {e}")
            continue
    if raw is None:
        logger.error(f"Could not load EDF file with any encoding: {filepath}")
        raise ValueError(f"Could not load EDF file: {filepath}")
    if successful_encoding != "utf-8":
        logger.debug(f"Loaded {filepath.name} with fallback encoding: {successful_encoding}")
    return raw


def load_edf(
    filepath: Path,
    target_sr: int = 200,
    use_standard_19: bool = False,
) -> Tuple[np.ndarray, float, List[str]]:
    """Load EDF file and return raw data.

    Args:
        filepath: Path to .edf file.
        target_sr: Target sample rate for resampling.
        use_standard_19: If True, restrict to the 19 standard 10-20 EEG
            channels (with old-MCN name renaming applied), per Stage C
            cohort-portable preprocessing. If False, retain the legacy
            behaviour of keeping every channel typed as ``eeg`` by the
            EDF header (preserves Stage A reproducibility).

    Returns:
        Tuple of (data array [channels x samples], sample rate, channel names).
    """
    raw = read_edf(filepath)

    if use_standard_19:
        # Stage C / cohort-portable path: name-filter to the 19 standard
        # 10-20 EEG channels. Drops EMG/PG/A/ECG/sop and any other
        # non-EEG channels regardless of how the EDF types them.
        raw = filter_to_standard_19(raw)
    else:
        # Legacy path: keep everything the EDF tags as eeg (matches
        # Stage A reproducibility for the 27-channel Alfred-only runs).
        try:
            raw.pick_types(eeg=True, exclude=[])
        except Exception as e:
            logger.debug(f"Could not filter EEG channels for {filepath.name}: {e}, keeping all channels")

    original_sr = raw.info["sfreq"]

    # Resample if needed
    if original_sr != target_sr:
        logger.debug(f"Resampling {filepath.name} from {original_sr}Hz to {target_sr}Hz")
        raw.resample(target_sr, verbose=False)

    data = raw.get_data()  # Shape: (n_channels, n_samples)
    ch_names = raw.ch_names

    return data, target_sr, ch_names


def apply_filters(
    data: np.ndarray,
    sr: float,
    lowcut: float = 0.1,
    highcut: float = 75.0,
    notch_freq: float = 50.0,
) -> np.ndarray:
    """Apply bandpass and notch filters to EEG data.

    Args:
        data: EEG data array [channels x samples].
        sr: Sample rate.
        lowcut: Low frequency cutoff for bandpass.
        highcut: High frequency cutoff for bandpass.
        notch_freq: Frequency for notch filter (power line noise).

    Returns:
        Filtered data array.
    """
    n_channels = data.shape[0]

    # Create MNE RawArray for filtering
    info = mne.create_info(
        ch_names=[f"EEG{i}" for i in range(n_channels)],
        sfreq=sr,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    # Apply bandpass filter
    raw.filter(lowcut, highcut, verbose=False)

    # Apply notch filter for power line noise
    raw.notch_filter(notch_freq, verbose=False)

    return raw.get_data()


def extract_time_window(
    data: np.ndarray,
    sr: float,
    skip_start_sec: float = 300,
    use_duration_sec: float = 1200,
    min_duration_sec: float = 600,
) -> Optional[np.ndarray]:
    """Extract the relevant time window from EEG data.

    Per OBJECTIVES.md:
    - Skip first 5 minutes
    - Use up to 20 minutes of data
    - Reject EEGs shorter than 10 minutes total

    Args:
        data: EEG data [channels x samples].
        sr: Sample rate.
        skip_start_sec: Seconds to skip at start (5 min = 300s).
        use_duration_sec: Max duration to use (20 min = 1200s).
        min_duration_sec: Minimum total duration required (10 min = 600s).

    Returns:
        Extracted data or None if too short.
    """
    n_samples = data.shape[1]
    total_duration = n_samples / sr

    # Reject if total duration < minimum
    if total_duration < min_duration_sec:
        return None

    # Calculate start and end samples
    start_sample = int(skip_start_sec * sr)

    # If the EEG is shorter than skip_start + some data, adjust
    if start_sample >= n_samples:
        # Not enough data after skipping
        return None

    # Calculate end sample
    end_sample = int(start_sample + use_duration_sec * sr)
    end_sample = min(end_sample, n_samples)

    return data[:, start_sample:end_sample]


def create_windows(
    data: np.ndarray,
    sr: float,
    window_sec: float = 10,
    max_windows: int = MAX_WINDOWS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split EEG data into fixed-size windows with padding.

    Args:
        data: EEG data [channels x samples].
        sr: Sample rate.
        window_sec: Window duration in seconds.
        max_windows: Maximum number of windows.

    Returns:
        Tuple of:
        - windows: Array of shape [max_windows, channels, samples_per_window]
        - padding_mask: Boolean array [max_windows], True for padded windows
    """
    n_channels, n_samples = data.shape
    samples_per_window = int(window_sec * sr)

    # Calculate actual number of windows
    n_windows = n_samples // samples_per_window
    n_windows = min(n_windows, max_windows)

    # Initialize output arrays
    windows = np.zeros((max_windows, n_channels, samples_per_window), dtype=np.float32)
    padding_mask = np.ones(max_windows, dtype=bool)  # True = padded

    # Fill in actual windows
    for i in range(n_windows):
        start = i * samples_per_window
        end = start + samples_per_window
        windows[i] = data[:, start:end]
        padding_mask[i] = False  # Not padded

    return windows, padding_mask


class EEGPreprocessor:
    """EEG preprocessing pipeline for loading, filtering, and windowing."""

    def __init__(
        self,
        target_sr: int = EEG_CONFIG["target_sr"],
        min_duration_sec: float = EEG_CONFIG["min_duration_sec"],
        skip_start_sec: float = EEG_CONFIG["skip_start_sec"],
        use_duration_sec: float = EEG_CONFIG["use_duration_sec"],
        window_sec: float = EEG_CONFIG["window_sec"],
        lowcut: float = EEG_CONFIG["lowcut"],
        highcut: float = EEG_CONFIG["highcut"],
        notch_freq: float = EEG_CONFIG["notch_freq"],
        compute_quality: bool = False,
        normalisation: str = "none",
        clip_std: float = None,
        use_standard_19: bool = False,
    ):
        """Initialise EEG preprocessor.

        Args:
            target_sr: Target sample rate.
            min_duration_sec: Minimum EEG duration required.
            skip_start_sec: Seconds to skip at start.
            use_duration_sec: Maximum duration to use.
            window_sec: Window size in seconds.
            lowcut: Low frequency cutoff for bandpass.
            highcut: High frequency cutoff for bandpass.
            notch_freq: Notch filter frequency.
            compute_quality: Whether to compute quality metrics.
            normalisation: Normalisation method ('none', 'zscore', 'window_zscore', 'robust').
            clip_std: If set, clip values beyond n standard deviations.
            use_standard_19: If True, restrict to the 19 standard 10-20
                EEG channels (Stage C cohort-portable mode; required for
                HEP1 external validation).
        """
        self.target_sr = target_sr
        self.min_duration_sec = min_duration_sec
        self.skip_start_sec = skip_start_sec
        self.use_duration_sec = use_duration_sec
        self.window_sec = window_sec
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.compute_quality = compute_quality
        self.normalisation = normalisation
        self.clip_std = clip_std
        self.use_standard_19 = use_standard_19

        self.samples_per_window = int(window_sec * target_sr)
        self.max_windows = int(use_duration_sec / window_sec)

        # Normaliser (will be fitted per-patient or globally)
        self.normaliser = None
        if normalisation != "none":
            self.normaliser = EEGNormaliser(method=normalisation, clip_std=clip_std)

    def process(
        self,
        edf_path: Path,
        return_quality: bool = None,
    ) -> Optional[Union[Tuple[np.ndarray, np.ndarray, int], Tuple[np.ndarray, np.ndarray, int, Dict]]]:
        """Process a single EEG file.

        Args:
            edf_path: Path to .edf file.
            return_quality: Whether to compute and return quality metrics.
                           Defaults to self.compute_quality.

        Returns:
            If return_quality is False:
                Tuple of (windows, padding_mask, n_channels) or None if invalid.
            If return_quality is True:
                Tuple of (windows, padding_mask, n_channels, quality_metrics) or None.
        """
        if return_quality is None:
            return_quality = self.compute_quality

        logger.debug(f"Processing EDF: {edf_path.name}")

        # Load EDF
        data, sr, ch_names = load_edf(edf_path, self.target_sr, use_standard_19=self.use_standard_19)
        n_channels = data.shape[0]
        total_duration_sec = data.shape[1] / sr
        logger.debug(f"  Loaded: {n_channels} channels, {total_duration_sec:.1f}s duration")

        # Apply filters
        data = apply_filters(
            data, sr,
            lowcut=self.lowcut,
            highcut=self.highcut,
            notch_freq=self.notch_freq,
        )
        logger.debug(f"  Applied filters: {self.lowcut}-{self.highcut}Hz bandpass, {self.notch_freq}Hz notch")

        # Compute quality metrics on filtered data (before time extraction)
        quality_metrics = None
        if return_quality:
            quality_metrics = self._compute_quality_metrics(data, sr)

        # Extract time window
        data = extract_time_window(
            data, sr,
            skip_start_sec=self.skip_start_sec,
            use_duration_sec=self.use_duration_sec,
            min_duration_sec=self.min_duration_sec,
        )

        if data is None:
            logger.debug(f"  Rejected: duration {total_duration_sec:.1f}s < minimum {self.min_duration_sec}s")
            return None

        extracted_duration = data.shape[1] / sr
        logger.debug(f"  Extracted: {extracted_duration:.1f}s after skipping first {self.skip_start_sec}s")

        # Create windows
        windows, padding_mask = create_windows(
            data, sr,
            window_sec=self.window_sec,
            max_windows=self.max_windows,
        )

        n_valid_windows = (~padding_mask).sum()
        logger.debug(f"  Windows: {n_valid_windows}/{len(padding_mask)} valid ({self.window_sec}s each)")

        # Apply normalisation to valid windows only
        if self.normaliser is not None and n_valid_windows > 0:
            valid_windows = windows[~padding_mask]
            if self.normalisation == "window_zscore":
                # Per-window normalisation doesn't need fitting
                windows[~padding_mask] = self.normaliser.transform(valid_windows)
            else:
                # Fit on valid windows and transform
                windows[~padding_mask] = self.normaliser.fit_transform(valid_windows)
            logger.debug(f"  Applied {self.normalisation} normalisation")

        if return_quality and quality_metrics is not None:
            quality_metrics["n_valid_windows"] = int(n_valid_windows)
            quality_metrics["n_channels"] = n_channels
            quality_metrics["extracted_duration_sec"] = float(extracted_duration)
            return windows, padding_mask, n_channels, quality_metrics

        return windows, padding_mask, n_channels

    def _compute_quality_metrics(self, data: np.ndarray, sr: float) -> Dict:
        """Compute quality metrics for EEG data.

        Args:
            data: Filtered EEG data [channels x samples].
            sr: Sample rate.

        Returns:
            Dict with quality metrics.
        """
        # Compute individual metrics
        snr_values = compute_snr(data, sr)
        artifact_stats = detect_artifacts(data)
        flatline_stats = detect_flatlines(data)
        correlation_stats = compute_channel_correlation(data)

        # Compute overall quality score
        quality_score = compute_quality_score(
            snr_values, artifact_stats, flatline_stats, correlation_stats
        )

        return {
            "snr_per_channel": snr_values.tolist(),
            "artifact_stats": artifact_stats,
            "flatline_stats": flatline_stats,
            "correlation_stats": correlation_stats,
            "quality_scores": quality_score,
        }


def test_pipeline():
    """Test the EEG pipeline on a sample file."""
    print("Testing EEG pipeline...")

    # Patient-EEG pairs by exact id (the cache builder's rule)
    patient_map = build_patient_eeg_map()
    print(f"Found {len(patient_map)} patients with an EEG file")

    if not patient_map:
        print("No EEG files found!")
        return

    # Process the first recording (ids are not printed)
    eeg_path = next(iter(sorted(patient_map.values())))
    preprocessor = EEGPreprocessor()
    result = preprocessor.process(Path(eeg_path))

    if result is None:
        print("EEG too short, skipping")
        return

    windows, padding_mask, n_channels = result
    n_valid_windows = (~padding_mask).sum()

    print(f"\nResults:")
    print(f"  Channels: {n_channels}")
    print(f"  Window shape: {windows.shape}")
    print(f"  Valid windows: {n_valid_windows} / {len(padding_mask)}")
    print(f"  Samples per window: {windows.shape[2]}")
    print(f"  Duration per window: {windows.shape[2] / preprocessor.target_sr}s")


if __name__ == "__main__":
    test_pipeline()
