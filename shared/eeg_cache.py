"""EEG cache, version 2: the one builder and the one loader for every EEG experiment.

Build once per cohort (CPU; about one to two hours for the Melbourne cohort)::

    python -m shared.eeg_cache build --cohort alfred   # outputs/eeg_cache/eeg19_v2_alfred.pkl
    python -m shared.eeg_cache build --cohort hep      # outputs/eeg_cache/eeg19_v2_hep.pkl
    python -m shared.eeg_cache stats outputs/eeg_cache/eeg19_v2_alfred.pkl

Each recording is stored as the 19 standard 10-20 channels in canonical order (old
names T3/T4/T5/T6 mapped to T7/T8/P7/P8; every other channel dropped), resampled to
200 Hz, band-passed 0.1-75 Hz, notch-filtered at the cohort's mains frequency, in
microvolts and unnormalised. The leading flat segment is skipped, then the first 300 s,
and the next 1200 s are split into 10 s windows (at most 120, zero-padded, with a
padding mask). Recordings missing any of the 19 channels, entirely flat, or shorter
than 600 s after the leading flat segment are skipped and counted in the sidecar
``<cache>.meta.json``, which holds aggregate statistics only.

Load::

    from shared.eeg_cache import load_cache
    eeg = load_cache(path, convention="zscore_window")   # {pid: (windows, padding_mask)}

Conventions, applied to the valid windows at load time:

- ``zscore_window``: per window and channel, (x - mean) / max(sd, 1e-6) over the
  window's samples, in microvolts (the supervisor's ``zscore_norm_epoch``).
- ``labram``: microvolts divided by 100 (the pretrained LaBraM input scale).
- ``raw_uv``: microvolts, unchanged.

A pickle without the version-2 header (the superseded ``processed_eeg*.pkl`` caches)
is refused unless ``allow_legacy=True`` is passed explicitly.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import pickle
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from exp2_fusion.config import EEG_CONFIG, MAX_WINDOWS
from exp2_fusion.eeg_pipeline import (
    STD_19_TEN_TWENTY,
    apply_filters,
    create_windows,
    extract_patient_id,
    extract_time_window,
    filter_to_standard_19,
    read_edf,
)
from shared.hep_cohort import EXPERIMENTS_ROOT, find_asm_data_dir

logger = logging.getLogger(__name__)

CACHE_VERSION = 2
CONVENTIONS = ("zscore_window", "labram", "raw_uv")
ZSCORE_SD_FLOOR_UV = 1e-6   # supervisor's zscore_norm_epoch sigma_floor
LABRAM_SCALE_UV = 100.0     # official LaBraM: microvolts / 100
FLAT_CHUNK_S = 100.0        # supervisor's find_signal_start chunk
FLAT_SD_UV = 1e-2           # supervisor's threshold (1e-8 V) in microvolts
CH_NAMES: Tuple[str, ...] = tuple(STD_19_TEN_TWENTY)
N_CHANNELS = len(CH_NAMES)
TARGET_SFREQ = float(EEG_CONFIG["target_sr"])
SAMPLES_PER_WINDOW = int(EEG_CONFIG["window_sec"] * EEG_CONFIG["target_sr"])

COHORTS: Dict[str, Dict[str, object]] = {
    "alfred": {"notch_hz": 50.0, "csv": "alfred_1st_regimen.csv", "edf_dir": ("Alfred",),
               "pid_col": "pid", "outcomes": (1, 2)},
    "hep": {"notch_hz": 60.0, "csv": "hep_1st_regimen.csv", "edf_dir": ("HEP", "EEG"),
            "pid_col": "patient", "outcomes": (0, 1)},
}
CACHE_DIR = EXPERIMENTS_ROOT / "outputs" / "eeg_cache"
CACHE_PATHS = {"alfred": CACHE_DIR / "eeg19_v2_alfred.pkl", "hep": CACHE_DIR / "eeg19_v2_hep.pkl"}
SKIP_REASONS = ("missing_channels", "units", "flat", "too_short", "non_finite", "read_error")
# EDF physical dimensions MNE scales to volts (lower case); "v" is already volts.
VOLTAGE_UNITS = {"v", "mv", "uv", "\u00b5v", "\u03bcv"}


# ---------------------------------------------------------------------------
# Per-recording processing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Recording:
    """One processed recording: windows in microvolts plus the padding mask."""
    windows_uv: np.ndarray      # float32 (MAX_WINDOWS, N_CHANNELS, SAMPLES_PER_WINDOW)
    padding_mask: np.ndarray    # bool (MAX_WINDOWS,), True = padded
    signal_start_s: float       # seconds of leading flat signal skipped (native-rate grid of 100 s)
    duration_s: float           # native recording duration before the skip

    @property
    def n_valid(self) -> int:
        return int((~self.padding_mask).sum())

    def to_entry(self, source_sha256: Optional[str]) -> dict:
        return {
            "windows_uv": self.windows_uv,
            "padding_mask": self.padding_mask,
            "ch_names": list(CH_NAMES),
            "sfreq": TARGET_SFREQ,
            "signal_start_s": self.signal_start_s,
            "duration_s": self.duration_s,
            "version": CACHE_VERSION,
            "source_sha256": source_sha256,
        }


class SkipRecording(Exception):
    """Raised by :func:`process_raw` when a recording cannot be used."""

    def __init__(self, reason: str, detail: str = ""):
        if reason not in SKIP_REASONS:
            raise ValueError(f"unknown skip reason {reason!r}")
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def find_signal_start(data_uv: np.ndarray, sfreq: float, chunk_s: float = FLAT_CHUNK_S,
                      flat_sd_uv: float = FLAT_SD_UV) -> Optional[int]:
    """First sample of the first ``chunk_s`` block whose standard deviation across all
    channels exceeds ``flat_sd_uv``; ``None`` if every block is flat.

    Mirrors the supervisor's ``find_signal_start`` (chunks of 100 s, threshold 1e-8 V).
    """
    n_samples = data_uv.shape[1]
    chunk = max(1, int(round(chunk_s * sfreq)))
    for start in range(0, n_samples, chunk):
        block = data_uv[:, start:start + chunk]
        if block.size and float(np.nanstd(block)) > flat_sd_uv:
            return start
    return None


def process_raw(raw, notch_hz: float) -> Recording:
    """Turn a loaded MNE Raw into cached windows.

    Steps, in order: keep the 19 standard channels (canonical order) and check that
    each is a voltage channel, skip the leading flat segment (detected on the native
    signal, as the supervisor's pipeline does; the skip lands on the 100 s chunk grid,
    so up to 99 s of flat signal can remain and is covered by the 300 s skip), resample
    to 200 Hz, convert to microvolts, band-pass and notch, skip 300 s and keep the next
    1200 s, split into 10 s windows, reject non-finite values.

    Raises:
        SkipRecording: with reason ``missing_channels``, ``units``, ``flat``,
        ``too_short`` or ``non_finite``.
    """
    try:
        raw = filter_to_standard_19(raw)
    except ValueError as exc:
        raise SkipRecording("missing_channels", str(exc)) from exc
    if list(raw.ch_names) != list(CH_NAMES):
        raise SkipRecording("missing_channels", f"channel order {raw.ch_names} differs from {CH_NAMES}")
    _require_voltage_channels(raw)
    native_sfreq = float(raw.info["sfreq"])
    duration_s = raw.n_times / native_sfreq
    start = find_signal_start(np.asarray(raw.get_data(units="uV"), dtype=np.float64), native_sfreq)
    if start is None:
        raise SkipRecording("flat", "no 100 s block above the flat threshold")
    signal_start_s = start / native_sfreq
    if start > 0:
        # crop before resampling so the step at the end of the flat segment cannot ring
        raw.crop(tmin=signal_start_s, include_tmax=True)
    if native_sfreq != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)
    sfreq = float(raw.info["sfreq"])
    data_uv = np.asarray(raw.get_data(units="uV"), dtype=np.float64)

    data_uv = apply_filters(data_uv, sfreq, lowcut=EEG_CONFIG["lowcut"],
                            highcut=EEG_CONFIG["highcut"], notch_freq=notch_hz)
    segment = extract_time_window(data_uv, sfreq, skip_start_sec=EEG_CONFIG["skip_start_sec"],
                                  use_duration_sec=EEG_CONFIG["use_duration_sec"],
                                  min_duration_sec=EEG_CONFIG["min_duration_sec"])
    if segment is None:
        raise SkipRecording("too_short", f"{(data_uv.shape[1] / sfreq):.0f} s after the leading flat segment")
    windows, padding_mask = create_windows(segment, sfreq, window_sec=EEG_CONFIG["window_sec"],
                                           max_windows=MAX_WINDOWS)
    if windows.shape != (MAX_WINDOWS, N_CHANNELS, SAMPLES_PER_WINDOW):
        raise RuntimeError(f"unexpected window shape {windows.shape}")
    if not np.isfinite(windows).all():
        raise SkipRecording("non_finite", "NaN or infinite samples after filtering")
    return Recording(windows_uv=windows.astype(np.float32, copy=False),
                     padding_mask=padding_mask.astype(bool, copy=False),
                     signal_start_s=signal_start_s, duration_s=duration_s)


def _require_voltage_channels(raw) -> None:
    """Every retained channel must come from a voltage unit, and be typed EEG so that
    ``get_data(units="uV")`` applies the microvolt scaling.

    MNE converts EDF voltage channels (V, mV, uV) to volts on reading; a channel whose
    physical dimension is not a voltage is left unscaled and would be mis-scaled here,
    so the recording is skipped instead. Raw objects without recorded original units
    (synthetic ``RawArray``) are taken to be in volts, which is MNE's convention.
    """
    orig_units = getattr(raw, "_orig_units", None) or {}
    bad = [ch for ch in raw.ch_names
           if str(orig_units.get(ch, "V")).strip().lower() not in VOLTAGE_UNITS]
    if bad:
        raise SkipRecording("units", f"{len(bad)} channel(s) without a voltage unit")
    not_eeg = [ch for ch, kind in zip(raw.ch_names, raw.get_channel_types()) if kind != "eeg"]
    if not_eeg:
        raw.set_channel_types({ch: "eeg" for ch in not_eeg}, verbose=False)


def process_edf(path: Path, notch_hz: float, reader: Callable[[Path], object] = read_edf) -> Recording:
    """Read an EDF file and process it (see :func:`process_raw`)."""
    try:
        raw = reader(Path(path))
    except Exception as exc:  # unreadable file: counted, never fatal for the build
        raise SkipRecording("read_error", f"{type(exc).__name__}: {exc}") from exc
    return process_raw(raw, notch_hz)


# ---------------------------------------------------------------------------
# Discovery of recordings
# ---------------------------------------------------------------------------

def scan_edf_files(root: Path) -> List[Path]:
    """Every ``.edf`` (any case) under ``root``, following symbolic links, sorted."""
    found: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if name.lower().endswith(".edf"):
                found.append(Path(dirpath) / name)
    return sorted(found)


_DATED_STEM = re.compile(r"^(?P<pid>[A-Za-z]*\d+)-\d{1,2}-\d{1,2}-\d{4}$")


def patient_id_from_filename(name: str) -> Optional[str]:
    """Candidate patient id in an EDF file name.

    The id before the first comma or underscore (``extract_patient_id``), else the
    whole stem when it has no separator (``1234.edf``), else the leading token of a
    ``<id>-<d>-<m>-<yyyy>`` stem; ``None`` when none of these forms applies.
    """
    pid = extract_patient_id(name)
    if pid:
        return pid
    stem = Path(name).stem.strip()
    if stem.isalnum():
        return stem
    match = _DATED_STEM.match(stem)
    return match.group("pid") if match else None


def map_patients_to_edf(edf_files: Iterable[Path], known_pids: Optional[Iterable[str]] = None
                        ) -> Tuple[Dict[str, Path], Dict[str, int]]:
    """Patient id to EDF file by exact equality of the id parsed from the file name
    (``patient_id_from_filename``); never by prefix.

    With ``known_pids`` only those ids are kept and every other file is counted as
    ``unmatched``. When a patient has several files the first in sorted order is kept
    and the patient is counted in ``multiple_files``.
    """
    known = None if known_pids is None else {str(k) for k in known_pids}
    mapping: Dict[str, Path] = {}
    counts = {"multiple_files": 0, "unmatched": 0}
    seen_multi = set()
    for path in sorted(edf_files):
        pid = patient_id_from_filename(path.name)
        if not pid or (known is not None and pid not in known):
            counts["unmatched"] += 1
            continue
        if pid in mapping:
            if pid not in seen_multi:
                counts["multiple_files"] += 1
                seen_multi.add(pid)
            continue
        mapping[pid] = path
    return mapping, counts


def cohort_patients(cohort: str, asm_data_dir: Optional[Path] = None) -> List[str]:
    """Patient ids with a usable outcome in the cohort's clinical CSV, as strings."""
    spec = COHORTS[cohort]
    data_dir = asm_data_dir or find_asm_data_dir()
    df = pd.read_csv(data_dir / str(spec["csv"]))
    df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")
    df = df[df["outcome"].isin(spec["outcomes"])]
    return sorted(set(df[str(spec["pid_col"])].astype(str)))


def discover_recordings(cohort: str, asm_data_dir: Optional[Path] = None) -> Tuple[List[Tuple[str, Path]], Dict[str, int]]:
    """(pid, edf path) for every cohort patient with an EDF, plus discovery counts."""
    spec = COHORTS[cohort]
    data_dir = asm_data_dir or find_asm_data_dir()
    edf_root = data_dir.joinpath(*spec["edf_dir"])
    pids = cohort_patients(cohort, data_dir)
    files = scan_edf_files(edf_root)
    mapping, counts = map_patients_to_edf(files, known_pids=pids)
    pairs = [(pid, mapping[pid]) for pid in pids if pid in mapping]
    counts.update({"csv_patients": len(pids), "edf_files": len(files), "patients_with_edf": len(pairs)})
    return pairs, counts


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=EXPERIMENTS_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def summarise(recordings: Dict[str, dict]) -> dict:
    """Aggregate statistics of a set of cache entries (no per-patient values)."""
    if not recordings:
        return {"n_recordings": 0}
    n_valid = np.array([int((~e["padding_mask"]).sum()) for e in recordings.values()])
    window_sd = []
    n_flat_channel = 0
    n_flat_windows = 0
    n_signal_start = 0
    for e in recordings.values():
        valid = e["windows_uv"][~e["padding_mask"]]
        if valid.size:
            sd = valid.std(axis=-1)                      # (n_valid, channels)
            window_sd.append(float(np.median(sd)))
            n_flat_channel += int((sd.max(axis=0) < ZSCORE_SD_FLOOR_UV).any())
            n_flat_windows += int((sd.max(axis=1) < FLAT_SD_UV).sum())   # flat in every channel
        n_signal_start += int(e.get("signal_start_s", 0.0) > 0)
    return {
        "n_recordings": len(recordings),
        "valid_windows": {"min": int(n_valid.min()), "median": float(np.median(n_valid)), "max": int(n_valid.max())},
        "n_full_length": int((n_valid == MAX_WINDOWS).sum()),
        "median_window_sd_uv": float(np.median(window_sd)) if window_sd else None,
        "n_recordings_with_flat_channel": n_flat_channel,
        "n_flat_valid_windows": n_flat_windows,
        "n_recordings_with_leading_flat_segment": n_signal_start,
    }


def build_cache(cohort: str, out_path: Optional[Path] = None, limit: Optional[int] = None,
                asm_data_dir: Optional[Path] = None, reader: Callable[[Path], object] = read_edf,
                pairs: Optional[Sequence[Tuple[str, Path]]] = None,
                expect_files: Optional[int] = None) -> dict:
    """Build the cohort's cache and sidecar; returns the sidecar metadata.

    ``expect_files`` is the number of EDF files the cohort is known to have; a warning
    is logged and recorded when fewer are found (an incomplete copy of the data).
    """
    if cohort not in COHORTS:
        raise ValueError(f"cohort must be one of {sorted(COHORTS)}, not {cohort!r}")
    spec = COHORTS[cohort]
    out_path = Path(out_path or CACHE_PATHS[cohort])
    if pairs is None:
        pairs, discovery = discover_recordings(cohort, asm_data_dir)
    else:
        pairs, discovery = list(pairs), {"patients_with_edf": len(pairs)}
    if limit is not None:
        pairs = pairs[:limit]
    n_files = discovery.get("edf_files", len(pairs))
    incomplete = expect_files is not None and n_files < expect_files
    if incomplete:
        logger.warning("%s: found %d EDF files, expected %d; the cache will be incomplete",
                       cohort, n_files, expect_files)
    notch_hz = float(spec["notch_hz"])

    recordings: Dict[str, dict] = {}
    skipped = {reason: 0 for reason in SKIP_REASONS}
    t0 = _dt.datetime.now()
    for i, (pid, path) in enumerate(pairs, 1):
        try:
            rec = process_edf(path, notch_hz, reader=reader)
        except SkipRecording as skip:
            skipped[skip.reason] += 1
            logger.info("skipped one recording (%s)", skip.reason)
            continue
        try:
            digest = _sha256(path)
        except OSError:
            digest = None
        recordings[pid] = rec.to_entry(digest)
        if i % 20 == 0 or i == len(pairs):
            logger.info("%s: processed %d/%d recordings (%d kept)", cohort, i, len(pairs), len(recordings))

    meta = {
        "version": CACHE_VERSION,
        "cohort": cohort,
        "built_at": t0.isoformat(timespec="seconds"),
        "build_seconds": round((_dt.datetime.now() - t0).total_seconds(), 1),
        "experiments_commit": _git_commit(),
        "ch_names": list(CH_NAMES),
        "n_channels": N_CHANNELS,
        "sfreq": TARGET_SFREQ,
        "window_s": EEG_CONFIG["window_sec"],
        "max_windows": MAX_WINDOWS,
        "bandpass_hz": [EEG_CONFIG["lowcut"], EEG_CONFIG["highcut"]],
        "notch_hz": notch_hz,
        "skip_start_s": EEG_CONFIG["skip_start_sec"],
        "use_duration_s": EEG_CONFIG["use_duration_sec"],
        "min_duration_s": EEG_CONFIG["min_duration_sec"],
        "flat_chunk_s": FLAT_CHUNK_S,
        "flat_sd_uv": FLAT_SD_UV,
        "units": "uV",
        "normalisation": "none (applied by load_cache)",
        "discovery": discovery,
        "expect_files": expect_files,
        "incomplete": bool(incomplete),
        "n_processed": len(pairs),
        "skipped": skipped,
        **summarise(recordings),
    }
    return write_cache(out_path, meta, recordings)


def write_cache(path: Path, meta: dict, recordings: Dict[str, dict]) -> dict:
    """Write the cache atomically plus its sidecar ``<path>.meta.json``; returns the
    metadata as written (``meta`` plus the channel and shape fields)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {**meta, **_shape_fields(recordings)}
    payload = {"meta": meta, "recordings": recordings}
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".partial")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    _write_text_atomic(sidecar_path(path), json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def _write_text_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".partial")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _shape_fields(recordings: Dict[str, dict]) -> dict:
    """Channel names and array shape taken from the stored entries themselves."""
    if not recordings:
        return {"ch_names": list(CH_NAMES), "n_channels": N_CHANNELS, "max_windows": MAX_WINDOWS,
                "samples_per_window": SAMPLES_PER_WINDOW}
    first = next(iter(recordings.values()))
    return {"ch_names": list(first["ch_names"]), "n_channels": int(first["windows_uv"].shape[1]),
            "max_windows": int(first["windows_uv"].shape[0]), "samples_per_window": int(first["windows_uv"].shape[2])}


def sidecar_path(path: Path) -> Path:
    return Path(path).with_name(Path(path).name + ".meta.json")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def normalise_windows(windows_uv: np.ndarray, padding_mask: np.ndarray, convention: str) -> np.ndarray:
    """Apply a convention to one recording's windows; padded windows stay zero."""
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, not {convention!r}")
    out = np.zeros(windows_uv.shape, dtype=np.float32)
    valid = ~np.asarray(padding_mask, dtype=bool)
    if not valid.any():
        return out
    x = np.asarray(windows_uv[valid], dtype=np.float64)
    if convention == "zscore_window":
        mean = x.mean(axis=-1, keepdims=True)
        sd = x.std(axis=-1, keepdims=True)
        sd = np.maximum(sd, ZSCORE_SD_FLOOR_UV)
        x = (x - mean) / sd
    elif convention == "labram":
        x = x / LABRAM_SCALE_UV
    out[valid] = x.astype(np.float32)
    return out


def _read_payload(path: Path, allow_legacy: bool) -> dict:
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    is_v2 = isinstance(payload, dict) and isinstance(payload.get("meta"), dict) \
        and payload["meta"].get("version") == CACHE_VERSION and isinstance(payload.get("recordings"), dict)
    if is_v2:
        return payload
    if allow_legacy:
        logger.warning("%s is not a version-%d cache; returned unchanged (allow_legacy=True)", path, CACHE_VERSION)
        return {"meta": {"version": "legacy"}, "recordings": payload}
    raise ValueError(
        f"{path} is not a version-{CACHE_VERSION} EEG cache (the superseded processed_eeg*.pkl caches "
        f"hold the wrong channels in volts). Build one with: python -m shared.eeg_cache build --cohort <cohort>"
    )


def load_cache(path: Path, convention: str, allow_legacy: bool = False) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """``{pid: (windows, padding_mask)}`` with ``convention`` applied to the valid windows.

    ``windows`` is float32 ``(max_windows, n_channels, samples_per_window)`` and
    ``padding_mask`` is bool ``(max_windows,)``, True for padded windows.

    ``allow_legacy=True`` returns a superseded version-1 cache exactly as stored (its
    own channels and units, no normalisation) and is accepted only with
    ``convention="raw_uv"``; it exists for inspection, not for training.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, not {convention!r}")
    if allow_legacy and convention != "raw_uv":
        raise ValueError("a legacy cache can only be loaded with convention='raw_uv' (its values are not microvolts)")
    payload = _read_payload(Path(path), allow_legacy)
    if payload["meta"].get("version") == "legacy":
        return {pid: (np.asarray(w, dtype=np.float32), np.asarray(m, dtype=bool))
                for pid, (w, m) in payload["recordings"].items()}
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for pid, entry in payload["recordings"].items():
        mask = np.asarray(entry["padding_mask"], dtype=bool)
        out[str(pid)] = (normalise_windows(entry["windows_uv"], mask, convention), mask)
    return out


def cache_info(path: Path) -> dict:
    """The cache metadata (version, cohort, channels, counts).

    Read from the sidecar when it exists and is newer than or as new as the pickle;
    otherwise from the pickle itself (which loads the whole cache).
    """
    path = Path(path)
    side = sidecar_path(path)
    if side.exists() and path.exists() and side.stat().st_mtime >= path.stat().st_mtime:
        try:
            meta = json.loads(side.read_text())
        except json.JSONDecodeError:
            meta = {}
        if meta.get("version") == CACHE_VERSION:
            return meta
    payload = _read_payload(path, allow_legacy=False)
    meta = {**payload["meta"], **_shape_fields(payload["recordings"])}
    meta["n_recordings"] = len(payload["recordings"])
    return meta


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build a cohort's cache and sidecar")
    b.add_argument("--cohort", choices=sorted(COHORTS), required=True)
    b.add_argument("--out", type=Path, default=None, help="cache path (default outputs/eeg_cache/eeg19_v2_<cohort>.pkl)")
    b.add_argument("--limit", type=int, default=None, help="process only the first N recordings (smoke runs)")
    b.add_argument("--asm-data-dir", type=Path, default=None)
    b.add_argument("--expect-files", type=int, default=None,
                   help="EDF files the cohort is known to have; warns if fewer are found")
    s = sub.add_parser("stats", help="print a cache's sidecar metadata")
    s.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "build":
        meta = build_cache(args.cohort, args.out, limit=args.limit, asm_data_dir=args.asm_data_dir,
                           expect_files=args.expect_files)
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0
    print(json.dumps(cache_info(args.path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
