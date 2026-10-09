"""Per-window EEG features computed outside training (frozen pretrained encoders).

A feature set is an npz written from the version-2 EEG cache with the layout
``features (n_recordings, MAX_WINDOWS, dim)`` float32, ``pids`` (n_recordings,) and
``valid_window_counts`` (n_recordings,) int; padded windows hold zeros and the valid
windows form a prefix, so the padding mask is ``j >= valid_window_counts``. Consumers
read them through :func:`load_features`, which returns the same ``{pid: (windows,
padding_mask)}`` mapping as :func:`shared.eeg_cache.load_cache`, with ``windows`` of
shape ``(MAX_WINDOWS, dim)`` instead of ``(MAX_WINDOWS, channels, samples)``.

    python -m shared.eeg_features check --feature-set reve_v2 labram_v2 --cohort alfred

compares a feature file with its cohort's cache: the same patients, the same valid
window counts.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from exp2_fusion.config import MAX_WINDOWS
from shared.eeg_cache import CACHE_PATHS, load_cache
from shared.paths import EXPERIMENTS_ROOT

logger = logging.getLogger(__name__)

OUT_DIR = EXPERIMENTS_ROOT / "outputs"

# name -> file pattern, feature width, the cache convention the producer read, producer
FEATURE_SETS: Dict[str, dict] = {
    "reve_v2": {
        "filename": "reve_features_v2_{cohort}.npz",
        "dim": 512,
        "input_convention": "zscore_window",
        "producer": "thesisStandalone/analysis/reve_extract_features.py (.venv-reve)",
        "description": "REVE-base attention-pooled embedding per 10-s window, frozen",
    },
    "labram_v2": {
        "filename": "labram_features_v2_{cohort}.npz",
        "dim": 200,
        "input_convention": "labram",
        "producer": "python -m shared.labram_pretrained extract --cohort <cohort> (.venv-reve)",
        "description": "LaBraM-base mean-pooled patch tokens per 10-s window, frozen",
    },
}


def feature_path(feature_set: str, cohort: str = "alfred") -> Path:
    """Where a cohort's feature file lives (``outputs/``, outside git)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; known: {sorted(FEATURE_SETS)}")
    if cohort not in CACHE_PATHS:
        raise ValueError(f"unknown cohort {cohort!r}; known: {sorted(CACHE_PATHS)}")
    return OUT_DIR / FEATURE_SETS[feature_set]["filename"].format(cohort=cohort)


def load_features(feature_set: str, cohort: str = "alfred",
                  path: Optional[Path] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """``{pid: (features (MAX_WINDOWS, dim) float32, padding_mask (MAX_WINDOWS,) bool)}``.

    The file must have the registered width, ``MAX_WINDOWS`` rows per recording, finite
    values, unique pids and valid counts within range; padded rows are zeroed.
    """
    spec = FEATURE_SETS[feature_set] if feature_set in FEATURE_SETS else None
    if spec is None:
        raise ValueError(f"unknown feature set {feature_set!r}; known: {sorted(FEATURE_SETS)}")
    path = Path(path) if path is not None else feature_path(feature_set, cohort)
    if not path.exists():
        raise FileNotFoundError(f"{feature_set} features not found at {path}; produce them with {spec['producer']}")
    data = np.load(path)
    for key in ("features", "pids", "valid_window_counts"):
        if key not in data:
            raise ValueError(f"{path} lacks the {key!r} array")
    features = data["features"]
    pids = [str(p) for p in data["pids"]]
    counts = data["valid_window_counts"].astype(int)
    if features.ndim != 3 or features.shape[1] != MAX_WINDOWS or features.shape[2] != spec["dim"]:
        raise ValueError(f"{path}: features are {features.shape}, expected (n, {MAX_WINDOWS}, {spec['dim']})")
    if len(pids) != features.shape[0] or len(counts) != features.shape[0]:
        raise ValueError(f"{path}: {features.shape[0]} recordings but {len(pids)} pids and {len(counts)} counts")
    if len(set(pids)) != len(pids):
        raise ValueError(f"{path}: duplicate patient ids")
    if counts.min() < 0 or counts.max() > MAX_WINDOWS:
        raise ValueError(f"{path}: valid window counts outside 0..{MAX_WINDOWS}")
    if not np.isfinite(features).all():
        raise ValueError(f"{path}: non-finite feature values")
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for i, pid in enumerate(pids):
        windows = np.ascontiguousarray(features[i], dtype=np.float32)
        mask = np.arange(MAX_WINDOWS) >= counts[i]
        windows[mask] = 0.0
        out[pid] = (windows, mask)
    logger.info("Loaded %s features for %d recordings from %s (dim %d)", feature_set, len(out), path.name, spec["dim"])
    return out


def feature_meta(feature_set: str, cohort: str = "alfred", path: Optional[Path] = None) -> dict:
    """The producer's metadata (the npz ``meta`` field or the ``.meta.json`` sidecar), or ``{}``."""
    path = Path(path) if path is not None else feature_path(feature_set, cohort)
    side = Path(str(path) + ".meta.json")
    if side.exists():
        return json.loads(side.read_text())
    data = np.load(path)
    return json.loads(data["meta"].item()) if "meta" in data else {}


def check_features_against_cache(features: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                 cache_path: Path) -> dict:
    """Require the feature file and the cache to hold the same patients with the same
    valid window counts; returns the counts compared. Loads the cache, so this is for
    preflight and tests rather than every run."""
    cache = load_cache(cache_path, "raw_uv")
    missing = sorted(set(cache) - set(features))
    extra = sorted(set(features) - set(cache))
    if missing or extra:
        raise ValueError(f"feature file and {Path(cache_path).name} differ: {len(missing)} recordings without "
                         f"features, {len(extra)} features without a recording")
    mismatched = [pid for pid in cache if not np.array_equal(cache[pid][1], features[pid][1])]
    if mismatched:
        raise ValueError(f"{len(mismatched)} recordings have a different padding mask in the feature file")
    return {"n_recordings": len(cache), "n_valid_windows": int(sum((~m).sum() for _, m in features.values()))}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="compare feature files with their cohort's cache")
    c.add_argument("--feature-set", nargs="+", choices=sorted(FEATURE_SETS), required=True)
    c.add_argument("--cohort", choices=sorted(CACHE_PATHS), default="alfred")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name in args.feature_set:
        result = check_features_against_cache(load_features(name, args.cohort), CACHE_PATHS[args.cohort])
        print(json.dumps({"feature_set": name, "cohort": args.cohort, **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
