"""Per-fold prediction logging for experiments that need patient-level OOF outputs.

Use this helper to dump a ``predictions_oof.json`` file with the same schema
as ``outputs/exp7_predictions/predictions_oof.json`` (which is consumed by
``thesisStandalone/analysis/compute_bootstrap_cis.py`` and the planned
all-pairs DeLong statistical-comparison script).

Usage pattern inside an experiment's training script:

    from shared.prediction_logger import PredictionLogger

    logger = PredictionLogger(exp_id="exp4a_mlp", output_dir=OUTPUT_DIR)
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(...)):
        # ... train ...
        val_y_prob = model.predict_proba(...)[:, 1]
        val_pids = pid_array[val_idx]
        logger.log_fold(
            fold=fold_idx,
            pids=val_pids,
            y_true=val_y_true,
            y_prob=val_y_prob,
            threshold=fold_threshold,
        )
    logger.save()

The resulting JSON has the schema:

    {
        "exp_id": "exp4a_mlp",
        "n_folds": 5,
        "folds": [
            {"fold": 0, "pids": [...], "y_true": [...], "y_prob": [...], "threshold": 0.42},
            ...
        ],
        "metadata": {"splitter": ..., "inner_val": ..., "provenance": {...}},
    }

``metadata`` records the CV protocol (see shared/cv_splits.py) and the git
commits the run came from; consumers that only read ``folds`` ignore it.
"""

from __future__ import annotations

import datetime
import json
import socket
import subprocess
from pathlib import Path
from typing import Iterable

_EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent


def _git_head(repo: Path) -> str | None:
    """Short HEAD hash (with '+dirty' if the tree has changes), or None."""
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{head}+dirty" if dirty else head


def run_provenance() -> dict:
    """Where and from which code a prediction file was produced."""
    from shared.cv_splits import _REPEAT_SEED

    thesis = _EXPERIMENTS_ROOT / "thesisStandalone"
    return {
        "cv_seed": _REPEAT_SEED,
        "experiments_commit": _git_head(_EXPERIMENTS_ROOT),
        "thesis_commit": _git_head(thesis) if (thesis / ".git").exists() else None,
        "host": socket.gethostname(),
        "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def protocol_metadata(inner_val: float) -> dict:
    """Selection protocol of the active run ('legacy' / 'innersplit' / 'refit'),
    so consumers need not infer it from the filename."""
    from shared.cv_splits import protocol_name, refit_folds

    return {"protocol": protocol_name(inner_val), "refit_folds": refit_folds()}


class PredictionLogger:
    """Accumulate per-fold OOF predictions and dump as JSON.

    The logger is intentionally minimal: it stores native-Python lists
    (not numpy arrays) so the JSON dump is portable across environments.
    """

    def __init__(
        self,
        exp_id: str,
        output_dir: str | Path,
        filename: str = "predictions_oof.json",
        metadata: dict | None = None,
    ):
        self.exp_id = exp_id
        self.output_path = Path(output_dir) / filename
        self.folds: list[dict] = []
        self.metadata: dict = dict(metadata or {})

    def log_fold(
        self,
        fold: int,
        pids: Iterable,
        y_true: Iterable,
        y_prob: Iterable,
        threshold: float | None = None,
        selection: dict | None = None,
    ) -> None:
        """Append one fold's held-out predictions to the accumulator.

        ``selection`` (refit protocol) records how the epoch count and the
        threshold were chosen: EpochChoice.as_metadata()."""
        pids_list = [str(p) for p in pids]
        y_true_list = [int(v) for v in y_true]
        y_prob_list = [float(v) for v in y_prob]
        n = len(y_prob_list)
        if not (len(pids_list) == len(y_true_list) == n):
            raise ValueError(
                f"PredictionLogger.log_fold: length mismatch for fold {fold}: "
                f"pids={len(pids_list)}, y_true={len(y_true_list)}, y_prob={n}"
            )
        entry = {
            "fold": int(fold),
            "n": n,
            "pids": pids_list,
            "y_true": y_true_list,
            "y_prob": y_prob_list,
        }
        if threshold is not None:
            entry["threshold"] = float(threshold)
        if selection is not None:
            entry["selection"] = selection
        self.folds.append(entry)

    def save(self) -> Path:
        """Write the accumulated payload to ``predictions_oof.json``."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "exp_id": self.exp_id,
            "n_folds": len(self.folds),
            "folds": self.folds,
            "metadata": {**protocol_metadata(self.metadata.get("inner_val", 0) or 0),
                         **self.metadata, "provenance": run_provenance()},
        }
        with self.output_path.open("w") as handle:
            json.dump(payload, handle, indent=2)
        return self.output_path
