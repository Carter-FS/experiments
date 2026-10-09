"""Run Experiment 15 (REVE-based quad-modal fusion) with per-patient
prediction logging and the Stage A discipline.

Usage:
    python -m exp15_reve_quad_mlp.run_experiments \\
        --seed 42 --asm-balance none \\
        --mode predictions --deterministic
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from shared.cv_splits import add_cv_args, current_seed, cv_suffix, outer_splits
from shared.epoch_selection import run_outer_fold
from shared.cv_splits import apply_cv_args  # noqa: E402
from shared.prediction_logger import protocol_metadata, run_provenance

from .config import ASM_NAME_MAPPING, CV_CONFIG, DEFAULT_FEATURE_SET, FEATURE_SETS, RESULTS_DIR
from .data_pipeline import (
    create_reve_quad_datasets,
    prepare_quad_modality_data_reve,
)
from .training import (
    run_cross_validation,
    train_fold_with_predictions,
)

logger = logging.getLogger("exp15")


def _save_predictions_json(payload: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_convert(payload), f, indent=2)
    logger.info(f"Predictions saved to {path}")


def _normalise_asm(name: str) -> str:
    s = str(name).strip()
    return ASM_NAME_MAPPING.get(s, s)


def _top_n_asms_for_cohort(df, n: int = 5) -> List[str]:
    """Return the top-n most-prescribed ASM short codes in this cohort.

    Mirrors exp7's _top_n_asms_for_cohort exactly so the counterfactual
    swap behaves identically.
    """
    counts: Dict[str, int] = {}
    short_for_canon: Dict[str, str] = {}
    for raw in df["ASM"].astype(str):
        short = raw.strip()
        canon = ASM_NAME_MAPPING.get(short, short)
        preferred = short.upper() if short.upper() in ASM_NAME_MAPPING else short
        counts[canon] = counts.get(canon, 0) + 1
        short_for_canon.setdefault(canon, preferred)
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [short_for_canon[canon] for canon, _ in ordered[:n]]


def _build_candidate_smiles(
    top_asms: List[str],
    smiles_embeddings: np.ndarray,
    smiles_indices: Dict[str, int],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for short in top_asms:
        canon = ASM_NAME_MAPPING.get(short, short)
        if canon not in smiles_indices:
            logger.warning(f"  Skipping ASM '{short}' (canon '{canon}') - not in smiles_indices.")
            continue
        idx = smiles_indices[canon]
        out[short] = np.asarray(smiles_embeddings[idx], dtype=np.float32)
    return out


def run_exp15_with_predictions(
    output_dir: Path,
    top_n_asms: int = 5,
    text_model: str = "clinicalbert",
    smiles_model: str = "chemberta",
    device: torch.device = None,
    asm_balance_mode: str = "none",
    seed: int = 42,
    output_suffix: str = "",
    splitter: str = "legacy",
    inner_val: float = 0.0,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> Dict[str, Path]:
    """Run exp15 with per-patient and ASM-swap prediction logging.

    ``splitter``/``inner_val`` set the CV protocol: legacy (the default)
    early-stops on the held-out fold; clean runs early-stop on an inner split
    of the training folds and predict the held-out fold once.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}, seed: {seed}, asm_balance: {asm_balance_mode}")

    df, smiles_embeddings, smiles_indices, text_embeddings, reve_data = (
        prepare_quad_modality_data_reve(text_model, smiles_model, feature_set)
    )
    outcomes = df["outcome"].values
    logger.info(f"  Cohort size: {len(df)} patients")

    top_asms = _top_n_asms_for_cohort(df, n=top_n_asms)
    logger.info(f"  Top-{top_n_asms} ASMs: {top_asms}")
    candidate_smiles = _build_candidate_smiles(top_asms, smiles_embeddings, smiles_indices)
    asms_used = list(candidate_smiles.keys())

    splits = outer_splits(
        df, mode=splitter, n_splits=CV_CONFIG["n_splits"], seed=CV_CONFIG["random_state"],
    )

    folds_payload: List[Dict] = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"Fold {fold + 1}/{CV_CONFIG['n_splits']}")
        # Re-seed determinism per fold so seed param actually varies init
        from shared.determinism import enable_determinism
        enable_determinism(seed + fold)

        # Clinical preprocessor is fitted on the fit set only. Clean runs select
        # on inner data (inner split, or inner folds + refit) and predict the outer fold once.
        result, selection = run_outer_fold(
            outcomes, train_idx, val_idx, fold, inner_val,
            make_datasets=lambda a, b: create_reve_quad_datasets(
                df, smiles_embeddings, smiles_indices, text_embeddings, reve_data, a, b,
                return_pid=True,
            )[:2],
            train=lambda tr, va, **kw: train_fold_with_predictions(
                tr, va, device=device, fold=fold, candidate_smiles=candidate_smiles,
                asm_balance_mode=asm_balance_mode, **kw,
            ),
            log=logger.info,
        )

        scalar_metrics = {
            k: float(v) for k, v in result["metrics"].items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        folds_payload.append({
            "fold": fold,
            "metrics": scalar_metrics,
            "pids": result["val_pids"],
            "y_true": result["val_y_true"],
            "y_prob": result["val_y_prob"],
            "y_prob_per_asm": {a: result["val_y_prob_per_asm"].get(a, []) for a in asms_used},
            **({"selection": selection} if selection else {}),
        })
        logger.info(
            f"  Fold {fold + 1}: AUC={result['metrics'].get('auc', float('nan')):.4f}, "
            f"BalAcc_tuned={result['metrics'].get('balanced_acc_tuned', float('nan')):.4f}"
        )

    oof_payload = {
        "experiment": f"exp15_{feature_set}" + (f"_{output_suffix}" if output_suffix else ""),
        "feature_set": feature_set,
        "asm_balance_mode": asm_balance_mode,
        "seed": seed,
        "text_model": text_model,
        "smiles_model": smiles_model,
        "asms": asms_used,
        "cv_random_state": current_seed(CV_CONFIG["random_state"]),
        "n_splits": CV_CONFIG["n_splits"],
        "folds": folds_payload,
        "metadata": {
            **protocol_metadata(inner_val),
            "splitter": splitter,
            "inner_val": inner_val,
            "asm_balance": asm_balance_mode,
            "provenance": run_provenance(),
        },
    }
    # Feature set, then the balance suffix, then the protocol suffix (empty for legacy).
    suffix_part = (f"_{output_suffix}" if output_suffix else "") + cv_suffix(splitter, inner_val)
    oof_path = output_dir / f"predictions_oof_{feature_set}{suffix_part}.json"
    _save_predictions_json(oof_payload, oof_path)

    return {"oof": oof_path}


def main():
    parser = argparse.ArgumentParser(
        description="Exp15: REVE-based quad-modal fusion (Exp7a architecture with REVE EEG)"
    )
    parser.add_argument(
        "--mode", type=str, choices=["cv", "predictions"], default="predictions",
        help="'predictions' (default) logs per-fold OOF + counterfactual swap; 'cv' is just metrics.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for determinism (default: --cv-seed if given, else 42).",
    )
    parser.add_argument(
        "--output-dir", "--output_dir", type=str, default=None,
        help="Output directory; default outputs/exp15_reve_quad/seed{S}_{balance}/",
    )
    parser.add_argument(
        "--asm-balance", type=str,
        choices=["none", "weighted", "stratified_batch"], default="none",
        help="ASM balancing mode (Stage B integration).",
    )
    parser.add_argument(
        "--top-n-asms", type=int, default=5,
        help="Top-N ASMs for counterfactual swap (default 5).",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Enable deterministic seeding (recommended).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (default: auto-detect).",
    )
    parser.add_argument(
        "--feature-set", type=str, choices=sorted(FEATURE_SETS), default=DEFAULT_FEATURE_SET,
        help="Stored per-window EEG features to fuse (default reve_v2; labram_v2 = pretrained LaBraM-base).",
    )
    add_cv_args(parser)
    args = parser.parse_args()
    apply_cv_args(args)  # repeated-CV seed and refit protocol (plan B.2)
    if args.seed is None:
        args.seed = 42 if args.cv_seed is None else args.cv_seed

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.deterministic:
        from shared.determinism import enable_determinism
        enable_determinism(args.seed)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")

    if args.mode == "predictions":
        if args.output_dir:
            out_dir = Path(args.output_dir)
        else:
            out_dir = RESULTS_DIR.parent / "exp15_reve_quad" / f"seed{args.seed}_{args.asm_balance}"
        suffix = ""
        if args.asm_balance == "weighted":
            suffix = "asmweighted"
        elif args.asm_balance == "stratified_batch":
            suffix = "asmstratbatch"
        run_exp15_with_predictions(
            output_dir=out_dir,
            top_n_asms=args.top_n_asms,
            device=device,
            asm_balance_mode=args.asm_balance,
            seed=args.seed,
            output_suffix=suffix,
            splitter=args.splitter,
            inner_val=args.inner_val,
            feature_set=args.feature_set,
        )
    else:
        run_cross_validation(
            device=device, asm_balance_mode=args.asm_balance,
            splitter=args.splitter, inner_val=args.inner_val,
            feature_set=args.feature_set,
        )


if __name__ == "__main__":
    main()
