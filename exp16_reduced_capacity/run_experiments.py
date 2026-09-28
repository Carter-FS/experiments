"""Run Experiment 16 (reduced-capacity quad-modal fusion) with per-patient
prediction logging, one OOF file per size variant.

Usage:
    python -m exp16_reduced_capacity.run_experiments \\
        --seed 42 --asm-balance none --mode predictions --deterministic \\
        --output-dir outputs/exp16_predictions
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from shared.cv_splits import add_cv_args, current_seed, cv_suffix, fold_indices, outer_splits
from shared.cv_splits import apply_cv_args  # noqa: E402
from shared.prediction_logger import run_provenance

from .config import ASM_NAME_MAPPING, CV_CONFIG, VARIANTS
from .data_pipeline import create_quad_modality_datasets, prepare_quad_modality_data
from .training import train_fold_with_predictions

logger = logging.getLogger("exp16")


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


def _top_n_asms_for_cohort(df, n: int = 5) -> List[str]:
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


def _build_candidate_smiles(top_asms, smiles_embeddings, smiles_indices) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for short in top_asms:
        canon = ASM_NAME_MAPPING.get(short, short)
        if canon not in smiles_indices:
            logger.warning(f"  Skipping ASM '{short}' (canon '{canon}') - not in smiles_indices.")
            continue
        out[short] = np.asarray(smiles_embeddings[smiles_indices[canon]], dtype=np.float32)
    return out


def run_exp16_with_predictions(output_dir: Path, top_n_asms: int, device, asm_balance_mode: str,
                               seed: int, output_suffix: str = "", splitter: str = "legacy",
                               inner_val: float = 0.0) -> None:
    """``splitter``/``inner_val`` set the CV protocol: legacy (the default)
    early-stops on the held-out fold; clean runs early-stop on an inner split
    of the training folds and predict the held-out fold once."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cohort + modalities are shared across variants -> load once.
    df, smiles_embeddings, smiles_indices, text_embeddings, eeg_data = prepare_quad_modality_data(
        text_model="clinicalbert", smiles_model="chemberta",
    )
    outcomes = df["outcome"].values
    logger.info(f"  Cohort size: {len(df)} patients")

    top_asms = _top_n_asms_for_cohort(df, n=top_n_asms)
    candidate_smiles = _build_candidate_smiles(top_asms, smiles_embeddings, smiles_indices)
    asms_used = list(candidate_smiles.keys())

    splits = outer_splits(
        df, mode=splitter, n_splits=CV_CONFIG["n_splits"], seed=CV_CONFIG["random_state"],
    )
    # Protocol suffix after the balance suffix; empty for the legacy protocol.
    file_suffix = output_suffix + cv_suffix(splitter, inner_val)

    for variant in VARIANTS:
        logger.info(f"=== Variant {variant['name']} (hidden_dim={variant['hidden_dim']}, "
                    f"eeg_embed_dim={variant['eeg_embed_dim']}, agg={variant['aggregator_type']}) ===")
        folds_payload: List[Dict] = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            from shared.determinism import enable_determinism
            enable_determinism(seed + fold)

            # Clinical preprocessor is fitted on the fit set only. Clean runs
            # early-stop on an inner split and predict the outer fold separately.
            fit_idx, es_idx, test_idx = fold_indices(outcomes, train_idx, val_idx, fold, inner_val)
            train_ds, val_ds, _ = create_quad_modality_datasets(
                df, smiles_embeddings, smiles_indices, text_embeddings, eeg_data,
                fit_idx, es_idx, return_pid=True,
            )
            test_ds = None
            if test_idx is not None:
                test_ds = create_quad_modality_datasets(
                    df, smiles_embeddings, smiles_indices, text_embeddings, eeg_data,
                    fit_idx, test_idx, return_pid=True,
                )[1]
            result = train_fold_with_predictions(
                train_ds, val_ds, variant=variant, device=device, fold=fold,
                candidate_smiles=candidate_smiles, asm_balance_mode=asm_balance_mode,
                test_dataset=test_ds,
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
            })
            logger.info(f"  Fold {fold + 1}: AUC={result['metrics'].get('auc', float('nan')):.4f}")

        oof_payload = {
            "experiment": variant["name"],
            "asm_balance_mode": asm_balance_mode,
            "seed": seed,
            "hidden_dim": variant["hidden_dim"],
            "eeg_embed_dim": variant["eeg_embed_dim"],
            "aggregator_type": variant["aggregator_type"],
            "text_model": variant["text_model"],
            "smiles_model": variant["smiles_model"],
            "asms": asms_used,
            "cv_random_state": current_seed(CV_CONFIG["random_state"]),
            "n_splits": CV_CONFIG["n_splits"],
            "folds": folds_payload,
            "metadata": {
                "splitter": splitter,
                "inner_val": inner_val,
                "asm_balance": asm_balance_mode,
                "provenance": run_provenance(),
            },
        }
        oof_path = output_dir / f"predictions_oof_{variant['name']}{file_suffix}.json"
        _save_predictions_json(oof_payload, oof_path)


def main():
    parser = argparse.ArgumentParser(description="Exp16: reduced-capacity quad-modal fusion")
    parser.add_argument("--mode", choices=["predictions"], default="predictions")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for determinism (default: --cv-seed if given, else 42).")
    parser.add_argument("--output-dir", "--output_dir", type=str, default=None)
    parser.add_argument("--asm-balance", choices=["none", "weighted", "stratified_batch"], default="none")
    parser.add_argument("--top-n-asms", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    add_cv_args(parser)
    args = parser.parse_args()
    apply_cv_args(args)  # repeated-CV seed and refit protocol (plan B.2)
    if args.seed is None:
        args.seed = 42 if args.cv_seed is None else args.cv_seed

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if args.deterministic:
        from shared.determinism import enable_determinism
        enable_determinism(args.seed)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")

    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent.parent / "outputs" / "exp16_predictions"
    )
    suffix = {"weighted": "_asmweighted", "stratified_batch": "_asmstratbatch"}.get(args.asm_balance, "")
    run_exp16_with_predictions(
        output_dir=out_dir, top_n_asms=args.top_n_asms, device=device,
        asm_balance_mode=args.asm_balance, seed=args.seed, output_suffix=suffix,
        splitter=args.splitter, inner_val=args.inner_val,
    )


if __name__ == "__main__":
    main()
