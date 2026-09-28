#!/usr/bin/env python3
"""Run all Experiment 3 configurations with cross-validation."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import EXPERIMENTS, RESULTS_DIR
from .training import run_cross_validation
from shared.cv_splits import add_cv_args, cv_suffix
from shared.cv_splits import apply_cv_args  # noqa: E402
from shared.prediction_logger import PredictionLogger

logger = logging.getLogger("exp3")


def run_single_experiment(
    exp_config: Dict,
    device: torch.device,
    log_predictions: bool = False,
    asm_balance_mode: str = "none",
    predictions_dir: Optional[Path] = None,
    splitter: str = "legacy",
    inner_val: float = 0.0,
) -> Dict[str, List[float]]:
    """Run a single experiment configuration."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running: {exp_config['name']}")
    logger.info(f"  Text: {exp_config['text']}")
    logger.info(f"  SMILES: {exp_config['smiles']}")
    logger.info(f"  Fusion: {exp_config['fusion']}")
    if exp_config.get("eeg_encoder"):
        logger.info(f"  EEG encoder: {exp_config['eeg_encoder']}")
    logger.info(f"{'=' * 60}")

    pred_logger = None
    if log_predictions:
        import os
        pred_dir = predictions_dir if predictions_dir is not None else os.path.join(
            os.path.dirname(RESULTS_DIR), "exp3_predictions")
        suffix = ""
        if asm_balance_mode == "weighted":
            suffix = "_asmweighted"
        elif asm_balance_mode == "stratified_batch":
            suffix = "_asmstratbatch"
        suffix += cv_suffix(splitter, inner_val)
        pred_logger = PredictionLogger(
            exp_id=exp_config['name'], output_dir=pred_dir,
            filename=f"predictions_oof_{exp_config['name']}{suffix}.json",
            metadata={"splitter": splitter, "inner_val": inner_val, "asm_balance": asm_balance_mode},
        )

    results = run_cross_validation(
        text_model=exp_config["text"],
        smiles_model=exp_config["smiles"],
        fusion_type=exp_config["fusion"],
        device=device,
        prediction_logger=pred_logger,
        asm_balance_mode=asm_balance_mode,
        eeg_encoder_type=exp_config.get("eeg_encoder"),
        splitter=splitter,
        inner_val=inner_val,
    )

    if pred_logger is not None:
        saved = pred_logger.save()
        logger.info(f"Per-fold predictions written to {saved}")

    return results


def run_all_experiments(
    experiments: List[Dict] = None,
    device: torch.device = None,
    log_predictions: bool = False,
    asm_balance_mode: str = "none",
    predictions_dir: Optional[Path] = None,
    splitter: str = "legacy",
    inner_val: float = 0.0,
) -> Dict[str, Dict]:
    """Run all experiments and collect results."""
    if experiments is None:
        experiments = EXPERIMENTS
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running {len(experiments)} experiments on {device}")

    all_results = {}

    for exp_config in experiments:
        name = exp_config["name"]
        try:
            results = run_single_experiment(
                exp_config, device,
                log_predictions=log_predictions,
                asm_balance_mode=asm_balance_mode,
                predictions_dir=predictions_dir,
                splitter=splitter,
                inner_val=inner_val,
            )
            all_results[name] = {
                "config": exp_config,
                "fold_metrics": results,
                "summary": {
                    key: {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                    for key, values in results.items()
                },
            }
        except Exception as e:
            logger.error(f"Error running {name}: {e}")
            all_results[name] = {"config": exp_config, "error": str(e)}

    return all_results


def print_results_table(all_results: Dict[str, Dict]):
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3 RESULTS SUMMARY")
    print("=" * 80)

    # Header
    print(f"{'Experiment':<35} {'AUC':>12} {'Accuracy':>12} {'F1':>12}")
    print("-" * 80)

    # Sort by AUC
    sorted_results = sorted(
        [(name, res) for name, res in all_results.items() if "summary" in res],
        key=lambda x: x[1]["summary"]["auc"]["mean"],
        reverse=True,
    )

    for name, res in sorted_results:
        summary = res["summary"]
        auc = f"{summary['auc']['mean']:.3f}±{summary['auc']['std']:.3f}"
        acc = f"{summary['accuracy']['mean']:.3f}±{summary['accuracy']['std']:.3f}"
        f1 = f"{summary['f1']['mean']:.3f}±{summary['f1']['std']:.3f}"
        print(f"{name:<35} {auc:>12} {acc:>12} {f1:>12}")

    print("=" * 80)

    # Best result
    if sorted_results:
        best_name, best_res = sorted_results[0]
        print(f"\nBest model: {best_name}")
        print(f"  AUC: {best_res['summary']['auc']['mean']:.4f}")


def save_results(all_results: Dict[str, Dict], output_path: Path):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    serializable = {}
    for name, res in all_results.items():
        serializable[name] = {}
        for key, val in res.items():
            if isinstance(val, dict):
                serializable[name][key] = {
                    k: convert(v) if not isinstance(v, dict) else {
                        kk: convert(vv) for kk, vv in v.items()
                    }
                    for k, v in val.items()
                }
            else:
                serializable[name][key] = convert(val)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 3 configurations")
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        help="Specific experiments to run (by name). If not specified, runs all.",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        choices=["mlp", "fusemoe"],
        help="Run only experiments with this fusion type",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, or auto)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--log-predictions",
        action="store_true",
        help="Dump per-fold OOF predictions to outputs/exp3_predictions/",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic training (seeds, cuDNN deterministic)",
    )
    parser.add_argument(
        "--asm-balance",
        type=str,
        choices=["none", "weighted", "stratified_batch"],
        default="none",
        help="Stage B ASM-balancing mode",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Directory for OOF prediction files (default: outputs/exp3_predictions).",
    )
    add_cv_args(parser)
    args = parser.parse_args()
    apply_cv_args(args)  # repeated-CV seed and refit protocol (plan B.2)

    if args.deterministic:
        from shared.determinism import enable_determinism
        enable_determinism()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Select device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Filter experiments
    experiments = EXPERIMENTS.copy()

    if args.experiments:
        experiments = [e for e in experiments if e["name"] in args.experiments]

    if args.fusion:
        experiments = [e for e in experiments if e["fusion"] == args.fusion]

    if not experiments:
        logger.error("No experiments match the specified filters")
        return

    logger.info(f"Will run {len(experiments)} experiments:")
    for exp in experiments:
        logger.info(f"  - {exp['name']}")

    # Run experiments
    all_results = run_all_experiments(
        experiments, device,
        log_predictions=args.log_predictions,
        asm_balance_mode=args.asm_balance,
        predictions_dir=Path(args.predictions_dir) if args.predictions_dir else None,
        splitter=args.splitter,
        inner_val=args.inner_val,
    )

    # Print results
    print_results_table(all_results)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"exp3_results_{timestamp}.json"

    save_results(all_results, output_path)

    failed = [name for name, r in all_results.items() if "error" in r]
    if failed:
        raise SystemExit(f"{len(failed)} configuration(s) failed: {failed}")


if __name__ == "__main__":
    main()
