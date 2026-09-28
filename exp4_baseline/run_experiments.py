"""Run Experiment 4 configurations with cross-validation.

Usage:
    python -m exp4_baseline.run_experiments
    python -m exp4_baseline.run_experiments --model mlp
    python -m exp4_baseline.run_experiments --model attention
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from shared.cv_splits import add_cv_args, cv_suffix
from shared.cv_splits import apply_cv_args  # noqa: E402
from shared.prediction_logger import PredictionLogger

from .config import EXPERIMENTS, RESULTS_DIR
from .training import run_cross_validation

logger = logging.getLogger("exp4")


def compute_summary(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of values.

    Args:
        values: List of metric values from each fold.

    Returns:
        Dictionary with mean, std, min, max.
    """
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def run_all_experiments(
    experiments: Optional[List[Dict]] = None,
    device: torch.device = None,
    log_predictions: bool = False,
    predictions_dir: Optional[Path] = None,
    asm_balance_mode: str = "none",
    splitter: str = "legacy",
    inner_val: float = 0.0,
) -> Dict[str, Dict]:
    """Run all experiments and collect results.

    Args:
        experiments: List of experiment configs. Defaults to EXPERIMENTS.
        device: Device to use.

    Returns:
        Dictionary mapping experiment names to results.
    """
    if experiments is None:
        experiments = EXPERIMENTS

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = {}

    for exp in experiments:
        exp_name = exp["name"]
        model_type = exp["model"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment: {exp_name}")
        logger.info(f"{'='*60}")

        pred_logger = None
        if log_predictions:
            target_dir = predictions_dir if predictions_dir is not None else RESULTS_DIR.parent / "exp4_predictions"
            # Distinct filename per config (so exp4b_attention does not clobber
            # exp4a_mlp) plus a balance-mode suffix, matching exp3/5/6/7.
            suffix = {"weighted": "_asmweighted", "stratified_batch": "_asmstratbatch"}.get(asm_balance_mode, "")
            suffix += cv_suffix(splitter, inner_val)
            pred_logger = PredictionLogger(
                exp_id=exp_name, output_dir=target_dir,
                filename=f"predictions_oof_{exp_name}{suffix}.json",
                metadata={"splitter": splitter, "inner_val": inner_val, "asm_balance": asm_balance_mode},
            )

        fold_metrics = run_cross_validation(
            model_type=model_type,
            device=device,
            prediction_logger=pred_logger,
            asm_balance_mode=asm_balance_mode,
            splitter=splitter,
            inner_val=inner_val,
        )

        if pred_logger is not None:
            saved_path = pred_logger.save()
            logger.info(f"Per-fold predictions written to {saved_path}")

        # Compute summary statistics
        summary = {}
        for metric_name, values in fold_metrics.items():
            summary[metric_name] = compute_summary(values)

        all_results[exp_name] = {
            "config": exp,
            "fold_metrics": fold_metrics,
            "summary": summary,
        }

    return all_results


def print_results_table(all_results: Dict[str, Dict]):
    """Print results in formatted table.

    Args:
        all_results: Dictionary of experiment results.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4 RESULTS SUMMARY")
    print("=" * 80)

    # Header
    print(f"{'Experiment':<25} {'AUC':>12} {'Bal Acc':>12} {'F1':>12} {'F1 Tuned':>12}")
    print("-" * 80)

    for exp_name, results in all_results.items():
        summary = results["summary"]
        auc = f"{summary['auc']['mean']:.4f} +/- {summary['auc']['std']:.4f}"
        bal_acc = f"{summary['balanced_acc_tuned']['mean']:.4f} +/- {summary['balanced_acc_tuned']['std']:.4f}"
        f1 = f"{summary['f1']['mean']:.4f} +/- {summary['f1']['std']:.4f}"
        f1_tuned = f"{summary['f1_tuned']['mean']:.4f} +/- {summary['f1_tuned']['std']:.4f}"

        print(f"{exp_name:<25} {auc:>12} {bal_acc:>12} {f1:>12} {f1_tuned:>12}")

    print("=" * 80)

    # Print per-fold details
    print("\nPer-fold AUC values:")
    for exp_name, results in all_results.items():
        aucs = results["fold_metrics"]["auc"]
        fold_str = ", ".join([f"{v:.4f}" for v in aucs])
        print(f"  {exp_name}: [{fold_str}]")


def save_results(all_results: Dict[str, Dict], output_path: Path):
    """Save results to JSON.

    Args:
        all_results: Dictionary of experiment results.
        output_path: Path to save results.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialisation
    def convert_to_serialisable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serialisable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serialisable(v) for v in obj]
        return obj

    serialisable_results = convert_to_serialisable(all_results)

    with open(output_path, "w") as f:
        json.dump(serialisable_results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Experiment 4: Clinical features baseline"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "attention", "all"],
        default="all",
        help="Model type to run (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: outputs/exp4_results/results_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--log-predictions",
        action="store_true",
        help="Dump per-fold OOF predictions to outputs/exp4_predictions/predictions_oof.json",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic training (seeds, cuDNN deterministic)",
    )
    parser.add_argument(
        "--asm-balance",
        type=str,
        choices=["none", "weighted"],
        default="none",
        help="ASM class-balancing mode (weighted = inverse-sqrt sample weighting).",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Directory for OOF prediction files (default: outputs/exp4_predictions).",
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

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Filter experiments if specific model requested
    if args.model == "all":
        experiments = EXPERIMENTS
    else:
        experiments = [exp for exp in EXPERIMENTS if exp["model"] == args.model]

    # Run experiments
    all_results = run_all_experiments(
        experiments=experiments,
        device=device,
        log_predictions=args.log_predictions,
        predictions_dir=Path(args.predictions_dir) if args.predictions_dir else None,
        asm_balance_mode=args.asm_balance,
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
        output_path = RESULTS_DIR / f"results_{timestamp}.json"

    save_results(all_results, output_path)


if __name__ == "__main__":
    main()
