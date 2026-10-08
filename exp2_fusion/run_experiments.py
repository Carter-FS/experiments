"""Run Experiment 2: EEG + SMILES fusion experiments."""

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch

from .config import EXPERIMENTS, OUTPUTS_DIR, SMILES_EMBED_DIMS
from .utils.logging_utils import setup_logging, log_environment_info, log_exception
from shared.cv_splits import add_cv_args, cv_suffix
from shared.cv_splits import apply_cv_args  # noqa: E402

# Set up module logger (will be configured in main)
logger = logging.getLogger("exp2")


def check_environment() -> bool:
    """Check if required dependencies are available."""
    logger.info("Checking environment...")

    # Check EEG encoder availability
    from .models.eeg_encoders import is_labram_available, get_labram_import_error

    if is_labram_available():
        logger.info("LaBraM encoder: available")
    else:
        logger.warning(f"LaBraM encoder: NOT available - {get_labram_import_error()}")
        logger.warning("Use --eeg-encoder simplecnn to avoid this issue")

    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA: available ({torch.cuda.get_device_name(0)})")
    else:
        logger.warning("CUDA: not available (will use CPU)")

    # Check data imports work
    try:
        from .data_pipeline import prepare_data
        from .training import run_cross_validation
        logger.info("Core modules: importable")
        return True
    except ImportError as e:
        logger.error(f"Core module import failed: {e}")
        return False


def run_all_experiments(
    eeg_encoder: str = "labram",
    smiles_model: str = None,
    fusion_type: str = None,
    dry_run: bool = False,
    output_dir: Path = OUTPUTS_DIR / "exp2_results",
    log_predictions: bool = False,
    predictions_dir: Path = None,
    asm_balance_mode: str = "none",
    splitter: str = "legacy",
    inner_val: float = 0.0,
):
    """Run all configured experiments.

    Args:
        eeg_encoder: EEG encoder type to use ('labram', 'simplecnn', 'eeg2vec').
        smiles_model: SMILES model filter (None = all).
        fusion_type: Fusion type filter (None = all).
        dry_run: If True, just print what would run.
        output_dir: Directory for results.
        splitter: Outer splitter ('legacy' = original multilabel splits).
        inner_val: Inner early-stopping fraction (0 = legacy protocol).
    """
    # Import here to allow --check-env to work even if these fail
    from .data_pipeline import prepare_data
    from .training import run_cross_validation

    # Filter experiments
    experiments = []
    for exp in EXPERIMENTS:
        if smiles_model and exp["smiles_model"] != smiles_model:
            continue
        if fusion_type and exp["fusion"] != fusion_type:
            continue
        experiments.append({**exp, "eeg_model": eeg_encoder})

    if dry_run:
        logger.info("DRY RUN - Would run the following experiments:")
        for exp in experiments:
            logger.info(f"  - {exp['eeg_model']} + {exp['smiles_model']} ({exp['fusion']})")
        return

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Running {len(experiments)} experiments")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    # Empty for the legacy protocol, so archived result/prediction names hold.
    protocol_suffix = cv_suffix(splitter, inner_val)
    results_suffix = protocol_suffix
    if protocol_suffix:
        results_suffix = {"weighted": "_asmweighted", "stratified_batch": "_asmstratbatch"}.get(
            asm_balance_mode, "") + protocol_suffix
    failed = []

    # Run experiments
    for i, exp in enumerate(experiments):
        exp_name = f"{exp['eeg_model']} + {exp['smiles_model']} ({exp['fusion']})"
        logger.info(f"[{i+1}/{len(experiments)}] Starting: {exp_name}")

        try:
            # Load data for this SMILES model
            smiles_model_name = exp["smiles_model"]
            logger.info(f"Loading data for SMILES model: {smiles_model_name}")
            eeg_data, smiles_embeddings, smiles_indices, df = prepare_data(
                smiles_model=smiles_model_name,
            )
            logger.info(f"Loaded {len(df)} patients with EEG data")

            # Run cross-validation
            logger.info("Starting cross-validation...")
            pred_logger = None
            if log_predictions:
                from shared.prediction_logger import PredictionLogger
                pred_dir = predictions_dir if predictions_dir is not None else OUTPUTS_DIR / "exp2_predictions"
                exp_id = f"exp2_{exp['eeg_model']}_{smiles_model_name}_{exp['fusion']}"
                suffix = {"weighted": "_asmweighted", "stratified_batch": "_asmstratbatch"}.get(asm_balance_mode, "")
                suffix += protocol_suffix
                pred_logger = PredictionLogger(
                    exp_id=exp_id, output_dir=pred_dir,
                    filename=f"predictions_oof_{exp_id}{suffix}.json",
                    metadata={"splitter": splitter, "inner_val": inner_val, "asm_balance": asm_balance_mode},
                )
            results = run_cross_validation(
                eeg_data=eeg_data,
                smiles_embeddings=smiles_embeddings,
                smiles_indices=smiles_indices,
                df=df,
                fusion_type=exp["fusion"],
                eeg_encoder_type=exp["eeg_model"],
                smiles_model=smiles_model_name,
                device=device,
                asm_balance_mode=asm_balance_mode,
                verbose=True,
                prediction_logger=pred_logger,
                splitter=splitter,
                inner_val=inner_val,
            )
            if pred_logger is not None:
                saved = pred_logger.save()
                logger.info(f"Per-fold predictions written to {saved}")

            results["timestamp"] = datetime.now().isoformat()
            all_results.append(results)

            # Save individual result
            result_file = output_dir / f"{results['experiment']}{results_suffix}.json"
            with open(result_file, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(
                f"[{i+1}/{len(experiments)}] Complete: "
                f"Acc={results['accuracy']['mean']:.3f}, "
                f"AUC={results['auc']['mean']:.3f}, "
                f"F1={results['f1']['mean']:.3f}"
            )

        except Exception as e:
            log_exception(logger, e, f"Experiment {exp_name} failed")
            logger.error(f"Skipping experiment {exp_name} due to error")
            failed.append(exp_name)
            continue

    if not all_results:
        raise RuntimeError("No experiments completed successfully!")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_experiments": len(all_results),
        "experiments": all_results,
    }
    encoders = "".join(f"_{e}" for e in sorted({exp["eeg_model"] for exp in experiments})) if results_suffix else ""
    summary_file = output_dir / f"summary{encoders}{results_suffix}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    if failed:
        raise RuntimeError(f"{len(failed)} experiment(s) failed: {failed}")

    logger.info(f"Results saved to {output_dir}")

    # Print summary table
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Experiment':<40} {'Acc':<10} {'AUC':<10} {'F1':<10}")
    logger.info("-" * 70)
    for r in sorted(all_results, key=lambda x: -x["auc"]["mean"]):
        name = r["experiment"]
        acc = f"{r['accuracy']['mean']:.3f}"
        auc = f"{r['auc']['mean']:.3f}"
        f1 = f"{r['f1']['mean']:.3f}"
        logger.info(f"{name:<40} {acc:<10} {auc:<10} {f1:<10}")


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 2: EEG + SMILES fusion")
    parser.add_argument("--eeg-encoder", type=str, default="simplecnn",
                        choices=["labram", "simplecnn", "eeg2vec"],
                        help="EEG encoder type (default: simplecnn). The pre-specified "
                             "exp2_eeg2vec_chemberta_mlp row is --eeg-encoder eeg2vec "
                             "--smiles-model chemberta --fusion mlp.")
    parser.add_argument("--smiles-model", type=str, default=None,
                        choices=["chemberta", "smilestrf"],
                        help="SMILES model to use (default: all)")
    parser.add_argument("--fusion", type=str, default=None,
                        choices=["mlp", "fusemoe"],
                        help="Fusion type (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiments without running")
    parser.add_argument("--check-env", action="store_true",
                        help="Check environment and exit")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")
    parser.add_argument("--log-predictions", action="store_true",
                        help="Dump per-fold OOF predictions to outputs/exp2_predictions/")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable deterministic training (seeds, cuDNN deterministic)")
    parser.add_argument("--asm-balance", type=str, default="none",
                        choices=["none", "weighted"],
                        help="ASM class-balancing mode (weighted = inverse-sqrt sample weighting).")
    parser.add_argument("--predictions-dir", type=str, default=None,
                        help="Directory for OOF prediction files (default: outputs/exp2_predictions).")
    add_cv_args(parser)

    args = parser.parse_args()

    apply_cv_args(args)  # repeated-CV seed and refit protocol (plan B.2)

    if args.deterministic:
        from shared.determinism import enable_determinism
        enable_determinism()

    # Set up logging
    log_level = getattr(logging, args.log_level)
    script_dir = Path(__file__).parent.parent  # experiments/
    log_dir = script_dir / "logs"

    global logger
    logger = setup_logging("exp2", log_dir=str(log_dir), level=log_level)

    # Log environment info
    log_environment_info(logger)

    # Check environment only
    if args.check_env:
        success = check_environment()
        if success:
            logger.info("Environment check passed!")
            sys.exit(0)
        else:
            logger.error("Environment check failed!")
            sys.exit(1)

    # Run experiments with exception handling
    try:
        run_all_experiments(
            eeg_encoder=args.eeg_encoder,
            smiles_model=args.smiles_model,
            fusion_type=args.fusion,
            dry_run=args.dry_run,
            log_predictions=args.log_predictions,
            predictions_dir=Path(args.predictions_dir) if args.predictions_dir else None,
            asm_balance_mode=args.asm_balance,
            splitter=args.splitter,
            inner_val=args.inner_val,
        )
        logger.info("Experiment 2 completed successfully")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        log_exception(logger, e, "Fatal error in Experiment 2")
        sys.exit(1)


if __name__ == "__main__":
    main()
