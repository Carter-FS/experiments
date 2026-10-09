"""Run Experiment 7 configurations with cross-validation.

Usage:
    python -m exp7_all_modalities.run_experiments
    python -m exp7_all_modalities.run_experiments --exp 7a
    python -m exp7_all_modalities.run_experiments --exp 7b
    python -m exp7_all_modalities.run_experiments --mode predictions --output_dir outputs/exp7_predictions
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from shared.cv_splits import add_cv_args, current_seed, cv_suffix, outer_splits, refit_folds, rethreshold, smoke_tag
from shared.epoch_selection import refit_protocol, run_outer_fold
from shared.cv_splits import apply_cv_args  # noqa: E402
from shared.prediction_logger import protocol_metadata, run_provenance

from .config import ASM_NAME_MAPPING, CV_CONFIG, EXPERIMENTS, RESULTS_DIR
from .data_pipeline import create_quad_modality_datasets, prepare_quad_modality_data
from .training import run_cross_validation, train_fold_with_predictions

logger = logging.getLogger("exp7")


def compute_summary(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of values."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def run_experiment(
    exp_config: Dict,
    device: torch.device,
    asm_balance_mode: str = "none",
    splitter: str = "legacy",
    inner_val: float = 0.0,
) -> Dict:
    """Run a single experiment configuration."""
    exp_name = exp_config["name"]
    fusion = exp_config["fusion"]
    text_model = exp_config["text_model"]
    smiles_model = exp_config["smiles_model"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Running experiment: {exp_name} (ASM balance: {asm_balance_mode})")
    logger.info(f"{'='*60}")

    fold_metrics = run_cross_validation(
        fusion=fusion,
        text_model=text_model,
        smiles_model=smiles_model,
        device=device,
        asm_balance_mode=asm_balance_mode,
        splitter=splitter,
        inner_val=inner_val,
    )

    # Compute summary statistics
    summary = {}
    for metric_name, values in fold_metrics.items():
        summary[metric_name] = compute_summary(values)

    return {
        "config": exp_config,
        "fold_metrics": fold_metrics,
        "summary": summary,
        "metadata": {**protocol_metadata(inner_val), "splitter": splitter, "inner_val": inner_val,
                     "asm_balance": asm_balance_mode},
    }


def run_all_experiments(
    experiments: Optional[List[Dict]] = None,
    device: torch.device = None,
    asm_balance_mode: str = "none",
    splitter: str = "legacy",
    inner_val: float = 0.0,
) -> Dict[str, Dict]:
    """Run all experiments and collect results."""
    if experiments is None:
        experiments = EXPERIMENTS

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = {}

    for exp_config in experiments:
        exp_name = exp_config["name"]
        results = run_experiment(
            exp_config, device, asm_balance_mode=asm_balance_mode,
            splitter=splitter, inner_val=inner_val,
        )
        all_results[exp_name] = results

    return all_results


def print_results_table(all_results: Dict[str, Dict]):
    """Print results in formatted table."""
    print("\n" + "=" * 90)
    print("EXPERIMENT 7 RESULTS SUMMARY")
    print("=" * 90)

    # Header
    print(f"{'Experiment':<35} {'AUC':>18} {'Bal Acc':>18} {'F1 Tuned':>18}")
    print("-" * 90)

    for exp_name, results in all_results.items():
        summary = results["summary"]
        auc = f"{summary['auc']['mean']:.4f} +/- {summary['auc']['std']:.4f}"
        bal_acc = f"{summary['balanced_acc_tuned']['mean']:.4f} +/- {summary['balanced_acc_tuned']['std']:.4f}"
        f1_tuned = f"{summary['f1_tuned']['mean']:.4f} +/- {summary['f1_tuned']['std']:.4f}"

        print(f"{exp_name:<35} {auc:>18} {bal_acc:>18} {f1_tuned:>18}")

    print("=" * 90)

    # Comparison with baselines
    print("\nComparison with baselines:")
    print("  Exp3b (Triple, no clinical): AUC 0.753")
    print("  Exp6a (Clinical + Text + SMILES): AUC 0.702")
    print("  Exp4a (Clinical only): AUC 0.664")
    print()

    for exp_name, results in all_results.items():
        auc_mean = results["summary"]["auc"]["mean"]
        diff_3b = auc_mean - 0.753
        diff_6a = auc_mean - 0.702
        sign_3b = "+" if diff_3b >= 0 else ""
        sign_6a = "+" if diff_6a >= 0 else ""
        print(f"  {exp_name}: {auc_mean:.4f} ({sign_3b}{diff_3b:.4f} vs Exp3b, {sign_6a}{diff_6a:.4f} vs Exp6a)")

    # Print per-fold details
    print("\nPer-fold AUC values:")
    for exp_name, results in all_results.items():
        aucs = results["fold_metrics"]["auc"]
        fold_str = ", ".join([f"{v:.4f}" for v in aucs])
        print(f"  {exp_name}: [{fold_str}]")


def save_results(all_results: Dict[str, Dict], output_path: Path):
    """Save results to JSON."""
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


def _normalise_asm(name: str) -> str:
    """Normalise an ASM string via ASM_NAME_MAPPING."""
    s = str(name).strip()
    return ASM_NAME_MAPPING.get(s, s)


def _top_n_asms_for_cohort(df, n: int = 5) -> List[str]:
    """Return the top-n most-prescribed ASM short codes in this cohort.

    The shortcode is the original CSV abbreviation (e.g. 'LEV', 'VPA') after
    string strip and case normalisation through ASM_NAME_MAPPING. We collapse
    duplicates that map to the same canonical full name so 'CBZ' and 'cBZ'
    count as one.
    """
    counts: Dict[str, int] = {}
    short_for_canon: Dict[str, str] = {}
    for raw in df["ASM"].astype(str):
        short = raw.strip()
        canon = ASM_NAME_MAPPING.get(short, short)
        # Pick the upper-cased shortcode if available, otherwise keep raw.
        preferred = short.upper() if short.upper() in ASM_NAME_MAPPING else short
        counts[canon] = counts.get(canon, 0) + 1
        # First seen preferred form for this canonical name.
        short_for_canon.setdefault(canon, preferred)

    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top = [short_for_canon[canon] for canon, _ in ordered[:n]]
    return top


def _build_candidate_smiles(
    top_asms: List[str],
    smiles_embeddings: np.ndarray,
    smiles_indices: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """Resolve each top ASM shortcode to its SMILES embedding vector."""
    out: Dict[str, np.ndarray] = {}
    for short in top_asms:
        canon = ASM_NAME_MAPPING.get(short, short)
        if canon not in smiles_indices:
            logger.warning(f"  Skipping ASM '{short}' (canon '{canon}') - not in smiles_indices.")
            continue
        idx = smiles_indices[canon]
        out[short] = np.asarray(smiles_embeddings[idx], dtype=np.float32)
    return out


def _save_predictions_json(payload: Dict, path: Path):
    """Serialise prediction payload, converting numpy/torch types as needed."""
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


def run_exp7a_with_predictions(
    output_dir: Path,
    top_n_asms: int = 5,
    text_model: str = "clinicalbert",
    smiles_model: str = "chemberta",
    device: torch.device = None,
    asm_balance_mode: str = "none",
    output_suffix: str = "",
    fusion: str = "mlp",
    splitter: str = "legacy",
    inner_val: float = 0.0,
) -> Dict[str, Path]:
    """Run Exp7a with per-patient and ASM-swap prediction logging.

    Performs 5-fold CV: trains on 4 folds, predicts on the held-out fold,
    and also predicts each held-out patient under each of the top-n
    most-prescribed ASMs (counterfactual SMILES swap). Then refits a final
    model on the full cohort using a 10% random early-stopping split and
    predicts for all patients across the same top-n ASMs (in-sample).

    ``splitter``/``inner_val`` set the CV protocol. Legacy (the default)
    early-stops on the held-out fold. Clean runs (e.g. multilabel, 0.2)
    early-stop on an inner split of the training folds and predict the
    held-out fold once with the restored weights. The all-data refit is
    unaffected.

    Writes ``predictions_oof.json`` and ``predictions_in_sample.json``
    inside ``output_dir`` (plus the protocol suffix for non-legacy runs).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Preparing quad-modality data for Exp7a predictions run")
    df, smiles_embeddings, smiles_indices, text_embeddings, eeg_data = prepare_quad_modality_data(
        text_model, smiles_model
    )
    outcomes = df["outcome"].values
    logger.info(f"  Cohort size: {len(df)} patients")

    # Determine top-N ASMs from this cohort.
    top_asms = _top_n_asms_for_cohort(df, n=top_n_asms)
    logger.info(f"  Top-{top_n_asms} ASMs: {top_asms}")
    candidate_smiles = _build_candidate_smiles(top_asms, smiles_embeddings, smiles_indices)
    asms_used = list(candidate_smiles.keys())

    # ------------------------------------------------------------------
    # 5-fold CV with prediction logging.
    # ------------------------------------------------------------------
    splits = outer_splits(
        df, mode=splitter, n_splits=CV_CONFIG["n_splits"], seed=CV_CONFIG["random_state"],
    )

    folds_payload: List[Dict] = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"Fold {fold + 1}/{CV_CONFIG['n_splits']}")
        # Clinical preprocessor is fitted on the fit set only. Clean runs select
        # on inner data (inner split, or inner folds + refit) and predict the outer fold once.
        result, selection = run_outer_fold(
            outcomes, train_idx, val_idx, fold, inner_val,
            make_datasets=lambda a, b: create_quad_modality_datasets(
                df, smiles_embeddings, smiles_indices, text_embeddings, eeg_data, a, b,
                return_pid=True,
            )[:2],
            train=lambda tr, va, **kw: train_fold_with_predictions(
                tr, va, fusion=fusion, text_model=text_model, smiles_model=smiles_model,
                device=device, fold=fold, candidate_smiles=candidate_smiles,
                asm_balance_mode=asm_balance_mode, **kw,
            ),
            log=logger.info,
        )

        # Skip non-scalar metric entries (y_prob, y_true added in Stage A).
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

    exp_name = "exp7b" if fusion == "moe" else "exp7a"
    fusion_part = "_7b" if fusion == "moe" else ""
    # Protocol suffix goes after the fusion/balance suffixes; it is empty for
    # the legacy protocol so archived filenames are unchanged.
    suffix_part = (f"_{output_suffix}" if output_suffix else "") + cv_suffix(splitter, inner_val)
    metadata = {
        **protocol_metadata(inner_val),
        "splitter": splitter,
        "inner_val": inner_val,
        "asm_balance": asm_balance_mode,
        "provenance": run_provenance(),
    }
    oof_payload = {
        "experiment": exp_name + (f"_{output_suffix}" if output_suffix else ""),
        "asm_balance_mode": asm_balance_mode,
        "text_model": text_model,
        "smiles_model": smiles_model,
        "asms": asms_used,
        "cv_random_state": current_seed(CV_CONFIG["random_state"]),
        "n_splits": CV_CONFIG["n_splits"],
        "folds": folds_payload,
        "metadata": metadata,
    }
    oof_path = output_dir / f"predictions_oof{fusion_part}{suffix_part}.json"
    _save_predictions_json(oof_payload, oof_path)

    # ------------------------------------------------------------------
    # Final all-data refit with 10% random early-stop split.
    # ------------------------------------------------------------------
    n = len(df)
    all_selection = None
    if refit_folds():
        # Refit protocol (analysis plan B.2): the epoch count and threshold come
        # from inner folds over every patient, then one model is trained on
        # all of them and predicts the full cohort (in sample, as before).
        logger.info("Final all-data refit (inner-fold epoch selection over every patient)")
        all_idx = np.arange(n)

        def make_all(a, b):
            return create_quad_modality_datasets(
                df, smiles_embeddings, smiles_indices, text_embeddings, eeg_data, a, b,
                return_pid=True,
            )[:2]

        def train_all(tr, va, **kw):
            return train_fold_with_predictions(
                tr, va, fusion=fusion, text_model=text_model, smiles_model=smiles_model,
                device=device, fold=-1, candidate_smiles=candidate_smiles,
                asm_balance_mode=asm_balance_mode, **kw,
            )

        def run_inner(fit, val):
            trace: Dict = {}
            train_all(*make_all(fit, val), trace=trace)
            return trace

        def run_refit(full, choice):
            train_ds_full, full_eval_ds = make_all(full, full)
            return train_all(train_ds_full, None, test_dataset=full_eval_ds,
                             fixed_epochs=choice.epoch, refit_choice=choice)

        choice, refit_result = refit_protocol(
            outcomes, outcomes, all_idx, -1, run_inner, run_refit, n_inner=refit_folds(),
        )
        refit_result["metrics"] = rethreshold(refit_result["metrics"], choice.threshold)
        all_selection = choice.as_metadata()
        pids_full = refit_result["val_pids"]
        y_true_full = refit_result["val_y_true"]
        y_prob_full = refit_result["val_y_prob"]
        y_prob_per_asm_full = refit_result["val_y_prob_per_asm"]
        early_stop_frac = None
    else:
        logger.info("Final all-data refit (90/10 random split for early stopping)")
        rng = np.random.RandomState(current_seed(CV_CONFIG["random_state"]))
        perm = rng.permutation(n)
        n_val = max(1, int(round(0.1 * n)))
        val_idx_full = perm[:n_val]
        train_idx_full = perm[n_val:]

        train_ds_full, val_ds_full, _ = create_quad_modality_datasets(
            df,
            smiles_embeddings,
            smiles_indices,
            text_embeddings,
            eeg_data,
            train_idx_full,
            val_idx_full,
            return_pid=True,
        )
        logger.info(f"  Refit train: {len(train_ds_full)}, early-stop val: {len(val_ds_full)}")

        refit_result = train_fold_with_predictions(
            train_ds_full,
            val_ds_full,
            fusion=fusion,
            text_model=text_model,
            smiles_model=smiles_model,
            device=device,
            fold=-1,
            candidate_smiles=candidate_smiles,
            asm_balance_mode=asm_balance_mode,
        )

        # Predict on the FULL cohort using the refit model. Build a "val
        # dataset" that contains every patient by passing all indices as the
        # val split.
        logger.info("Predicting on full cohort with refit model")
        all_idx = np.arange(n)
        # The training preprocessor in create_quad_modality_datasets is fitted
        # on the train split and applied to the val split. To avoid refitting
        # on different data, we re-build using the same train indices used
        # above so the preprocessor is identical, but with val_idx = all
        # indices.
        train_ds_for_pp, full_eval_ds, _ = create_quad_modality_datasets(
            df,
            smiles_embeddings,
            smiles_indices,
            text_embeddings,
            eeg_data,
            train_idx_full,
            all_idx,
            return_pid=True,
        )
        del train_ds_for_pp

        # Inference on full cohort using the model we just trained (load best
        # weights manually from refit_result and run predictions).
        from torch.utils.data import DataLoader as _DL
        from .models import get_model as _get_model
        from .training import _predict_with_smiles_override
        from .config import MLP_CONFIG as _MLP_CONFIG

        model_full = _get_model(fusion=fusion, text_model=text_model, smiles_model=smiles_model, device=device)
        if refit_result["model_state_dict"] is not None:
            model_full.load_state_dict(refit_result["model_state_dict"])

        full_loader = _DL(
            full_eval_ds,
            batch_size=_MLP_CONFIG["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        pids_full, y_true_full, y_prob_full = _predict_with_smiles_override(
            model_full, full_loader, device, fusion=fusion, smiles_override=None
        )
        y_prob_per_asm_full: Dict[str, List[float]] = {}
        for asm_name, smiles_vec in candidate_smiles.items():
            override = torch.from_numpy(np.asarray(smiles_vec, dtype=np.float32))
            _, _, probs = _predict_with_smiles_override(
                model_full, full_loader, device, fusion=fusion, smiles_override=override
            )
            y_prob_per_asm_full[asm_name] = probs
        early_stop_frac = 0.1

    in_sample_payload = {
        "experiment": exp_name + (f"_{output_suffix}" if output_suffix else ""),
        "asm_balance_mode": asm_balance_mode,
        "text_model": text_model,
        "smiles_model": smiles_model,
        "asms": asms_used,
        "refit_random_state": CV_CONFIG["random_state"],
        "early_stop_frac": early_stop_frac,
        **({"selection": all_selection} if all_selection else {}),
        "metrics": {k: float(v) for k, v in refit_result["metrics"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
        "pids": pids_full,
        "y_true": y_true_full,
        "y_prob": y_prob_full,
        "y_prob_per_asm": y_prob_per_asm_full,
        "metadata": metadata,
    }
    in_sample_path = output_dir / f"predictions_in_sample{fusion_part}{suffix_part}.json"
    _save_predictions_json(in_sample_payload, in_sample_path)

    return {"oof": oof_path, "in_sample": in_sample_path}


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Experiment 7: All Four Modalities Fusion"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cv", "predictions"],
        default="cv",
        help="'cv' (default) runs the standard CV loop. 'predictions' runs "
             "Exp7a with per-patient and ASM-swap prediction logging.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for the predictions mode "
             "(default: outputs/exp7_predictions).",
    )
    parser.add_argument(
        "--top_n_asms",
        type=int,
        default=5,
        help="Number of top-prescribed ASMs to include for counterfactual "
             "swaps (default: 5).",
    )
    parser.add_argument(
        "--exp",
        type=str,
        choices=["7a", "7b", "all"],
        default="all",
        help="Experiment to run in cv mode (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for cv mode (default: outputs/exp7_results/results_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
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
        help="ASM-balancing mode (Stage B): 'weighted' applies inverse-sqrt sample weights, 'stratified_batch' uses a per-batch sampler that includes every ASM.",
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

    if args.mode == "predictions":
        out_dir = Path(args.output_dir) if args.output_dir else (RESULTS_DIR.parent / "exp7_predictions")
        suffix = ""
        if args.asm_balance == "weighted":
            suffix = "asmweighted"
        elif args.asm_balance == "stratified_batch":
            suffix = "asmstratbatch"
        run_exp7a_with_predictions(
            output_dir=out_dir,
            top_n_asms=args.top_n_asms,
            device=device,
            asm_balance_mode=args.asm_balance,
            output_suffix=suffix,
            fusion="moe" if args.exp == "7b" else "mlp",
            splitter=args.splitter,
            inner_val=args.inner_val,
        )
        return

    # Default cv mode (legacy flow).
    if args.exp == "all":
        experiments = EXPERIMENTS
    elif args.exp == "7a":
        experiments = [exp for exp in EXPERIMENTS if exp["fusion"] == "mlp"]
    elif args.exp == "7b":
        experiments = [exp for exp in EXPERIMENTS if exp["fusion"] == "moe"]
    else:
        experiments = EXPERIMENTS

    logger.info(f"Running {len(experiments)} experiment(s)")

    # Run experiments
    all_results = run_all_experiments(
        experiments=experiments,
        device=device,
        asm_balance_mode=args.asm_balance,
        splitter=args.splitter,
        inner_val=args.inner_val,
    )

    # Print results
    print_results_table(all_results)

    # Save results
    if args.output:
        output_path = smoke_tag(Path(args.output))
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = smoke_tag(RESULTS_DIR / f"results_{timestamp}.json")

    save_results(all_results, output_path)


if __name__ == "__main__":
    main()
