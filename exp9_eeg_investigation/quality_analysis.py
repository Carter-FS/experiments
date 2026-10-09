"""EEG quality analysis to understand variance sources.

This script computes quality metrics for all EEG recordings and analyses
their relationship with model performance.
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add parent directory for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from exp2_fusion.eeg_pipeline import (
    EEGPreprocessor,
    build_patient_eeg_map,
    compute_snr,
    detect_artifacts,
    detect_flatlines,
    compute_channel_correlation,
    compute_quality_score,
    load_edf,
    apply_filters,
)
from exp2_fusion.config import EEG_DIR, CSV_PATH
from exp8_stratification.data_cleaning import load_and_clean_data
from .config import RESULTS_DIR, EXP5C_FOLD_AUCS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_all_quality_metrics(
    eeg_dir: Path = EEG_DIR,
    target_sr: int = 200,
    save_path: Path = None,
) -> Dict[str, Dict]:
    """Compute quality metrics for all EEG files.

    Args:
        eeg_dir: Directory containing EEG files.
        target_sr: Target sample rate.
        save_path: Optional path to save results.

    Returns:
        Dict mapping patient ID to quality metrics.
    """
    patient_eeg_map = build_patient_eeg_map(eeg_dir)
    logger.info(f"Found {len(patient_eeg_map)} patients with EEG files")

    preprocessor = EEGPreprocessor(target_sr=target_sr, compute_quality=True)
    quality_results = {}

    for i, (pid, edf_path) in enumerate(patient_eeg_map.items()):
        if i % 20 == 0:
            logger.info(f"Processing {i}/{len(patient_eeg_map)} EEG files...")

        try:
            result = preprocessor.process(edf_path, return_quality=True)
            if result is not None:
                windows, padding_mask, n_channels, quality_metrics = result
                quality_results[pid] = quality_metrics
        except Exception as e:
            logger.warning(f"Failed to process recording {i + 1}: {e}")
            continue

    logger.info(f"Computed quality for {len(quality_results)} patients")

    if save_path:
        with open(save_path, "w") as f:
            json.dump(quality_results, f, indent=2)
        logger.info(f"Saved quality metrics to {save_path}")

    return quality_results


def summarise_quality_metrics(quality_results: Dict[str, Dict]) -> pd.DataFrame:
    """Create summary DataFrame from quality results.

    Args:
        quality_results: Dict from compute_all_quality_metrics.

    Returns:
        DataFrame with one row per patient.
    """
    rows = []
    for pid, metrics in quality_results.items():
        scores = metrics.get("quality_scores", {})
        row = {
            "pid": pid,
            "n_channels": metrics.get("n_channels", 0),
            "n_valid_windows": metrics.get("n_valid_windows", 0),
            "duration_sec": metrics.get("extracted_duration_sec", 0),
            "overall_quality": scores.get("overall_quality_score", 0),
            "snr_score": scores.get("snr_score", 0),
            "artifact_score": scores.get("artifact_score", 0),
            "flatline_score": scores.get("flatline_score", 0),
            "correlation_score": scores.get("correlation_score", 0),
            "mean_snr_db": scores.get("mean_snr_db", 0),
            "artifact_ratio": metrics.get("artifact_stats", {}).get("overall_artifact_ratio", 0),
            "flatline_ratio": metrics.get("flatline_stats", {}).get("overall_flatline_ratio", 0),
            "mean_correlation": metrics.get("correlation_stats", {}).get("mean_correlation", 0),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def analyse_quality_vs_performance(
    quality_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    fold_aucs: List[float] = None,
) -> Dict:
    """Analyse relationship between EEG quality and model performance.

    Args:
        quality_df: DataFrame from summarise_quality_metrics.
        clinical_df: DataFrame with clinical features and outcome.
        fold_aucs: Per-fold AUC values.

    Returns:
        Analysis results dict.
    """
    if fold_aucs is None:
        fold_aucs = EXP5C_FOLD_AUCS

    # Merge quality with clinical data
    quality_df["pid"] = quality_df["pid"].astype(str)
    clinical_df["pid"] = clinical_df["pid"].astype(str)

    merged = quality_df.merge(clinical_df[["pid", "outcome"]], on="pid", how="inner")
    logger.info(f"Merged {len(merged)} patients with both quality and outcome data")

    # Quality statistics by outcome
    outcome_stats = {}
    for outcome in [0, 1]:
        subset = merged[merged["outcome"] == outcome]
        outcome_stats[f"outcome_{outcome}"] = {
            "n": len(subset),
            "mean_quality": float(subset["overall_quality"].mean()),
            "mean_snr_db": float(subset["mean_snr_db"].mean()),
            "mean_artifact_ratio": float(subset["artifact_ratio"].mean()),
            "mean_duration": float(subset["duration_sec"].mean()),
        }

    # Quality distribution statistics
    quality_stats = {
        "n_patients": len(merged),
        "quality_mean": float(merged["overall_quality"].mean()),
        "quality_std": float(merged["overall_quality"].std()),
        "quality_min": float(merged["overall_quality"].min()),
        "quality_max": float(merged["overall_quality"].max()),
        "snr_mean_db": float(merged["mean_snr_db"].mean()),
        "snr_std_db": float(merged["mean_snr_db"].std()),
        "low_quality_patients": int((merged["overall_quality"] < 0.5).sum()),
        "high_artifact_patients": int((merged["artifact_ratio"] > 0.01).sum()),
    }

    # Correlation with outcome
    if len(merged) > 10:
        correlations = {}
        for col in ["overall_quality", "mean_snr_db", "artifact_ratio", "duration_sec", "n_valid_windows"]:
            corr = merged[col].corr(merged["outcome"])
            correlations[col] = float(corr) if not np.isnan(corr) else 0.0
    else:
        correlations = {}

    return {
        "quality_stats": quality_stats,
        "outcome_stats": outcome_stats,
        "correlations_with_outcome": correlations,
    }


def identify_problem_recordings(
    quality_df: pd.DataFrame,
    quality_threshold: float = 0.4,
    artifact_threshold: float = 0.02,
) -> pd.DataFrame:
    """Identify low-quality EEG recordings.

    Args:
        quality_df: DataFrame from summarise_quality_metrics.
        quality_threshold: Minimum acceptable quality score.
        artifact_threshold: Maximum acceptable artifact ratio.

    Returns:
        DataFrame of problem recordings.
    """
    problems = quality_df[
        (quality_df["overall_quality"] < quality_threshold) |
        (quality_df["artifact_ratio"] > artifact_threshold)
    ].copy()

    problems["issue"] = ""
    problems.loc[problems["overall_quality"] < quality_threshold, "issue"] += "low_quality; "
    problems.loc[problems["artifact_ratio"] > artifact_threshold, "issue"] += "high_artifacts; "

    return problems.sort_values("overall_quality")


def print_quality_report(analysis: Dict, quality_df: pd.DataFrame):
    """Print formatted quality analysis report."""
    print("\n" + "=" * 70)
    print("EEG QUALITY ANALYSIS REPORT")
    print("=" * 70)

    stats = analysis["quality_stats"]
    print(f"\n--- Quality Distribution (n={stats['n_patients']}) ---")
    print(f"Overall quality: {stats['quality_mean']:.3f} +/- {stats['quality_std']:.3f}")
    print(f"Quality range: [{stats['quality_min']:.3f}, {stats['quality_max']:.3f}]")
    print(f"Mean SNR: {stats['snr_mean_db']:.1f} +/- {stats['snr_std_db']:.1f} dB")
    print(f"Low quality (<0.5): {stats['low_quality_patients']} patients")
    print(f"High artifact (>1%): {stats['high_artifact_patients']} patients")

    print("\n--- Quality by Outcome ---")
    for key, vals in analysis["outcome_stats"].items():
        print(f"{key}: n={vals['n']}, quality={vals['mean_quality']:.3f}, "
              f"SNR={vals['mean_snr_db']:.1f}dB, duration={vals['mean_duration']:.0f}s")

    print("\n--- Correlations with Outcome ---")
    for feat, corr in analysis.get("correlations_with_outcome", {}).items():
        print(f"  {feat}: r={corr:.3f}")

    print("\n--- Quality Percentiles ---")
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        val = quality_df["overall_quality"].quantile(p/100)
        print(f"  P{p}: {val:.3f}")


def run_quality_analysis(
    recompute: bool = False,
    save_results: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """Run full EEG quality analysis.

    Args:
        recompute: Whether to recompute quality metrics (slow).
        save_results: Whether to save results to disk.

    Returns:
        Tuple of (quality_df, analysis_results).
    """
    cache_path = RESULTS_DIR / "eeg_quality_metrics.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load or compute quality metrics
    if cache_path.exists() and not recompute:
        logger.info(f"Loading cached quality metrics from {cache_path}")
        with open(cache_path, "r") as f:
            quality_results = json.load(f)
    else:
        logger.info("Computing quality metrics (this may take a while)...")
        quality_results = compute_all_quality_metrics(
            save_path=cache_path if save_results else None
        )

    # Create summary DataFrame
    quality_df = summarise_quality_metrics(quality_results)
    logger.info(f"Quality summary: {len(quality_df)} patients")

    # Load clinical data
    clinical_df, _ = load_and_clean_data()

    # Analyse quality vs performance
    analysis = analyse_quality_vs_performance(quality_df, clinical_df)

    # Identify problem recordings
    problems = identify_problem_recordings(quality_df)
    analysis["problem_recordings"] = len(problems)

    # Print report
    print_quality_report(analysis, quality_df)

    if len(problems) > 0:
        print(f"\n--- Problem Recordings ({len(problems)}) ---")
        print(problems[["overall_quality", "artifact_ratio", "mean_snr_db", "issue"]].head(10).to_string(index=False))  # no patient ids on the console

    # Save summary
    if save_results:
        summary_path = RESULTS_DIR / "eeg_quality_summary.csv"
        quality_df.to_csv(summary_path, index=False)
        logger.info(f"Saved quality summary to {summary_path}")

        analysis_path = RESULTS_DIR / "eeg_quality_analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Saved analysis to {analysis_path}")

    return quality_df, analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EEG Quality Analysis")
    parser.add_argument("--recompute", action="store_true", help="Recompute quality metrics")
    args = parser.parse_args()

    quality_df, analysis = run_quality_analysis(recompute=args.recompute)
