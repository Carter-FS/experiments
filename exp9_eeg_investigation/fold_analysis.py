"""Fold composition analysis to understand EEG variance.

This script analyses what makes specific folds perform exceptionally well or poorly.
Particularly focuses on Exp5c fold 4 (AUC 0.866) vs fold 5 (AUC 0.545).
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add parent directory for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from exp8_stratification.stratified_cv import get_multilabel_splits, get_outcome_only_splits
from exp8_stratification.data_cleaning import load_and_clean_data
from exp8_stratification.config import EEG_CACHE_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_eeg_metadata(cache_path: Path = EEG_CACHE_PATH) -> Dict[str, Dict]:
    """Load EEG metadata from cache.

    Returns:
        Dict mapping patient ID to {n_windows, n_padded, n_channels, duration_sec}.
    """
    from shared.eeg_cache import load_cache
    # Only window counts and durations are read, so the unscaled microvolt
    # windows are enough; no model consumes these arrays.
    eeg_cache = load_cache(cache_path, "raw_uv")

    metadata = {}
    for pid, (windows, padding_mask) in eeg_cache.items():
        n_windows = len(windows)
        n_valid = int((~padding_mask).sum())
        n_padded = int(padding_mask.sum())
        n_channels = windows.shape[1] if len(windows) > 0 else 0

        metadata[pid] = {
            "n_windows": n_windows,
            "n_valid_windows": n_valid,
            "n_padded_windows": n_padded,
            "padding_ratio": n_padded / n_windows if n_windows > 0 else 0,
            "n_channels": n_channels,
            "duration_sec": n_valid * 10,  # 10 seconds per window
        }

    return metadata


def analyse_fold_composition(
    df: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    eeg_metadata: Dict[str, Dict] = None,
) -> pd.DataFrame:
    """Analyse clinical and EEG feature distribution per fold.

    Args:
        df: DataFrame with clinical features.
        splits: List of (train_idx, val_idx) tuples.
        eeg_metadata: Optional dict of EEG metadata per patient.

    Returns:
        DataFrame with per-fold statistics.
    """
    results = []

    clinical_features = ["focal", "sex", "outcome"]
    numeric_features = ["age_init"]

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        val_df = df.iloc[val_idx]
        fold_data = {
            "fold": fold_idx + 1,
            "n_val": len(val_idx),
            "n_train": len(train_idx),
        }

        # Clinical feature distributions (validation set)
        for col in clinical_features:
            if col in val_df.columns:
                # Proportion of positive class (1)
                series = val_df[col].dropna()
                if len(series) > 0:
                    fold_data[f"{col}_positive_pct"] = (series == 1).mean() * 100
                    fold_data[f"{col}_n_positive"] = int((series == 1).sum())

        # Numeric features
        for col in numeric_features:
            if col in val_df.columns:
                series = val_df[col].dropna()
                if len(series) > 0:
                    fold_data[f"{col}_mean"] = series.mean()
                    fold_data[f"{col}_std"] = series.std()

        # EEG metadata if available
        if eeg_metadata is not None:
            val_pids = val_df["pid"].values if "pid" in val_df.columns else val_df.index
            eeg_durations = []
            eeg_padding_ratios = []
            eeg_channels = []

            for pid in val_pids:
                if pid in eeg_metadata:
                    meta = eeg_metadata[pid]
                    eeg_durations.append(meta["duration_sec"])
                    eeg_padding_ratios.append(meta["padding_ratio"])
                    eeg_channels.append(meta["n_channels"])

            if eeg_durations:
                fold_data["eeg_duration_mean"] = np.mean(eeg_durations)
                fold_data["eeg_duration_std"] = np.std(eeg_durations)
                fold_data["eeg_duration_min"] = np.min(eeg_durations)
                fold_data["eeg_padding_ratio_mean"] = np.mean(eeg_padding_ratios)
                fold_data["eeg_n_channels_mean"] = np.mean(eeg_channels)
                fold_data["eeg_n_patients"] = len(eeg_durations)

        results.append(fold_data)

    return pd.DataFrame(results)


def correlate_with_performance(
    fold_stats: pd.DataFrame,
    fold_aucs: List[float] = None,
) -> pd.DataFrame:
    """Correlate fold statistics with AUC performance.

    Args:
        fold_stats: DataFrame from analyse_fold_composition.
        fold_aucs: List of AUC values per fold (from Exp5c).

    Returns:
        DataFrame with correlations.
    """
    if fold_aucs is None:
        # Default: Exp5c per-fold AUC values from experiment_findings.md
        # [0.600, 0.604, 0.604, 0.866, 0.545]
        fold_aucs = [0.600, 0.604, 0.604, 0.866, 0.545]

    fold_stats = fold_stats.copy()
    fold_stats["auc"] = fold_aucs

    # Compute correlations with AUC for numeric columns
    correlations = []
    for col in fold_stats.columns:
        if col in ["fold", "auc"]:
            continue
        if fold_stats[col].dtype in [np.float64, np.int64, float, int]:
            series = fold_stats[col].dropna()
            if len(series) >= 3:  # Need at least 3 points
                corr = np.corrcoef(series, fold_stats.loc[series.index, "auc"])[0, 1]
                correlations.append({
                    "feature": col,
                    "correlation_with_auc": corr,
                    "abs_correlation": abs(corr),
                })

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values("abs_correlation", ascending=False)

    return corr_df


def compare_extreme_folds(
    df: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    best_fold_idx: int = 3,  # Fold 4 (0-indexed)
    worst_fold_idx: int = 4,  # Fold 5 (0-indexed)
) -> Dict:
    """Compare characteristics of best vs worst performing folds.

    Args:
        df: DataFrame with clinical features.
        splits: List of (train_idx, val_idx) tuples.
        best_fold_idx: Index of best performing fold.
        worst_fold_idx: Index of worst performing fold.

    Returns:
        Dict with comparison statistics.
    """
    best_val_idx = splits[best_fold_idx][1]
    worst_val_idx = splits[worst_fold_idx][1]

    best_df = df.iloc[best_val_idx]
    worst_df = df.iloc[worst_val_idx]

    comparison = {
        "best_fold": best_fold_idx + 1,
        "worst_fold": worst_fold_idx + 1,
        "best_n": len(best_val_idx),
        "worst_n": len(worst_val_idx),
        "features": {},
    }

    # Compare binary features
    binary_cols = ["focal", "sex", "outcome", "fam_hx", "head", "drug", "alcohol", "cvd", "psy"]
    for col in binary_cols:
        if col in df.columns:
            best_pct = (best_df[col] == 1).mean() * 100
            worst_pct = (worst_df[col] == 1).mean() * 100
            comparison["features"][col] = {
                "best_fold_pct": round(best_pct, 1),
                "worst_fold_pct": round(worst_pct, 1),
                "difference": round(best_pct - worst_pct, 1),
            }

    # Compare age
    if "age_init" in df.columns:
        comparison["features"]["age_init"] = {
            "best_fold_mean": round(best_df["age_init"].mean(), 1),
            "worst_fold_mean": round(worst_df["age_init"].mean(), 1),
            "difference": round(best_df["age_init"].mean() - worst_df["age_init"].mean(), 1),
        }

    return comparison


def run_analysis(
    use_multilabel: bool = False,
    fold_aucs: List[float] = None,
) -> Dict:
    """Run full fold composition analysis.

    Args:
        use_multilabel: Whether to use multi-label stratification.
        fold_aucs: AUC values per fold (defaults to Exp5c outcome-only results).

    Returns:
        Dict with analysis results.
    """
    logger.info("Loading data...")
    df, _ = load_and_clean_data()

    # Try to load EEG metadata
    eeg_metadata = None
    if EEG_CACHE_PATH.exists():
        logger.info("Loading EEG metadata...")
        eeg_metadata = load_eeg_metadata()
        logger.info(f"Loaded EEG data for {len(eeg_metadata)} patients")

    # Generate splits
    logger.info(f"Generating {'multi-label' if use_multilabel else 'outcome-only'} splits...")
    if use_multilabel:
        splits = list(get_multilabel_splits(df, stratify_cols=["outcome", "focal", "sex"]))
    else:
        splits = list(get_outcome_only_splits(df))

    # Analyse fold composition
    logger.info("Analysing fold composition...")
    fold_stats = analyse_fold_composition(df, splits, eeg_metadata)

    # Correlate with performance
    if fold_aucs is None:
        # Exp5c default AUC values
        fold_aucs = [0.600, 0.604, 0.604, 0.866, 0.545]

    logger.info("Computing correlations with AUC...")
    correlations = correlate_with_performance(fold_stats, fold_aucs)

    # Compare extreme folds
    logger.info("Comparing best vs worst folds...")
    extreme_comparison = compare_extreme_folds(df, splits, best_fold_idx=3, worst_fold_idx=4)

    return {
        "fold_stats": fold_stats,
        "correlations": correlations,
        "extreme_comparison": extreme_comparison,
        "stratification": "multilabel" if use_multilabel else "outcome_only",
    }


def print_report(results: Dict):
    """Print a formatted analysis report."""
    print("\n" + "=" * 70)
    print("FOLD COMPOSITION ANALYSIS REPORT")
    print(f"Stratification: {results['stratification']}")
    print("=" * 70)

    print("\n--- Per-Fold Statistics ---")
    fold_stats = results["fold_stats"]
    display_cols = ["fold", "n_val", "outcome_positive_pct", "focal_positive_pct", "sex_positive_pct"]
    if "eeg_duration_mean" in fold_stats.columns:
        display_cols.extend(["eeg_duration_mean", "eeg_padding_ratio_mean"])
    if "auc" in fold_stats.columns:
        display_cols.append("auc")
    print(fold_stats[display_cols].to_string(index=False))

    print("\n--- Feature Correlations with AUC ---")
    corr = results["correlations"]
    print(corr.head(10).to_string(index=False))

    print("\n--- Best vs Worst Fold Comparison ---")
    comp = results["extreme_comparison"]
    print(f"Best fold: {comp['best_fold']} (n={comp['best_n']})")
    print(f"Worst fold: {comp['worst_fold']} (n={comp['worst_n']})")
    print("\nFeature differences (best - worst):")
    for feat, stats in comp["features"].items():
        diff = stats["difference"]
        if abs(diff) > 5:  # Only show notable differences
            if "best_fold_pct" in stats:
                print(f"  {feat}: {stats['best_fold_pct']}% vs {stats['worst_fold_pct']}% (diff: {diff:+.1f}%)")
            elif "best_fold_mean" in stats:
                print(f"  {feat}: {stats['best_fold_mean']} vs {stats['worst_fold_mean']} (diff: {diff:+.1f})")


if __name__ == "__main__":
    # Run with outcome-only stratification (current Exp5c setting)
    print("\n" + "#" * 70)
    print("# OUTCOME-ONLY STRATIFICATION (Current Exp5c)")
    print("#" * 70)
    results_baseline = run_analysis(use_multilabel=False)
    print_report(results_baseline)

    # Run with multi-label stratification (proposed improvement)
    print("\n" + "#" * 70)
    print("# MULTI-LABEL STRATIFICATION (Proposed)")
    print("#" * 70)
    results_multilabel = run_analysis(use_multilabel=True)
    print_report(results_multilabel)
