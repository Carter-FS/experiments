"""Data cleaning utilities for clinical features.

Handles mixed types and missing values in the clinical dataset.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import CSV_PATH, OUTCOME_MAPPING

logger = logging.getLogger(__name__)


def clean_binary_column(series: pd.Series, col_name: str) -> pd.Series:
    """Clean a binary column with mixed types.

    Handles: 0, 1, 0.0, 1.0, '0', '1', '0.0', '1.0', '?', NaN

    Args:
        series: Column to clean.
        col_name: Column name for logging.

    Returns:
        Cleaned series with int dtype (0 or 1), NaN for invalid.
    """
    original_dtype = series.dtype
    n_original = len(series)

    # Convert to string for uniform handling
    str_series = series.astype(str).str.strip().str.lower()

    # Map values
    mapping = {
        '0': 0, '0.0': 0, 'false': 0, 'no': 0, 'n': 0,
        '1': 1, '1.0': 1, 'true': 1, 'yes': 1, 'y': 1,
        '?': np.nan, 'nan': np.nan, 'none': np.nan, '': np.nan,
        'not available': np.nan, 'na': np.nan,
    }

    cleaned = str_series.map(mapping)

    # Log cleaning results
    n_invalid = cleaned.isna().sum() - series.isna().sum()
    if n_invalid > 0:
        invalid_values = series[cleaned.isna() & series.notna()].unique()
        logger.warning(
            f"{col_name}: {n_invalid} invalid values converted to NaN: {invalid_values}"
        )

    return cleaned.astype('Int64')  # Nullable integer


def clean_categorical_column(series: pd.Series, col_name: str) -> pd.Series:
    """Clean a categorical column with mixed types.

    Args:
        series: Column to clean.
        col_name: Column name for logging.

    Returns:
        Cleaned series with consistent integer categories.
    """
    str_series = series.astype(str).str.strip().str.lower()

    # Handle special values
    invalid_markers = ['?', 'nan', 'none', '', 'not available', 'na']
    cleaned = str_series.replace(invalid_markers, np.nan)

    # Convert numeric strings to integers
    def to_int(x):
        if pd.isna(x):
            return np.nan
        try:
            return int(float(x))
        except (ValueError, TypeError):
            return np.nan

    cleaned = cleaned.apply(to_int)

    n_invalid = cleaned.isna().sum() - series.isna().sum()
    if n_invalid > 0:
        logger.warning(f"{col_name}: {n_invalid} invalid values converted to NaN")

    return cleaned.astype('Int64')


def load_and_clean_data(filepath: str = None) -> Tuple[pd.DataFrame, Dict[str, List]]:
    """Load and clean the clinical dataset.

    Args:
        filepath: Path to CSV file. Defaults to config CSV_PATH.

    Returns:
        Tuple of (cleaned_df, cleaning_report).
    """
    filepath = filepath or CSV_PATH
    logger.info(f"Loading data from {filepath}")

    df = pd.read_csv(filepath)
    original_shape = df.shape
    logger.info(f"Loaded {original_shape[0]} rows, {original_shape[1]} columns")

    cleaning_report = {
        "original_rows": original_shape[0],
        "columns_cleaned": [],
        "rows_dropped": 0,
        "warnings": [],
    }

    # Binary columns to clean
    binary_cols = [
        "sex", "pretrt_sz_5", "focal", "fam_hx", "febrile", "ci",
        "birth_t", "head", "drug", "alcohol", "cvd", "psy", "ld"
    ]

    for col in binary_cols:
        if col in df.columns:
            df[col] = clean_binary_column(df[col], col)
            cleaning_report["columns_cleaned"].append(col)

    # Categorical columns to clean
    categorical_cols = ["lesion", "eeg_cat"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = clean_categorical_column(df[col], col)
            cleaning_report["columns_cleaned"].append(col)

    # Clean outcome column
    if "outcome" in df.columns:
        # Convert to numeric first (handles string '1', '2')
        df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")

        # Filter to valid outcomes (raw 1 = success, 2 = failure)
        valid_outcomes = df["outcome"].isin([1, 2, 1.0, 2.0])
        n_invalid_outcome = (~valid_outcomes).sum()
        if n_invalid_outcome > 0:
            logger.warning(f"Dropping {n_invalid_outcome} rows with invalid outcome")
            cleaning_report["warnings"].append(
                f"Dropped {n_invalid_outcome} rows with invalid outcome"
            )
            df = df[valid_outcomes].copy()

        # Map to 0/1
        df["outcome"] = df["outcome"].astype(int).map(OUTCOME_MAPPING)

    cleaning_report["rows_dropped"] = original_shape[0] - len(df)
    cleaning_report["final_rows"] = len(df)

    logger.info(f"Cleaning complete: {len(df)} rows retained")

    return df, cleaning_report


def generate_cleaning_report(report: Dict) -> str:
    """Generate a human-readable cleaning report.

    Args:
        report: Cleaning report dictionary.

    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 50,
        "DATA CLEANING REPORT",
        "=" * 50,
        f"Original rows: {report['original_rows']}",
        f"Final rows: {report['final_rows']}",
        f"Rows dropped: {report['rows_dropped']}",
        "",
        f"Columns cleaned: {', '.join(report['columns_cleaned'])}",
        "",
    ]

    if report["warnings"]:
        lines.append("Warnings:")
        for warning in report["warnings"]:
            lines.append(f"  - {warning}")

    lines.append("=" * 50)

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    df, report = load_and_clean_data()
    print(generate_cleaning_report(report))

    print("\nSample of cleaned data:")
    print(df[["outcome", "focal", "sex", "psy", "lesion"]].head(10))  # no patient ids on the console
