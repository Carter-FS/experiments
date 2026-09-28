"""Outcome-polarity diagnostic (analysis plan Addendum B.1).

For each cohort, prints the seizure-free rate (label 1) with and without each
established predictor of drug resistance. Every one should LOWER the rate. A
cohort where most of them raise it has an inverted label. Aggregate counts
only; the table is also written to outputs/polarity_check.csv (gitignored).

    python -m shared.polarity_check
"""
from __future__ import annotations

import pandas as pd

from shared.hep_cohort import EXPERIMENTS_ROOT, load_alfred, load_hep

# Predictor -> (column, rule giving the "present" mask). Each is associated
# with a lower chance of seizure freedom on the first ASM in the literature.
PREDICTORS = {
    ">5 pre-treatment seizures": ("pretrt_sz_5", lambda s: s == 1),
    "psychiatric history": ("psy", lambda s: s == 1),
    "epileptiform EEG": ("eeg_cat", lambda s: s == 3),
    "abnormal imaging": ("lesion", lambda s: s > 1),
    "head trauma": ("head", lambda s: s == 1),
    "learning disability": ("ld", lambda s: s == 1),
}


def table(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    rows = []
    y = df["outcome"]
    for name, (col, rule) in PREDICTORS.items():
        s = pd.to_numeric(df[col], errors="coerce")
        known = s.notna()
        present = rule(s) & known
        absent = ~rule(s) & known
        if present.sum() == 0 or absent.sum() == 0:
            continue
        rows.append({
            "cohort": cohort, "predictor": name,
            "n_present": int(present.sum()), "sf_rate_present": float(y[present].mean()),
            "n_absent": int(absent.sum()), "sf_rate_absent": float(y[absent].mean()),
        })
    out = pd.DataFrame(rows)
    out["expected_direction"] = out["sf_rate_present"] < out["sf_rate_absent"]
    return out


def main() -> int:
    res = pd.concat([table(load_alfred(), "Melbourne"), table(load_hep(), "HEP1")], ignore_index=True)
    with pd.option_context("display.width", 140, "display.float_format", "{:.2f}".format):
        print(res.to_string(index=False))
    for cohort, g in res.groupby("cohort"):
        print(f"{cohort}: {int(g['expected_direction'].sum())}/{len(g)} predictors in the expected direction")
    out = EXPERIMENTS_ROOT / "outputs" / "polarity_check.csv"
    res.to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
