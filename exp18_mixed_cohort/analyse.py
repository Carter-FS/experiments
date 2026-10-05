"""Metrics and hypothesis tests for Experiment 18, as fixed in the analysis plan.

Reads outputs/exp18_mixed_cohort/predictions_*.csv and folds_*.csv and writes:

    per_seed.csv   one row per (config, variant, seed, arm, metric, cohort)
    summary.csv    the same metrics averaged over seeds (mean, SD, min, max)
    tests.csv      primary Nadeau-Bengio tests (Holm) and exploratory DeLong contrasts

Metrics (docs/analysis_plan_clean_rerun_exp18.md, section 6):
  - within-cohort out-of-fold AUC with a DeLong 95% CI, per seed;
  - cohort-stratified AUC (primary overall): only same-cohort positive/negative
    pairs, i.e. the pair-weighted mean of the within-cohort AUCs;
  - whole-fold AUC (secondary, requested by the supervisor): per fold over all
    test patients, next to the cohort-only floor (score = the training fold's
    seizure-free rate of the patient's cohort);
  - calibration per cohort: Brier, calibration-in-the-large (mean predicted
    minus observed, not the logistic-offset intercept) and calibration slope.

Variants: "_rf<k>" refit protocol (primary from analysis plan Addendum B.2;
no prefix = the pre-registered inner-split protocol), "_h12" harmonised HEP1
label (B.5), "_noRMH" and "_dedup" as in section 6. Primary tests use the
refit runs on the provided HEP1 label (deduplicated when that run exists);
the same tests on the inner-split runs are reported as "preregistered" and on
the harmonised label as "sensitivity". The Nadeau-Bengio variance inflation
uses n_test / n_train of the test cohort's own share of each outer fold, the
training set of the own-cohort arm (B.6).

    python -m exp18_mixed_cohort.analyse
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from shared.stats_util import delong_ci, delong_test

from .config import CONFIGS, OUT_DIR

COHORTS = ("MEL", "HEP")
OWN = {"MEL": "mel_only", "HEP": "hep_only"}
OTHER = {"MEL": "hep_only", "HEP": "mel_only"}
FILE_RE = re.compile(r"predictions_(" + "|".join(re.escape(c) for c in sorted(CONFIGS, key=len, reverse=True))
                     + r")((?:_rf\d+)?(?:_h12)?(?:_noRMH)?(?:_dedup)?)_seed(\d+)\.csv$")


def auc_or_nan(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan


def stratified_auc(df: pd.DataFrame) -> float:
    """P(seizure-free outranks non-seizure-free | same cohort)."""
    from shared.stats_util import cohort_stratified_auc
    return cohort_stratified_auc(df["y_true"], df["y_prob"], df["cohort"])


def calibration(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    slope = np.nan
    if len(np.unique(y)) > 1:
        slope = float(LogisticRegression(C=1e6, max_iter=1000).fit(logit, y).coef_[0, 0])
    return {"brier": float(np.mean((p - y) ** 2)), "citl": float(p.mean() - y.mean()), "slope": slope}


def load_runs(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds, folds = [], []
    for f in sorted(out_dir.glob("predictions_*.csv")):
        m = FILE_RE.search(f.name)
        if not m:  # smoke runs and anything unexpected are ignored
            continue
        variant = m.group(2) or ""
        preds.append(pd.read_csv(f, dtype={"pid": str}).assign(variant=variant))
        folds.append(pd.read_csv(f.with_name(f.name.replace("predictions_", "folds_"))).assign(variant=variant))
    if not preds:
        raise SystemExit(f"no exp18 prediction files in {out_dir}")
    return pd.concat(preds, ignore_index=True), pd.concat(folds, ignore_index=True)


def per_seed_metrics(preds: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    main = preds[preds["draw"] == -1]
    for (cfg, var, seed, arm), g in main.groupby(["config", "variant", "seed", "arm"]):
        base = {"config": cfg, "variant": var, "seed": seed, "arm": arm}
        for c in COHORTS:
            gc = g[g["cohort"] == c]
            auc, lo, hi, _, _ = delong_ci(gc["y_true"].to_numpy(), gc["y_prob"].to_numpy())
            rows += [{**base, "metric": "auc", "cohort": c, "value": auc, "ci_lo": lo, "ci_hi": hi,
                      "n": len(gc)}]
            rows += [{**base, "metric": k, "cohort": c, "value": v, "n": len(gc)}
                     for k, v in calibration(gc["y_true"].to_numpy(), gc["y_prob"].to_numpy()).items()]
        rows.append({**base, "metric": "stratified_auc", "cohort": "both", "value": stratified_auc(g),
                     "n": len(g)})
        fold_aucs = [auc_or_nan(f["y_true"], f["y_prob"]) for _, f in g.groupby("fold")]
        rows.append({**base, "metric": "whole_fold_auc", "cohort": "both", "value": np.nanmean(fold_aucs),
                     "sd": np.nanstd(fold_aucs, ddof=1), "n": len(g)})
    # Cohort-only floor: one row per (config, variant, seed), reported alongside the whole-fold AUC.
    for (cfg, var, seed), fd in folds.groupby(["config", "variant", "seed"]):
        g = main[(main["config"] == cfg) & (main["variant"] == var) & (main["seed"] == seed)
                 & (main["arm"] == "mixed")]
        rate = fd.set_index("fold")
        floor = [auc_or_nan(f["y_true"], f["cohort"].map(lambda c, k=k: rate.at[k, f"train_rate_{c}"]))
                 for k, f in g.groupby("fold")]
        rows.append({"config": cfg, "variant": var, "seed": seed, "arm": "cohort_only_floor",
                     "metric": "whole_fold_auc", "cohort": "both", "value": np.nanmean(floor),
                     "sd": np.nanstd(floor, ddof=1), "n": len(g)})
    # Size-matched mixed training: per draw, pooled out-of-fold AUC on its test cohort.
    sm = preds[preds["arm"].str.startswith("sizematched_")]
    for (cfg, var, seed, arm, draw), g in sm.groupby(["config", "variant", "seed", "arm", "draw"]):
        rows.append({"config": cfg, "variant": var, "seed": seed, "arm": arm, "draw": draw,
                     "metric": "auc", "cohort": arm.split("_")[1],
                     "value": auc_or_nan(g["y_true"], g["y_prob"]), "n": len(g)})
    return pd.DataFrame(rows)


def summarise(per_seed: pd.DataFrame) -> pd.DataFrame:
    keys = ["config", "variant", "arm", "metric", "cohort"]
    return (per_seed.groupby(keys, dropna=False)["value"]
            .agg(mean="mean", sd="std", min="min", max="max", n_values="count").reset_index())


def nadeau_bengio(d: np.ndarray, test_train_ratio: float) -> tuple[float, float, float]:
    """Corrected resampled t-test on per-(seed, fold) differences: (mean, t, two-sided p)."""
    d = d[~np.isnan(d)]
    j = len(d)
    var = np.var(d, ddof=1)
    t = d.mean() / np.sqrt((1 / j + test_train_ratio) * var) if var > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), df=j - 1) if np.isfinite(t) else np.nan
    return float(d.mean()), float(t), float(p)


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def test_variants(preds: pd.DataFrame) -> list[tuple[str, str]]:
    """(kind, variant) pairs for the Exp4a mixed-vs-own tests: the refit runs
    (primary), the inner-split runs (preregistered) and the harmonised-label
    refit runs (sensitivity), each on the deduplicated cohort when that run
    exists."""
    have = set(preds.loc[preds["config"] == "Exp4a", "variant"])
    # The refit protocol of Addendum B.2 is _rf5; another fold count would be a
    # different analysis, never the primary one.
    refit = ["_rf5"] if any(v.startswith("_rf5") for v in have) else []
    out = []
    for kind, base in [("primary", refit[0] if refit else None), ("preregistered", ""),
                       ("sensitivity", (refit[0] + "_h12") if refit else None)]:
        if base is None:
            continue
        var = base + "_dedup" if base + "_dedup" in have else base
        if var in have:
            out.append((kind, var))
    return out


def primary_tests(preds: pd.DataFrame, folds: pd.DataFrame) -> list[dict]:
    """Exp4a: mixed vs own-cohort training, per test cohort (Holm over the two
    cohorts within each kind)."""
    rows = []
    for kind, var in test_variants(preds):
        g = preds[(preds["config"] == "Exp4a") & (preds["variant"] == var) & (preds["draw"] == -1)]
        fd = folds[(folds["config"] == "Exp4a") & (folds["variant"] == var)]
        kind_rows = []
        for c in COHORTS:
            ratio = float((fd[f"n_test_{c}"] / fd[f"n_train_{c}"]).mean())
            gc = g[g["cohort"] == c]
            per = pd.DataFrame(
                [{"seed": s, "fold": k, "arm": a, "auc": auc_or_nan(f["y_true"], f["y_prob"])}
                 for (s, k, a), f in gc.groupby(["seed", "fold", "arm"])]
            ).pivot_table(index=["seed", "fold"], columns="arm", values="auc", dropna=False)
            d = (per["mixed"] - per[OWN[c]]).to_numpy()
            mean_d, t, p = nadeau_bengio(d, ratio)
            kind_rows.append({"kind": kind, "config": "Exp4a", "variant": var, "cohort": c,
                              "contrast": f"mixed - {OWN[c]}", "estimate": mean_d, "stat": t, "p": p,
                              "nb_ratio": ratio, "n_resamples": int(np.isfinite(d).sum())})
        for row, p_adj in zip(kind_rows, holm([r["p"] for r in kind_rows])):
            row["p_holm"] = p_adj
        rows += kind_rows
    return rows


def exploratory_tests(preds: pd.DataFrame) -> list[dict]:
    """Paired DeLong per seed on pooled out-of-fold predictions (no significance claims)."""
    rows = []
    main = preds[preds["draw"] == -1]
    for (cfg, var, seed), g in main.groupby(["config", "variant", "seed"]):
        wide = g.pivot_table(index=["pid", "cohort", "y_true"], columns="arm", values="y_prob").reset_index()
        for c in COHORTS:
            w = wide[wide["cohort"] == c]
            for a, b in (("mixed", OWN[c]), ("mixed", OTHER[c]), (OWN[c], OTHER[c])):
                if {a, b} <= set(w.columns):
                    auc_a, auc_b, diff, z, p = delong_test(w["y_true"].to_numpy(), w[a].to_numpy(), w[b].to_numpy())
                    rows.append({"kind": "exploratory", "config": cfg, "variant": var, "seed": seed,
                                 "cohort": c, "contrast": f"{a} - {b}", "estimate": diff,
                                 "stat": z, "p": p, "auc_a": auc_a, "auc_b": auc_b})
    # Size-matched draws score only their own test cohort's patients, the same
    # patients as the own-cohort and mixed arms, so they pair per draw.
    sm = preds[preds["arm"].str.startswith("sizematched_")]
    for (cfg, var, seed, arm, draw), g in sm.groupby(["config", "variant", "seed", "arm", "draw"]):
        c = arm.split("_")[1]
        ref = main[(main["config"] == cfg) & (main["variant"] == var) & (main["seed"] == seed)
                   & (main["cohort"] == c)].pivot_table(index=["pid", "y_true"], columns="arm", values="y_prob")
        w = g.set_index("pid")["y_prob"].rename(arm).to_frame().join(ref.reset_index("y_true"), how="inner")
        for b in (OWN[c], "mixed"):
            if b in w.columns and len(w):
                auc_a, auc_b, diff, z, p = delong_test(w["y_true"].to_numpy(), w[arm].to_numpy(), w[b].to_numpy())
                rows.append({"kind": "exploratory", "config": cfg, "variant": var, "seed": seed, "draw": draw,
                             "cohort": c, "contrast": f"{arm} - {b}", "estimate": diff,
                             "stat": z, "p": p, "auc_a": auc_a, "auc_b": auc_b})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    preds, folds = load_runs(args.out_dir)
    per_seed = per_seed_metrics(preds, folds)
    summary = summarise(per_seed)
    tests = pd.DataFrame(primary_tests(preds, folds) + exploratory_tests(preds))
    per_seed.to_csv(args.out_dir / "per_seed.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    tests.to_csv(args.out_dir / "tests.csv", index=False)

    view = summary[summary["metric"].isin(["auc", "stratified_auc", "whole_fold_auc"])]
    print(view.pivot_table(index=["config", "variant", "arm"], columns=["metric", "cohort"],
                           values="mean").round(3).to_string())
    confirmatory = tests[tests["kind"] != "exploratory"] if not tests.empty else tests
    if not confirmatory.empty:
        print("\nExp4a mixed vs own-cohort (Nadeau-Bengio, Holm over 2 within each kind):")
        print(confirmatory[["kind", "variant", "cohort", "contrast", "estimate", "stat", "p", "p_holm"]]
              .round(4).to_string(index=False))
    print(f"\nwrote per_seed.csv, summary.csv, tests.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
