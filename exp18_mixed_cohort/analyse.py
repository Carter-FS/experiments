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

Primary tests use the deduplicated cohort (variant "_dedup") when that run
exists, otherwise the unmodified one. The Nadeau-Bengio variance inflation
uses n_test / n_train of the outer folds (about 0.25); each arm's model is
fitted on 80% of its share of the outer training fold.

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
                     + r")(_noRMH)?(_dedup)?_seed(\d+)\.csv$")


def auc_or_nan(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan


def stratified_auc(df: pd.DataFrame) -> float:
    """P(seizure-free outranks non-seizure-free | same cohort)."""
    num = den = 0.0
    for _, g in df.groupby("cohort"):
        n_pos, n_neg = int(g["y_true"].sum()), int((1 - g["y_true"]).sum())
        if n_pos and n_neg:
            num += auc_or_nan(g["y_true"], g["y_prob"]) * n_pos * n_neg
            den += n_pos * n_neg
    return num / den if den else np.nan


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
        variant = (m.group(2) or "") + (m.group(3) or "")
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


def primary_variant(preds: pd.DataFrame) -> str:
    """The deduplicated cohort once confirmed duplicates were excluded, else the unmodified one."""
    exp4a = preds.loc[preds["config"] == "Exp4a", "variant"]
    return "_dedup" if (exp4a == "_dedup").any() else ""


def primary_tests(preds: pd.DataFrame, folds: pd.DataFrame) -> list[dict]:
    """Exp4a: mixed vs own-cohort training, per test cohort."""
    var = primary_variant(preds)
    g = preds[(preds["config"] == "Exp4a") & (preds["variant"] == var) & (preds["draw"] == -1)]
    if g.empty:
        return []
    fd = folds[(folds["config"] == "Exp4a") & (folds["variant"] == var)]
    ratio = float((fd["n_test"] / fd["n_train"]).mean())
    rows = []
    for c in COHORTS:
        gc = g[g["cohort"] == c]
        per = pd.DataFrame(
            [{"seed": s, "fold": k, "arm": a, "auc": auc_or_nan(f["y_true"], f["y_prob"])}
             for (s, k, a), f in gc.groupby(["seed", "fold", "arm"])]
        ).pivot_table(index=["seed", "fold"], columns="arm", values="auc", dropna=False)
        d = (per["mixed"] - per[OWN[c]]).to_numpy()
        mean_d, t, p = nadeau_bengio(d, ratio)
        rows.append({"kind": "primary", "config": "Exp4a", "variant": var, "cohort": c,
                     "contrast": f"mixed - {OWN[c]}", "estimate": mean_d, "stat": t, "p": p,
                     "n_resamples": int(np.isfinite(d).sum())})
    for row, p_adj in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = p_adj
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
    if not tests.empty and (tests["kind"] == "primary").any():
        print("\nPrimary tests (Exp4a, Nadeau-Bengio, Holm over 2):")
        print(tests[tests["kind"] == "primary"][["cohort", "contrast", "estimate", "stat", "p", "p_holm"]]
              .round(4).to_string(index=False))
    print(f"\nwrote per_seed.csv, summary.csv, tests.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
