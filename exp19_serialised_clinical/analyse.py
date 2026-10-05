"""Metrics, the primary contrast and descriptive comparisons for Experiment 19 (Addendum A / A.1).

Reads outputs/exp19_predictions/ (and the embedding stores for the cohort
probe) and writes, next to the predictions:

Every file is written per selection protocol: exp19_<name>_rf5.csv for the
refit protocol (primary, analysis plan B.2), exp19_<name>.csv for the
inner-split protocol (pre-registered); --protocol chooses.

    exp19_per_seed.csv   per (tag, estimator, seed): Melbourne pooled out-of-fold AUC (DeLong CI), fold-mean
                         AUC, HEP1 external AUC overall / complete-case / seen-drug
    exp19_summary.csv    means over seeds, plus HEP1 AUC of the seed-averaged ensemble with a bootstrap CI.
                         Its internal ci_lo/ci_hi are means of per-seed DeLong intervals on pooled
                         out-of-fold predictions; the paper's internal CIs come from
                         thesisStandalone/analysis/output/metrics_oof.csv (Addendum B.11).
    exp19_contrasts.csv  the primary contrast (B vs T5a-full, MLP; three encoders; internal
                         Nadeau-Bengio two one-sided tests against the +/-0.05 margin with Holm, i.e.
                         the 90% CI, and the Holm-adjusted two-sided test for superiority (plan B.7,
                         B.9); external paired
                         bootstrap) and the descriptive contrasts
    exp19_zeroshot_fp32.csv   zero-shot AUCs (float32 answer logits, plan B.7)
    exp19_probe.csv      fold-internal cohort probe (Melbourne vs HEP1) per representation

    python -m exp19_serialised_clinical.analyse
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shared.serialise_clinical import fit_fills
from shared.stats_util import delong_ci, delong_test
from exp18_mixed_cohort.analyse import holm

from .config import ENCODERS, MARGIN, POOLINGS, PRED_DIR, TEXT_COHORT_CONFIGS
from .texts import complete_case_mask, load_frames, seen_drug_mask, texts

OOF_RE = re.compile(r"predictions_oof_exp19_(?P<tag>.+)_(?P<est>mlp|pca32|lr)_sp-multilabel_(?P<proto>iv20|rf\d+)"
                    r"_s(?P<seed>\d+)\.json$")
N_BOOT = 2000
PRIMARY_ENCODERS = ("pubmedbert", "clinicalbert", "llama31_8b")


def _auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan


def load(pred_dir: Path, protocol: str = "rf5"):
    """Out-of-fold payloads and HEP1 ensembles of one selection protocol
    ("iv20" inner split, "rf5" refit)."""
    oof, ext = {}, {}
    hep_tag = "" if protocol == "iv20" else f"_{protocol}"
    for f in sorted(pred_dir.glob("predictions_oof_exp19_*.json")):
        m = OOF_RE.search(f.name)
        if not m or m["proto"] != protocol:
            continue
        key = (m["tag"], m["est"], int(m["seed"]))
        oof[key] = json.loads(f.read_text())
        hep = pred_dir / f"hep_external_exp19_{m['tag']}_{m['est']}{hep_tag}_s{m['seed']}.csv"
        if hep.exists():
            ext[key] = pd.read_csv(hep, dtype={"pid": str})
    return oof, ext


def pooled(payload: dict) -> pd.DataFrame:
    rows = [(str(p), f["fold"], int(y), float(q))
            for f in payload["folds"] for p, y, q in zip(f["pids"], f["y_true"], f["y_prob"])]
    return pd.DataFrame(rows, columns=["pid", "fold", "y_true", "y_prob"]).sort_values("pid").reset_index(drop=True)


def subgroup_masks(frames) -> dict[str, dict[str, np.ndarray]]:
    """{scope: {subgroup: pid -> bool}} for HEP1."""
    out = {}
    mel = frames[("MEL", "full")]
    for scope in ("full", "text"):
        hep = frames[("HEP", scope)]
        pids = hep["pid"].astype(str)
        out[scope] = {"complete_case": dict(zip(pids, complete_case_mask(hep))),
                      "seen_drug": dict(zip(pids, seen_drug_mask(hep, mel)))}
    return out


def _subset(h: pd.DataFrame, masks, scope, group):
    if group == "all":
        return h
    flags = h["pid"].map(masks[scope][group])
    if flags.isna().any():
        raise KeyError(f"{int(flags.isna().sum())} HEP1 patients missing from the {group} mask")
    return h[flags.astype(bool)]


def scope_of(tag: str) -> str:
    return "text" if tag.split("_")[0] in TEXT_COHORT_CONFIGS else "full"


def per_seed(oof, ext, masks) -> pd.DataFrame:
    rows = []
    for (tag, est, seed), payload in oof.items():
        df = pooled(payload)
        auc, lo, hi, _, _ = delong_ci(df["y_true"].to_numpy(), df["y_prob"].to_numpy())
        row = {"tag": tag, "estimator": est, "seed": seed, "n": len(df), "auc": auc, "ci_lo": lo, "ci_hi": hi,
               "fold_mean_auc": np.nanmean([_auc(g["y_true"], g["y_prob"]) for _, g in df.groupby("fold")])}
        if (tag, est, seed) in ext:
            for group in ("all", "complete_case", "seen_drug"):
                h = _subset(ext[(tag, est, seed)], masks, scope_of(tag), group)
                row[f"hep_{group}_n"], row[f"hep_{group}_auc"] = len(h), _auc(h["y_true"], h["y_prob"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["tag", "estimator", "seed"])


def seed_averaged(ext, tag, est) -> pd.DataFrame | None:
    frames = [h for (t, e, _), h in ext.items() if t == tag and e == est]
    if not frames:
        return None
    df = pd.concat(frames).groupby(["pid", "y_true"], as_index=False)["y_prob"].mean()
    return df.sort_values("pid").reset_index(drop=True)


def bootstrap_auc(y, p, rng, n_boot=N_BOOT):
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            vals.append(roc_auc_score(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


def bootstrap_diff(y, pa, pb, rng, n_boot=N_BOOT):
    """Paired patient-level bootstrap of AUC(a) - AUC(b): (diff, lo, hi, two-sided p)."""
    y, pa, pb = map(np.asarray, (y, pa, pb))
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            diffs.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return _auc(y, pa) - _auc(y, pb), *np.percentile(diffs, [2.5, 97.5]), min(1.0, p)


def nb_ci(d: np.ndarray, ratio: float, alpha: float = 0.05):
    """Nadeau-Bengio corrected resampled t over seed x fold differences: (mean, lo, hi, p)."""
    d = d[~np.isnan(d)]
    j = len(d)
    se = np.sqrt((1 / j + ratio) * np.var(d, ddof=1))
    t_crit = stats.t.ppf(1 - alpha / 2, df=j - 1)
    t = d.mean() / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), df=j - 1) if np.isfinite(t) else np.nan
    return float(d.mean()), float(d.mean() - t_crit * se), float(d.mean() + t_crit * se), float(p)


def tost_p(d: np.ndarray, ratio: float, margin: float = MARGIN) -> float:
    """Two one-sided Nadeau-Bengio tests of |difference| < margin: the larger
    one-sided p (below alpha exactly when the 1 - 2 alpha CI lies inside the
    margin)."""
    d = d[~np.isnan(d)]
    j = len(d)
    se = np.sqrt((1 / j + ratio) * np.var(d, ddof=1))
    if not se > 0:
        return float("nan")
    p_low = stats.t.sf((d.mean() + margin) / se, df=j - 1)    # H0: diff <= -margin
    p_high = stats.t.cdf((d.mean() - margin) / se, df=j - 1)  # H0: diff >= +margin
    return float(max(p_low, p_high))


def verdict(diff: float, p_holm: float, p_tost_holm: float, margin: float = MARGIN) -> str:
    """Equivalence first (Holm-adjusted two one-sided tests, i.e. the 90% CI
    inside the margin), then superiority from the Holm-adjusted two-sided
    Nadeau-Bengio test (Addendum A.1 Holm family; plan B.7 and B.9)."""
    if p_tost_holm < 0.05:
        return f"equivalent within +/-{margin}"
    if p_holm < 0.05:
        return "text better" if diff > 0 else "tabular better"
    return "inconclusive"


def contrast_row(oof, ext, masks, label, kind, a, b, est_a="mlp", est_b="mlp", seed=42):
    seeds = sorted({s for (t, e, s) in oof if t == a and e == est_a} & {s for (t, e, s) in oof if t == b and e == est_b})
    if not seeds:
        return None
    fold_d, ratios, delong_p = [], [], []
    for s in seeds:
        da, db = pooled(oof[(a, est_a, s)]), pooled(oof[(b, est_b, s)])
        assert da["pid"].tolist() == db["pid"].tolist(), f"{a} and {b} scored different patients"
        delong_p.append(delong_test(da["y_true"].to_numpy(), da["y_prob"].to_numpy(), db["y_prob"].to_numpy())[4])
        for k in sorted(da["fold"].unique()):
            fa, fb = da[da["fold"] == k], db[db["fold"] == k]
            assert fa["pid"].tolist() == fb["pid"].tolist(), "folds differ between the paired tags"
            fold_d.append(_auc(fa["y_true"], fa["y_prob"]) - _auc(fb["y_true"], fb["y_prob"]))
            ratios.append(len(fa) / (len(da) - len(fa)))
    ratio = float(np.mean(ratios))
    mean, lo, hi, p = nb_ci(np.array(fold_d), ratio)
    _, lo90, hi90, _ = nb_ci(np.array(fold_d), ratio, alpha=0.10)
    row = {"kind": kind, "comparison": label, "a": f"{a}/{est_a}", "b": f"{b}/{est_b}", "n_seeds": len(seeds),
           "internal_diff": mean, "internal_ci_lo": lo, "internal_ci_hi": hi, "internal_nb_p": p,
           "internal_ci90_lo": lo90, "internal_ci90_hi": hi90,
           "internal_tost_p": tost_p(np.array(fold_d), ratio),
           "internal_delong_p_median": float(np.median(delong_p))}
    ea, eb = seed_averaged(ext, a, est_a), seed_averaged(ext, b, est_b)
    if ea is not None and eb is not None:
        assert ea["pid"].tolist() == eb["pid"].tolist()
        for group in ("all", "complete_case", "seen_drug"):
            ga, gb = _subset(ea, masks, scope_of(a), group), _subset(eb, masks, scope_of(b), group)
            diff, elo, ehi, ep = bootstrap_diff(ga["y_true"], ga["y_prob"], gb["y_prob"], np.random.default_rng(seed))
            row.update({f"hep_{group}_diff": diff, f"hep_{group}_ci_lo": elo, f"hep_{group}_ci_hi": ehi,
                        f"hep_{group}_p": ep})
    return row


def contrasts(oof, ext, masks) -> pd.DataFrame:
    rows = [contrast_row(oof, ext, masks, "text clinical vs information-matched tabular (B vs T5a-full)",
                         "primary", f"B_{enc}_mean", "T5a-full") for enc in PRIMARY_ENCODERS]
    rows = [r for r in rows if r]
    for row, p_adj, t_adj in zip(rows, holm([r["internal_nb_p"] for r in rows]),
                                 holm([r["internal_tost_p"] for r in rows])):
        row["internal_nb_p_holm"] = p_adj
        row["internal_tost_p_holm"] = t_adj
        row["verdict"] = verdict(row["internal_diff"], p_adj, t_adj)
    for enc, spec in ENCODERS.items():
        for pool in POOLINGS[spec["kind"]]:
            s = f"_{enc}_{pool}"
            for label, a, b in [("drug by name vs tabular + structure (A vs T5a)", f"A{s}", "T5a"),
                                ("imputed text vs paper tabular (B-imp vs T5a)", f"B-imp{s}", "T5a"),
                                ("text vs information-matched tabular, no drug (E vs T4-full)", f"E{s}", "T4-full"),
                                ("imputed text vs paper tabular, no drug (E-imp vs T4)", f"E-imp{s}", "T4"),
                                ("segment-balanced vs token mean (D vs D-tok)", f"D{s}", f"D-tok{s}"),
                                ("one document vs separate branches (D vs D-split)", f"D{s}", f"D-split{s}"),
                                ("split text vs tabular with report (D-split vs T6a)", f"D-split{s}", "T6a")]:
                rows.append(contrast_row(oof, ext, masks, label, "descriptive", a, b))
            for est in ("pca32", "lr"):
                rows.append(contrast_row(oof, ext, masks, f"B vs T5a-full with {est}", "descriptive",
                                         f"B{s}", "T5a-full", est, est))
        if spec["kind"] == "decoder":
            for cfg in ("A", "B", "B-imp", "E", "E-imp", "D", "D-tok", "D-split"):
                rows.append(contrast_row(oof, ext, masks, f"mean vs last-token pooling ({cfg})", "descriptive",
                                         f"{cfg}_{enc}_mean", f"{cfg}_{enc}_last"))
    return pd.DataFrame([r for r in rows if r])


def zero_shot(pred_dir: Path, masks) -> pd.DataFrame:
    rows = []
    for cohort in ("MEL", "HEP"):
        f = pred_dir / f"zeroshot_fp32_{cohort}.csv"
        if not f.exists():
            continue
        z = pd.read_csv(f, dtype={"pid": str})
        groups = ("all", "complete_case", "seen_drug") if cohort == "HEP" else ("all",)
        for group in groups:
            g = _subset(z.rename(columns={"score": "y_prob"}), masks, "full", group)
            auc, lo, hi, _, _ = delong_ci(g["y_true"].to_numpy(), g["y_prob"].to_numpy())
            rows.append({"cohort": cohort, "subgroup": group, "n": len(g), "auc": auc, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def cohort_probe(frames) -> pd.DataFrame:
    """How well each representation identifies the cohort (5-fold CV logistic regression), descriptive."""
    from shared.portable_models import refit_clinical
    from .run_experiments import lookup
    from .tabular import FullClinicalPreprocessor
    mel, hep = frames[("MEL", "full")], frames[("HEP", "full")]
    both = pd.concat([mel, hep], ignore_index=True)
    y = np.r_[np.zeros(len(mel)), np.ones(len(hep))]
    reps = {"tabular (paper)": refit_clinical(both, both, np.arange(len(both)))[0].numpy(),
            "tabular (information-matched)": FullClinicalPreprocessor().fit(both).transform(both)}
    fills = fit_fills(mel)
    for enc, spec in ENCODERS.items():
        for pool in POOLINGS[spec["kind"]]:
            for variant in ("v1nodrug", "v1nodrugimp"):
                try:
                    f = fills if variant == "v1nodrugimp" else None
                    reps[f"{enc}_{pool} {variant}"] = np.vstack([
                        lookup(enc, pool, texts(mel, variant, f)).numpy(), lookup(enc, pool, texts(hep, variant, f)).numpy()])
                except (KeyError, FileNotFoundError):
                    continue
    rows = []
    for name, x in reps.items():
        aucs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(x, y):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(x[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(x[te])[:, 1]))
        rows.append({"representation": name, "cohort_probe_auc": float(np.mean(aucs))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred-dir", type=Path, default=PRED_DIR)
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--protocol", default="rf5", help='"rf5" refit (primary) or "iv20" inner split')
    args = parser.parse_args()
    frames = load_frames()
    masks = subgroup_masks(frames)
    oof, ext = load(args.pred_dir, args.protocol)
    suffix = "" if args.protocol == "iv20" else f"_{args.protocol}"
    if not oof:
        raise SystemExit(f"no exp19 prediction files in {args.pred_dir}")
    seeds = per_seed(oof, ext, masks)
    summary = seeds.groupby(["tag", "estimator"]).agg(
        n=("n", "first"), n_seeds=("seed", "nunique"), auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
        ci_lo_mean=("ci_lo", "mean"), ci_hi_mean=("ci_hi", "mean"), fold_mean_auc=("fold_mean_auc", "mean"),
        hep_auc_mean=("hep_all_auc", "mean"), hep_auc_sd=("hep_all_auc", "std")).reset_index()
    ens = []
    for tag, est in summary[["tag", "estimator"]].itertuples(index=False):
        e = seed_averaged(ext, tag, est)
        if e is not None:
            lo, hi = bootstrap_auc(e["y_true"], e["y_prob"], np.random.default_rng(42))
            ens.append({"tag": tag, "estimator": est, "hep_ensemble_auc": _auc(e["y_true"], e["y_prob"]),
                        "hep_ensemble_ci_lo": lo, "hep_ensemble_ci_hi": hi})
    summary = summary.merge(pd.DataFrame(ens), on=["tag", "estimator"], how="left")
    comps = contrasts(oof, ext, masks)
    zs = zero_shot(args.pred_dir, masks)
    seeds.to_csv(args.pred_dir / f"exp19_per_seed{suffix}.csv", index=False)
    summary.to_csv(args.pred_dir / f"exp19_summary{suffix}.csv", index=False)
    comps.to_csv(args.pred_dir / f"exp19_contrasts{suffix}.csv", index=False)
    zs.to_csv(args.pred_dir / "exp19_zeroshot_fp32.csv", index=False)
    if not args.no_probe:
        cohort_probe(frames).to_csv(args.pred_dir / "exp19_probe.csv", index=False)
    print(summary[summary["estimator"] == "mlp"][["tag", "n", "auc_mean", "auc_sd", "hep_ensemble_auc"]]
          .round(3).to_string(index=False))
    prim = comps[comps["kind"] == "primary"] if len(comps) else comps
    if len(prim):
        print("\nPrimary contrast (B vs T5a-full, MLP):")
        print(prim[["a", "internal_diff", "internal_ci90_lo", "internal_ci90_hi", "internal_tost_p_holm", "verdict",
                    "hep_all_diff", "hep_all_ci_lo", "hep_all_ci_hi"]].round(3).to_string(index=False))
    print(f"\nwrote exp19_per_seed/summary/contrasts{suffix}, zeroshot_fp32{'' if args.no_probe else ', probe'}"
          f" to {args.pred_dir}")


if __name__ == "__main__":
    main()
