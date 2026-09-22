"""Run Experiment 19: serialised-text vs tabular representations (Addendum A / A.1).

For each configuration, encoder, pooling, estimator and seed: the clean CV
protocol on the Melbourne cohort (multilabel outer folds, inner 20% early
stopping, C selection and threshold), with every fold's model also applied to
HEP1 (five-fold ensemble). Per fold, imputed texts use that fold's fills,
every frozen embedding input is z-scored on the fit rows, and PCA and the
clinical preprocessors are fitted on the fit rows. Configurations on the same
cohort scope share folds, so serialised and tabular configurations pair fold
by fold.

    python -m exp19_serialised_clinical.run_experiments                  # everything
    python -m exp19_serialised_clinical.run_experiments --configs B T5a-full --encoders pubmedbert --seeds 42

Outputs (outputs/exp19_predictions/, patient-level, gitignored):
    predictions_oof_exp19_<tag>_<est>_sp-multilabel_iv20_s<seed>.json   Melbourne out-of-fold
    hep_external_exp19_<tag>_<est>_s<seed>.csv                           HEP1 five-fold ensemble
<tag> = configuration, plus _<encoder>_<pooling> for serialised ones.
Existing outputs are skipped unless --force.
"""

from __future__ import annotations

import argparse
import logging
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import shared.portable_models as portable
from shared.cv_splits import cv_suffix, youden_threshold
from shared.determinism import enable_determinism
from shared.hep_cohort import build_smiles_feature_matrix, build_smiles_lookup
from shared.prediction_logger import PredictionLogger
from shared.serialise_clinical import fit_fills, text_key

from .config import CONFIGS, EMB_DIR, ENCODERS, ESTIMATORS, INNER_FRAC, LR_C_GRID, PCA_DIM, POOLINGS, PRED_DIR, \
    SEEDS, TABULAR_CONFIGS, TEXT_COHORT_CONFIGS
from .tabular import FullClinicalPreprocessor
from .texts import FOLD_IMPUTED, SPLITTER, fold_plan, load_frames, texts

logger = logging.getLogger("exp19")
EMBEDDING_INPUTS = ("emb:", "rep:", "smiles")


@lru_cache(maxsize=None)
def store(encoder: str, pooling: str) -> tuple[dict, np.ndarray]:
    keys = (EMB_DIR / f"{encoder}_store.keys.txt").read_text().split()
    arr = np.load(EMB_DIR / f"{encoder}_store_{pooling}.npy")
    if len(arr) != len(keys):
        raise RuntimeError(f"{encoder} store is inconsistent")
    return {k: i for i, k in enumerate(keys)}, arr


def lookup(encoder: str, pooling: str, text_list: list[str]) -> torch.Tensor:
    index, arr = store(encoder, pooling)
    rows = [index.get(text_key(t)) for t in text_list]
    if any(r is None for r in rows):
        raise KeyError(f"{sum(r is None for r in rows)} texts missing from the {encoder} store; run embed.py")
    return torch.from_numpy(arr[rows].astype(np.float32))


def _is_embedding(name: str) -> bool:
    return name.startswith(EMBEDDING_INPUTS)


def fold_inputs(names, frames, scope, encoder, pooling, fit, smiles_lookup, fills):
    """Raw input tensors (Melbourne, HEP1) for one fold, before standardisation."""
    mel, hep = frames[("MEL", scope)], frames[("HEP", scope)]
    out_mel, out_hep = [], []
    for name in names:
        if name == "smiles":
            m = torch.from_numpy(build_smiles_feature_matrix(mel, smiles_lookup))
            h = torch.from_numpy(build_smiles_feature_matrix(hep, smiles_lookup))
        elif name == "clinical":
            m, h = portable.refit_clinical(mel, hep, fit)
        elif name == "clinical_full":
            pre = FullClinicalPreprocessor().fit(mel.iloc[fit])
            m, h = torch.from_numpy(pre.transform(mel)), torch.from_numpy(pre.transform(hep))
        elif name.startswith("emb:"):
            variant = name[4:]
            f = fills if variant in FOLD_IMPUTED else None
            m = lookup(encoder, pooling, texts(mel, variant, f))
            h = lookup(encoder, pooling, texts(hep, variant, f))
        elif name.startswith("rep:"):
            m = lookup(name[4:], "mean", texts(mel, "rep"))
            h = lookup(name[4:], "mean", texts(hep, "rep"))
        else:
            raise ValueError(name)
        out_mel.append(m.float())
        out_hep.append(h.float())
    return out_mel, out_hep


def standardise(names, t_mel, t_hep, fit, pca: bool):
    """z-score embedding inputs on the fit rows (and, for PCA32, project them to
    PCA_DIM fit-row components). Clinical inputs pass through unchanged."""
    s_mel, s_hep = [], []
    for name, m, h in zip(names, t_mel, t_hep):
        if _is_embedding(name):
            mu = m[fit].mean(0)
            sd = m[fit].std(0, unbiased=False).clamp(min=1e-6)
            m, h = (m - mu) / sd, (h - mu) / sd
            if pca:
                p = PCA(n_components=min(PCA_DIM, len(fit) - 1), svd_solver="full").fit(m[fit].numpy())
                m = torch.from_numpy(p.transform(m.numpy()).astype(np.float32))
                h = torch.from_numpy(p.transform(h.numpy()).astype(np.float32))
        s_mel.append(m)
        s_hep.append(h)
    return s_mel, s_hep


def fit_predict_lr(t_mel, t_hep, y, fit, es, test):
    """L2 logistic regression: C chosen by early-stopping-set AUC, fitted on the fit rows."""
    x_mel, x_hep = torch.cat(t_mel, 1).numpy(), torch.cat(t_hep, 1).numpy()
    best = None
    for c in LR_C_GRID:
        lr = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(x_mel[fit], y[fit])
        auc = roc_auc_score(y[es], lr.predict_proba(x_mel[es])[:, 1])
        if best is None or auc > best[0]:
            best = (auc, lr)
    lr = best[1]
    es_p, te_p = lr.predict_proba(x_mel[es])[:, 1], lr.predict_proba(x_mel[test])[:, 1]
    return es_p, te_p, lr.predict_proba(x_hep)[:, 1]


def run_one(frames, config, encoder, pooling, estimator, seed, device, smiles_lookup):
    names = CONFIGS[config]
    scope = "text" if config in TEXT_COHORT_CONFIGS else "full"
    mel, hep = frames[("MEL", scope)], frames[("HEP", scope)]
    y = mel["outcome"].to_numpy().astype(np.int64)
    tag = config if encoder is None else f"{config}_{encoder}_{pooling}"
    lg = PredictionLogger(
        exp_id=f"exp19_{tag}_{estimator}", output_dir=PRED_DIR,
        filename=f"predictions_oof_exp19_{tag}_{estimator}{cv_suffix(SPLITTER, INNER_FRAC)}_s{seed}.json",
        metadata={"splitter": SPLITTER, "inner_val": INNER_FRAC, "config": config, "inputs": list(names),
                  "encoder": encoder, "pooling": pooling, "estimator": estimator, "seed": seed},
    )
    hep_probs = []
    full_mel = frames[("MEL", "full")]
    for fold, test, fit, es in fold_plan(mel, seed):
        enable_determinism(seed + fold)
        # Imputed texts use the fold's fills from the full cohort's fit rows (the
        # only configurations with imputed texts run on the full cohort).
        fills = fit_fills(full_mel.iloc[fit]) if scope == "full" else None
        t_mel, t_hep = fold_inputs(names, frames, scope, encoder, pooling, fit, smiles_lookup, fills)
        t_mel, t_hep = standardise(names, t_mel, t_hep, fit, pca=estimator == "pca32")
        if estimator == "lr":
            es_p, te_p, hep_p = fit_predict_lr(t_mel, t_hep, y, fit, es, test)
        else:
            dims = [t.shape[1] for t in t_mel]
            cfg = f"{portable.LATE_FUSION_PREFIX}{tag}"
            yt = torch.from_numpy(y)
            model = portable.train_fold(cfg, [t[fit] for t in t_mel], [t[es] for t in t_mel], yt[fit], yt[es],
                                        device, model_factory=lambda: portable.LateFusionMLP(dims))
            es_p = portable.predict(model, [t[es] for t in t_mel], cfg, device)
            te_p = portable.predict(model, [t[test] for t in t_mel], cfg, device)
            hep_p = portable.predict(model, t_hep, cfg, device)
        lg.log_fold(fold, mel["pid"].astype(str).iloc[test], y[test], te_p, threshold=youden_threshold(y[es], es_p))
        hep_probs.append(hep_p)
    lg.save()
    pd.DataFrame({"pid": hep["pid"].astype(str), "y_true": hep["outcome"].astype(int),
                  "y_prob": np.mean(hep_probs, axis=0)}).to_csv(
        PRED_DIR / f"hep_external_exp19_{tag}_{estimator}_s{seed}.csv", index=False)


def planned_runs(configs, encoders, estimators):
    runs = []
    for config in configs:
        has_embedding = any(_is_embedding(n) for n in CONFIGS[config])
        ests = [e for e in estimators if e != "pca32" or has_embedding]
        if config in TABULAR_CONFIGS:
            runs += [(config, None, None, e) for e in ests]
            continue
        for encoder in encoders:
            for pooling in POOLINGS[ENCODERS[encoder]["kind"]]:
                runs += [(config, encoder, pooling, e) for e in ests]
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 19: serialised-text vs tabular representations")
    parser.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--encoders", nargs="+", choices=sorted(ENCODERS), default=list(ENCODERS))
    parser.add_argument("--estimators", nargs="+", choices=ESTIMATORS, default=list(ESTIMATORS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    frames, smiles_lookup = load_frames(), build_smiles_lookup()
    runs = planned_runs(args.configs, args.encoders, args.estimators)
    logger.info(f"{len(runs)} runs x {len(args.seeds)} seeds")
    for config, encoder, pooling, estimator in runs:
        tag = config if encoder is None else f"{config}_{encoder}_{pooling}"
        for seed in args.seeds:
            out = PRED_DIR / f"predictions_oof_exp19_{tag}_{estimator}{cv_suffix(SPLITTER, INNER_FRAC)}_s{seed}.json"
            if out.exists() and not args.force:
                continue
            run_one(frames, config, encoder, pooling, estimator, seed, device, smiles_lookup)
        logger.info(f"done {tag} {estimator}")


if __name__ == "__main__":
    main()
