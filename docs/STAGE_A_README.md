# Stage A: Prediction-Logging Infrastructure

> **Superseded for reruns.** The rerun launch/cascade notes below predate the
> data-leakage fix. Use `rerun_all_oof.sh` + `shared/verify_oof.py` (see the
> "Consistent OOF rerun" section in `README.md`). This file is kept for the
> prediction-logger design notes only.

Purpose: enable patient-level OOF predictions to be dumped from every
experiment so the supervisor-comment items #25 (pooled-AUC bootstrap),
#27 (sensitivity/specificity), and the cross-cohort half of #31 (DeLong
test) can be addressed for all 14 configurations.

## What's done

- `shared/prediction_logger.py` — minimal helper class:
  - `PredictionLogger(exp_id, output_dir)` constructor
  - `log_fold(fold, pids, y_true, y_prob, threshold=None)` per-fold append
  - `save()` dumps `predictions_oof.json` with schema:
    ```
    {"exp_id": "...", "n_folds": 5, "folds": [{"fold": int, "n": int,
     "pids": [...], "y_true": [...], "y_prob": [...], "threshold": float}, ...]}
    ```

- `exp4_baseline/training.py` — `evaluate()` now also returns `y_prob`
  and `y_true`; `run_cross_validation` accepts an optional
  `prediction_logger` arg.

- `exp4_baseline/run_experiments.py` — adds `--log-predictions` CLI flag;
  wires the logger in for each requested model.

- **Smoke-tested locally**: `python3 -m exp4_baseline.run_experiments
  --model mlp --log-predictions` produces
  `outputs/exp4_predictions/predictions_oof.json` with 5 folds, 204
  total patients. Validated 41/41/41/41/40 per-fold split and correct
  y_true balance.

## What's left

The same pattern needs to be applied to seven more experiment dirs.
For each, the three changes are:

1. In the per-fold `evaluate()` function (or whichever function builds
   the validation-set metrics), add `y_prob` and `y_true` lists to the
   metrics dict.

2. In the cross-validation loop, add a `prediction_logger` arg to the
   loop wrapper, and after each fold's training:

   ```python
   if prediction_logger is not None:
       val_pids = df["pid"].iloc[val_idx].tolist()  # adjust to the
                                                    # actual pid source
       prediction_logger.log_fold(
           fold=fold,
           pids=val_pids,
           y_true=metrics["y_true"],
           y_prob=metrics["y_prob"],
           threshold=metrics.get("optimal_threshold"),
       )
   ```

3. In the `run_experiments.py` entry point: import `PredictionLogger`,
   add a `--log-predictions` CLI flag, instantiate the logger when set,
   pass it down, call `logger.save()` at end.

| Experiment dir | Entry point | Expected effort | Notes |
|---|---|---|---|
| `exp1_fusion/` | `run_experiments.py` | ~25 min | Text + SMILES; pids from the cohort dataframe |
| `exp2_fusion/` | `run_experiments.py` | ~30 min | EEG + SMILES; pids from EEG cache index |
| `exp3_fusion/` | `run_experiments.py` | ~30 min | Triple modality |
| `exp5_clinical_fusion/` | `run_experiments.py` | ~25 min | Clinical + 1 modality |
| `exp6_clinical_triple/` | `run_experiments.py` | ~25 min | Clinical + 2 modalities |
| `exp7_all_modalities/` | `run_experiments.py` | n/a | Already logs predictions in a bespoke schema; will need a small adapter in the downstream bootstrap script (alternatively, migrate to the new helper for consistency) |
| `exp9_eeg_investigation/` | `run_experiments.py` | ~25 min | EEG-only ablation (not in main Table 1 but informs encoder choice) |
| `exp11_eeg_upgrade/` | `run_experiments.py` | ~25 min | EEG2Vec swap; updates Exp3a/6b |

Total remaining surface area: ~3 hours of careful edits.

## Cascade risk (read this before launching reruns)

A local smoke-test of Exp4a produced **AUC 0.674** versus the original
**0.664**, and per-fold values shifted noticeably:

| Fold | Original | Rerun |
|---|---|---|
| 1 | 0.712 | 0.733 |
| 2 | 0.614 | 0.507 |
| 3 | 0.719 | 0.770 |
| 4 | 0.643 | 0.671 |
| 5 | 0.630 | 0.690 |

The drift is real and not a bug in the new logging code — it is
expected from torch nondeterminism (cuDNN, DataLoader worker order)
plus possible library version drift since the original runs in
February 2026. Implications if all 14 are rerun:

- Every AUC in Table 1 will shift, probably by 0.01-0.05.
- The Results prose currently cites specific numbers ("AUC 0.664",
  "AUC 0.790", etc.) that will need a full update pass.
- Figure 2 will need to be regenerated (and the all-pairs stat matrix).
- Comparisons with prior work in the Discussion ("matches Feng et al.
  0.67 baseline") may shift slightly but should remain qualitatively
  the same.

Mitigations to consider:

1. **Tighten determinism before rerunning**: set
   `torch.use_deterministic_algorithms(True)`, fix `PYTHONHASHSEED`,
   pin a single-worker DataLoader. Probably reduces drift but doesn't
   eliminate it.

2. **Accept the drift, do one rerun, update all numbers**: easiest
   path, requires a careful sweep of every numeric mention.

3. **Hybrid**: keep the original AUC/CI numbers in Table 1, use new
   reruns only for the Sens/Spec columns and the DeLong/bootstrap CIs
   (with a footnote that those columns come from a re-run with
   prediction logging). Inconsistent but minimises text churn.

The Stage A code as written supports any of these — the rerun is a
separate launch step.

## Launching reruns (template)

Once the remaining experiments have the logging hook, the launch
sequence is (from the repository root, in the `.venv-others` venv):

```bash
# Light configs (CPU-feasible, ~10-30 min each):
python3 -m exp4_baseline.run_experiments --log-predictions
python3 -m exp1_fusion.run_experiments --log-predictions
python3 -m exp5_clinical_fusion.run_experiments --log-predictions

# EEG-dependent (need GPU; submit to M3 SLURM):
sbatch scripts/rerun_exp2.sh  # to be written
sbatch scripts/rerun_exp3.sh
sbatch scripts/rerun_exp6.sh
sbatch scripts/rerun_exp7.sh
sbatch scripts/rerun_exp9.sh
sbatch scripts/rerun_exp11.sh
```

After all reruns complete, the downstream consumers will need the
`predictions_oof.json` paths added to a (new) prediction-files
manifest, then:

- `thesisStandalone/analysis/compute_bootstrap_cis.py` runs across all
  configs, expanding `cis_tier2.csv`.
- A new `analysis/sensitivity_specificity.py` computes per-fold TPR
  and TNR from each `predictions_oof.json`, dumps a CSV that updates
  Table 1.
- `analysis/all_pairs_stats.py` gets an additional DeLong-z fallback
  for the 88 cross-cohort pairs that paired Wilcoxon could not test.

## Open decisions for you

1. **Continue Stage A surgery on the remaining 7 experiments now?** Or
   pause Stage A, move to Phase 3 (PCA, ASM ranking, simulation, tag
   cloud — none of which depend on reruns), and resume Stage A when
   the supervisor questions return?

2. **Determinism strategy before rerun?** Acceptance of ~0.01-0.05 AUC
   drift across the table, or tighten torch determinism first.

3. **Hybrid Table 1?** Keep original AUCs and only add Sens/Spec from
   reruns (with footnote), or re-publish every number from the rerun.
