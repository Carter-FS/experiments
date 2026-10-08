# Stage B: ASM-balanced training

> **Reruns:** `--asm-balance weighted` is now wired into every table
> experiment and the whole set regenerates via `rerun_all_oof.sh` (see the
> "Consistent OOF rerun" section in `README.md`). `stratified_batch` remains
> available only in the experiments that implement the sampler (exp3/5/6/7/15).

Addresses Duong's follow-up email to the Phase 3b best-ASM simulation
finding (LEV recommended for 98 % of patients, mirroring the LEV-dominated
training distribution).

## What's implemented

- `shared/asm_balancing.py` provides:
  - `compute_asm_sample_weights(asm_labels, min_count_floor=0)` -- inverse
    sqrt sample weights normalised so the mean weight is 1.
  - `StratifiedASMBatchSampler` -- PyTorch BatchSampler that ensures
    every mini-batch contains at least one sample per ASM.
  - `WeightedASMDataset` wrapper that threads per-sample weights through
    the default collate.
  - `weighted_cross_entropy(logits, targets, sample_weights, class_weight)`
    -- per-sample weighted CE preserving the existing outcome class
    weighting.

- `exp7_all_modalities/` is wired end-to-end for both
  `--asm-balance weighted` and `--asm-balance stratified_batch` in
  `predictions` mode (the mode that produces `y_prob_per_asm` for the
  best-ASM simulation comparison). Smoke tested locally: 5-fold CV
  completes, predictions land in
  `outputs/exp7_predictions/predictions_oof_asmweighted.json` and
  `..._asmstratbatch.json` without overwriting the Stage A baseline.

- `launch_stage_b_reruns.sh` submits the two Stage B variants via the
  existing `submit_job.sh` infrastructure.

## What's pending

- Wiring for exp3 (Triple, no clinical), exp5 (Clinical + single modality),
  and exp6 (Clinical + dual modality) per the Stage B plan.
  Each follows the same four-edit pattern used in exp7:
    1. Add `asm_balance_mode` arg to `train_fold` (and per-modality
       siblings).
    2. Wrap `train_dataset` with `WeightedASMDataset` when weighted,
       swap the DataLoader to use `StratifiedASMBatchSampler` when
       stratified.
    3. Pass `asm_weighted=True` and `class_weights` through to the
       per-epoch train fn so it switches to `weighted_cross_entropy`.
    4. Add `--asm-balance` CLI flag to `run_experiments.py`.

  Estimated effort: ~45 min per experiment, similar to the exp7 work.
  These experiments do *not* expose `y_prob_per_asm` so they will only
  report AUC and sens/spec deltas under each balancing strategy, not a
  recommendation-distribution shift (which is the LEV-bias diagnostic).

- `analysis/asm_balancing_comparison.py` -- one-shot comparison script
  that loads `predictions_oof.json` (baseline), `..._asmweighted.json`,
  `..._asmstratbatch.json` and produces: AUC delta table,
  recommendation-distribution comparison (exp7 only), sens/spec table,
  combined figure.

## Caveats

The TPM (n=1) and PTN (n=2) ASMs get heavy upweighting under the
inverse-sqrt scheme. On the 111-patient quad-modal cohort the max
sample weight is around 2.1; on the 147-patient cohort it reaches
about 5.2. Training does not appear to destabilise at these values in
the local smoke tests, but per-fold loss curves should be inspected on
the M3 runs.

The stratified-batch sampler requires `batch_size >= n_unique_ASMs`.
With 6 ASMs (post CBZ-typo collapse) and `batch_size=8`, 6 of every 8
slots are fixed by stratification and 2 are random. Effective shuffling
is reduced; flag in any methods description.

## Running on M3

```bash
cd /fs04/scratch2/he12/carter/experiments
bash launch_stage_b_reruns.sh                 # submit both jobs
squeue -u $USER                               # monitor
ls outputs/exp7_predictions/predictions_*.json  # confirm landing
```
