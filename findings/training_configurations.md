# Training configurations (clean rerun)

Effective training settings for every model in the paper, as used in the final clean
rerun: refit protocol, prediction files suffixed `_sp-multilabel_rf5_s<seed>`, seeds
42-46, run on M3 by `rerun_clean.sh`. Values were read at the point of use in the
code, not only from `config.py`, because some config values are overridden. Parameter
counts were obtained by building each model on CPU (no training). Compiled 2026-10-09.

Where to find them in the repository:

| What | File |
|---|---|
| Exact commands and flags for every run | `rerun_clean.sh` (the `items` list) |
| Outer and inner cross-validation, thresholds, seeds | `shared/cv_splits.py`, `shared/epoch_selection.py`, `shared/determinism.py` |
| Default hyperparameters per experiment | `exp*/config.py` |
| Model definitions | `exp*/models.py`, `exp2_fusion/models/`, `exp1_fusion/` |
| Training loops | `exp*/training.py`, `exp9_eeg_investigation/run_experiments.py` |
| Cross-cohort and HEP1 models | `shared/portable_models.py`, `exp18_mixed_cohort/` |
| ASM balancing | `shared/asm_balancing.py` |
| Pre-registered plan and deviations | `docs/analysis_plan_clean_rerun_exp18.md` (Addenda A and B) |


## 1. Protocol common to every model

| Setting | Value | Source |
|---|---|---|
| Outer CV | 5 folds, iterative multilabel stratification on outcome, focal epilepsy and sex | `shared/cv_splits.py:94-114`; `rerun_clean.sh:127` |
| Repeats | 5 seeds (42, 43, 44, 45, 46); each seed re-splits the outer folds | `rerun_clean.sh` |
| De-duplication | one record per patient before splitting | `shared/cohort.py` |
| Inner CV (epoch and threshold choice) | StratifiedKFold(5) on outcome only, seed = cv-seed + fold | `shared/epoch_selection.py:56-65` |
| Inner runs | the experiment's own loop, with early stopping on inner-validation AUC; a run that stops early carries its last predictions forward | `shared/epoch_selection.py:106-107` |
| Epoch choice (E*) | pooled inner out-of-fold AUC per epoch, centred 3-epoch moving average, earliest maximum | `shared/epoch_selection.py:111-113` |
| Threshold | Youden's J on the pooled inner out-of-fold predictions at E* (0.5 if undefined) | `shared/epoch_selection.py:120`; `shared/cv_splits.py:225-247` |
| Refit | fresh model on the whole outer training fold for exactly E* epochs; final weights scored once on the outer test fold | `shared/epoch_selection.py:193-202` |
| Optimiser | AdamW | each training loop |
| Loss | cross-entropy, outcome class weights 1/count normalised to sum 1, from the current training split | each training loop |
| Mixed precision | none | |
| Determinism | `--deterministic`: seeds Python, NumPy and PyTorch; cuDNN deterministic, benchmark off; deterministic algorithms with warnings only | `shared/determinism.py:29-53` |
| Clinical input | 23 dims: 13 binary (mode imputation), age in 4 bands (mean imputation before banding), MRI and EEG findings one-hot over 3 levels; preprocessor fitted on each training split | `exp4_baseline/data_pipeline.py:175-229` |
| Cross-cohort clinical input | 20 dims (drug, alcohol and focal dropped; constant in HEP1) | `shared/hep_cohort.py:74` |
| Text input | precomputed frozen embeddings, 768 dims: ClinicalBERT (pre-specified) or PubMedBERT, mask-aware token mean | `outputs/*.npy` (not in git) |
| Drug input | precomputed frozen SMILES embeddings: ChemBERTa 768 dims (pre-specified, token mean) or SMILES Transformer 256 dims | `exp1_misc/e1_LLM+SMILES_ChemBERTa.py:33-67` |
| EEG input, main cache (27 channels) | 200 Hz, 0.1-75 Hz bandpass, 50 Hz notch; first 300 s skipped, next 1200 s used, recordings under 600 s rejected; 10 s windows (2000 samples), at most 120, zero-padded with a padding mask; channels padded to 27; no amplitude normalisation (signals in volts); built once for the whole cohort | `exp2_fusion/config.py:17-29`; `exp2_fusion/eeg_pipeline.py:779` |
| EEG input, cross-cohort caches (19 channels) | 19 standard 10-20 channels, 50 Hz (Melbourne) or 60 Hz (HEP1) notch, per-window z-score, amplitude clipped at 5 SD | `thesisStandalone/analysis/hep_eeg_preprocess.py:66-71` |
| EEG encoders | every EEG window encoder in exp2-7, 9, 11, 16, 17 is trained from scratch with the model; EEG2Vec uses its mean head only (no KL term) | `exp2_fusion/models/eeg_encoders.py` |
| Frozen pretrained models | ClinicalBERT, PubMedBERT, ChemBERTa, SMILES Transformer (embeddings precomputed); REVE-base (features precomputed, exp15 and REVE standalone only) | |


## 1a. Preprocessing: EEG conventions and the comparison with the supervisor's pipeline

Compiled 2026-10-09.
The comparison is against Duong Nhu's `code-fury/eeg-foundation-model` (master,
2026-09-14), whose benchmark pipeline for a pretrained encoder (the REVE benchmark) is
`benchmark/preprocessing/preprocess_multichannel.py` run with `--bandpass-low 0.5
--bandpass-high 70 --resampling-frequency 200 --power-noise-frequency 60
--epoch-length 16 --normalization zscore`.

The EEG rerun uses the per-epoch, per-channel z-score of that pipeline
(`zscore_norm_epoch`: mean and standard deviation over the time axis of each window, floor
1e-6, no clipping, in microvolts), applied to each 10-second window at load time, for every
trained-from-scratch encoder and for REVE. The pretrained LaBraM input is microvolts
divided by 100, as in its official code.

Pretrained LaBraM channel names (`shared/labram_pretrained.py`): LaBraM selects a
learned channel embedding by name from the official `standard_1020` list, which holds
separate rows for the legacy temporal names T3/T4/T5/T6 (positions 88-91) and the modern
names T7/T8/P7/P8 (positions 37, 45, 59, 67). The official clinical fine-tuning runs
(TUAB and TUEV in `run_class_finetuning.py`) name the TUH channels T3/T4/T5/T6, so the
four temporal channels of the cache (stored as T7/T8/P7/P8) are given to LaBraM under
the legacy names; the other 15 names are unchanged. Duong's repository has no LaBraM
path to compare with. braindecode's canonical list matches `standard_1020` position
for position over its 128 entries, and each of the 221 hub tensors is bit-identical to
the official `labram-base.pth` tensor of the corresponding name (`student.*` encoder
weights; the eight official tensors left over are the pretraining-only tokenizer heads,
logit scale, mask token and projection head). Per-window features are the mean over
the patch tokens through a parameter-free LayerNorm (`use_mean_pooling=True`, the
official fine-tuning default) with the [CLS] token stored alongside.

| Aspect | Ours (rerun) | Duong (benchmark z-score path) | Assessment |
|---|---|---|---|
| Units | microvolts (`get_data(units="uV")`) | microvolts | same |
| Amplitude normalisation | per-window (10 s), per-channel z-score, floor 1e-6, no clipping | per-epoch (16 s), per-channel z-score, floor 1e-6, no clipping | same convention; window length differs (ours pre-registered) |
| Where statistics come from | the window itself | the epoch itself (his own model: causal per-recording EMA RMS) | same; nothing fitted on the training split in either |
| Band-pass | 0.1-75 Hz, zero-phase FIR (MNE), after resampling | 0.5-70 Hz, causal Butterworth order 4, before resampling | differs; ours follows LaBraM's pretraining band and is pre-registered; kept |
| Mains | FIR notch 50 Hz (60 Hz HEP1) | ZapLine (`meegkit.dss.dss_line`) | differs; both standard; kept |
| Resampling | MNE `resample` to 200 Hz | `resample_poly` to 200 Hz | same rate |
| Segment | skip 300 s, next 1200 s, reject under 600 s | skip leading flat data, then 300 s, drop the last 10 s, use the rest (his loader then takes at most 38 epochs) | ours pre-registered; the leading-flat detection is adopted in the rerun |
| Epochs | 10 s, no overlap, at most 120, zero-padded and masked | 16 s (8 s TUSZ), zero-padded and masked | differs; ours pre-registered |
| Channels | 19 standard 10-20 by name, canonical order, recording dropped if any is missing | the same 19 by name (TUH names), zero-filled with a mask if missing | same set; order differs (irrelevant, encoders resolve by name or position); missing-channel policy stricter in ours |
| Old channel names | T3/T4/T5/T6 to T7/T8/P7/P8 | same aliases | same |
| Reference | as recorded | as recorded | same |
| Loss and balance | class-weighted cross-entropy | unweighted BCE with a balanced batch sampler | differs; two routes to the same end |
| Optimiser | AdamW, weight decay 1e-4 | AdamW, lr 1e-4, weight decay 1e-4 (v2: 1e-3 with cosine) | close |
| Precision | fp32 | bf16-mixed, torch.compile | differs; fp32 kept for reproducibility |
| Model selection | inner 5-fold CV, refit, patience 20 | early stopping on validation AP, patience 10, best checkpoint | ours stricter |
| Gradient clipping | 1.0 | none | differs; kept |
| Window aggregation | 2-layer transformer with sinusoidal positions, masked mean | 6-layer pre-norm transformer, learned-query pooling (8 queries) | differs; design choice |
| REVE features | attention-pooled 512 per window, frozen | all tokens kept (19 x 17 x 512), pooled in the classifier, frozen | differs; frozen use matches |
| FuseMoE | adapted from Duong's `models/fuse_moe.py`: same Laplace gating, same unweighted top-k sum, mutual-information loss | reference | core identical; the unweighted sum is inherited from the reference |

Differences that should not have existed, all fixed by the EEG rerun: the 27-channel cache
in volts with no normalisation; the EMG, PG, ECG and ear-reference channels in it; the
paper's "z-scored per window" and "pretrained LaBraM" statements; the absence of
leading-flat detection.

## 2. Main configurations (Table 2 of the paper)

All late-fusion MLPs: each modality passes through Linear to 64, ReLU, LayerNorm and
Dropout 0.3; the 64-dim projections are concatenated and passed to Linear to 64, ReLU,
LayerNorm, Dropout 0.3, Linear to 2. Gradient clipping at norm 1.0, no LR scheduler.

| Paper name | Code id | Inputs (dims) | Model | Params | LR | WD | Batch | Max epochs | Patience |
|---|---|---|---|---|---|---|---|---|---|
| Clinical features only | exp4a_mlp | clinical 23 | MLP 23 -> 64 -> 32 -> 2 (ReLU, LayerNorm, Dropout 0.3) | not logged | 1e-3 | 1e-4 | 16 | 100 | 20 |
| Clinical features only, self-attention | exp4b_attention | clinical 23 | per-feature Linear(1,64) + positional embedding + CLS; transformer d 64, 4 heads, 2 layers, FFN 256, dropout 0.2; head 64 -> 32 -> 2 | not logged | 5e-4 | 1e-4 | 16 | 100 | 20 |
| Clinical Model | exp5a_chemberta | clinical 23, ChemBERTa 768 | late-fusion MLP (2 branches, concat 128) | not logged | 1e-3 | 1e-4 | 16 | 100 | 20 |
| Clinical + Text (no drug) | exp5b_clinicalbert | clinical 23, ClinicalBERT 768 | late-fusion MLP (2 branches) | not logged | 1e-3 | 1e-4 | 16 | 100 | 20 |
| Clinical + EEG (no drug) | exp5c_eeg2vec | clinical 23, EEG windows | late-fusion MLP; EEG2Vec (256 per window) -> transformer aggregator (d 256, 4 heads, 2 layers, FFN 512, dropout 0.3) -> 64 | not logged | 1e-3 | 1e-4 | 16 | 100 | 20 |
| Clinical + Text Model | exp6a_clinicalbert_chemberta | clinical, ClinicalBERT, ChemBERTa | late-fusion MLP (3 branches, concat 192) | not logged | 1e-3 | 1e-4 | 16 | 100 | 20 |
| Clinical + EEG Model | exp6b_eeg2vec_chemberta | clinical, EEG, ChemBERTa | late-fusion MLP (3 branches); EEG branch as exp5c | not logged | 1e-3 | 1e-4 | 8 | 100 | 20 |
| Full Multi-Modal Fusion Model | exp7a | clinical, ClinicalBERT, EEG, ChemBERTa | late-fusion MLP (4 branches, concat 256); EEG branch as exp5c | 1,698,178 | 1e-3 | 1e-4 | 8 | 100 | 20 |
| Text | exp1a_clinicalbert_chemberta | ClinicalBERT, ChemBERTa | concat -> 512 -> 256 -> 128 -> 2 (ReLU, LayerNorm, Dropout 0.3/0.3/0.2) | 953,218 | 1e-4 | 1e-4 | 16 | 100 | 15 |
| EEG | exp2_eeg2vec_chemberta_mlp | EEG, ChemBERTa | EEG2Vec -> transformer aggregator (FFN 512, GELU, dropout 0.1) -> 256; SMILES -> 256; concat 512 -> 256 -> 128 -> 2 | 1,926,786 | 5e-5 | 1e-4 | 8 | 100 | 20 |
| Text + EEG | exp3a_clinicalbert_chemberta_eeg2vec | ClinicalBERT, EEG, ChemBERTa | text and SMILES -> 256; EEG2Vec -> aggregator (dropout 0.1) -> 256; concat 768 -> 256 (Dropout 0.3) -> 128 (Dropout 0.2) -> 2 | 2,190,210 | 1e-4 | 1e-4 | 8 | 100 | 20 |

LR scheduler: exp1 and exp2 use ReduceLROnPlateau (maximise validation AUC, factor 0.5,
patience 5) in the inner runs, and the refit replays the per-epoch median learning rate
of the inner runs. Every other configuration uses a constant learning rate.


## 3. Fusion and encoder variants

| Variant | Code id | Difference from the main configuration | Params | LR | Batch |
|---|---|---|---|---|---|
| Text, mixture-of-experts | exp1b_* | each modality -> 256 + modality token; 2 MoE fusion layers (4-head self-attention, per-modality FuseMoE: 4 experts, top-2, hidden 256, FFN 1024); auxiliary router loss (negative mutual information) x 0.1; temperature fixed at 1.0 | 8,574,722 | 5e-5 | 16 |
| EEG (SimpleCNN), late-fusion MLP | exp2_simplecnn_chemberta_mlp | SimpleCNN window encoder (Conv1d 64/128/256/256, kernels 25/15/9/5, BatchNorm, ReLU, Dropout 0.1, global average pool, Linear to 200) | 1,990,378 | 5e-5 | 8 |
| EEG (SimpleCNN), mixture-of-experts | exp2_simplecnn_chemberta_fusemoe | EEG cross-attends to SMILES, joint FuseMoE (4 experts, top-2, hidden 256), auxiliary loss x 0.1, temperature 1.0 | about 5.12M | 5e-5 | 8 |
| Text + EEG (SimpleCNN), MLP | exp3a_clinicalbert_chemberta | exp3a with SimpleCNN (256 per window) | 2,536,834 | 1e-4 | 8 |
| Text + EEG (SimpleCNN), mixture-of-experts | exp3b_clinicalbert_chemberta | 3-token self-attention, one per-modality FuseMoE (4 experts, top-2, hidden 256), auxiliary loss x 0.1; temperature annealed 1.0 to 0.5 (x 0.9995 per step, rescaled in the refit); no scheduler | 6,068,226 | 5e-5 | 8 |
| Full model, mixture-of-experts | 7b | each modality -> 256; modality tokens; one 4-head cross-modal attention layer; one per-modality FuseMoE (4 experts, top-2, each expert 3 residual blocks with Dropout 0.2); head LayerNorm, Dropout 0.1, Linear to 2; auxiliary loss x 0.1; temperature 1.0; EEG aggregator dropout 0.1 | 6,255,362 | 5e-5 | 8 |
| Full model, inverse-frequency ASM weighting | exp7 `asmweighted` | per-sample loss weight 1/sqrt(n prescribed ASM), mean-normalised on the training split | 1,698,178 | 1e-3 | 8 |
| Full model, stratified ASM mini-batches | exp7 `asmstratbatch` | batch sampler: every batch of 8 holds at least one patient per ASM (rare ASMs resampled), the rest drawn at random; ceil(n/8) batches per epoch | 1,698,178 | 1e-3 | 8 |
| Full model, REVE-base EEG | exp15 | EEG branch replaced by frozen pretrained REVE-base features (19 channels, 512 per window, 120 windows), trained Linear 512 -> 256 into the aggregator | 1,319,554 | 1e-3 | 8 |
| Full model, reduced capacity (small) | exp16_small | hidden 32, EEG2Vec embedding 64, mean-max aggregator | 187,458 | 1e-3 | 8 |
| Full model, reduced capacity (tiny) | exp16_tiny | hidden 16, otherwise as small | 157,154 | 1e-3 | 8 |
| Full model, focal epilepsy only | exp17_focal | exp7a trained and tested on patients with focal epilepsy | 1,698,178 | 1e-3 | 8 |
| EEG upgrade grid | exp11_* | EEG2Vec embedding 128; transformer (4 heads, 2 layers) or mean-max aggregator; 3a variants lr 1e-4 with aggregator dropout 0.1, 6b and 7a variants lr 1e-3 with dropout 0.3 | not logged | as stated | 8 |

All variants: weight decay 1e-4, maximum 100 epochs, patience 20, gradient clipping 1.0.


## 4. Standalone EEG encoder comparison (Supplementary encoder table)

`exp9_eeg_investigation/run_experiments.py`. Each arm is EEG plus the ChemBERTa drug
embedding (SMILES -> out with LayerNorm; head 2*out -> out with LayerNorm, ReLU,
Dropout 0.3 -> 2). Window aggregator: transformer, dropout 0.1. Learning rate 1e-3
(hard-coded), weight decay 1e-4, batch 8 (1 for LaBraM), maximum 100 epochs,
patience 20, gradient clipping 1.0, no ASM balancing.

| Arm | Window encoder | Embedding | Pretrained? |
|---|---|---|---|
| SimpleCNN | 4 Conv1d layers, Dropout 0.1 | 256 | no, trained from scratch |
| EEGNet | braindecode EEGNet (F1 8, D 2, F2 16) | 256 | no, trained from scratch |
| EEG2Vec | EEGNet-style variational encoder (mean head only) | 256 | no, architecture reimplemented and trained from scratch |
| LaBraM (scratch), arm `encoder_labram_scratch` | braindecode 1.2 Labram architecture (2 layers, 4 heads, patch 200) | 128 | no: the architecture only, trained from scratch, not the published pretrained weights |
| LaBraM-base (pretrained, frozen), arm `encoder_labram_pretrained_frozen` | published LaBraM-base weights (12 layers, 10 heads, patch 200; Section 1a), features computed once per window (`shared/labram_pretrained.py`), read as the `labram_v2` feature set | 200 | yes, frozen |
| REVE-base (pretrained, frozen), arm `encoder_reve_frozen` | published REVE-base weights, attention-pooled features computed once per window (`reve_extract_features.py`), read as the `reve_v2` feature set | 512 | yes, frozen |

Both frozen arms go through the same aggregator and head as the trained-from-scratch
encoders (encoder type `precomputed`, `shared/eeg_features.py`); the standalone REVE
script no longer supplies the encoder-table row. The raw-EEG arms read the cache under
the per-window z-score; the feature sets were written from the same cache. A fine-tuned
LaBraM-base arm is scheduled after the frozen-feature results.

The exp9 aggregator ablations (attention, LSTM, max, mean-max, depth 1 and 4, embedding
64 and 128) use the same settings. `aggregator_depth_0` is the same model as
`aggregator_attention`.


## 5. Cross-cohort and HEP1 models (`shared/portable_models.py`)

| Setting | Non-EEG configurations | EEG configurations |
|---|---|---|
| Optimiser, LR, weight decay | AdamW, 1e-3, 1e-4 | AdamW, 1e-3, 1e-4 |
| Batch size | 16 | 4 |
| Maximum epochs | 80 | 60 |
| Early-stopping patience | 15 | 10 |
| Gradient clipping | none | 1.0 |
| Scheduler | none | none |
| Clinical input | 20 dims | 20 dims |
| EEG input | n/a | 19-channel cache (z-scored per window, clipped at 5 SD) |

HEP1 external prediction: mean of the five fold models' probabilities; threshold is the
mean of the five inner Youden thresholds.

Multi-site training (exp18): outer 5-fold StratifiedKFold on outcome x cohort (inner
folds on the same key); three arms (Melbourne only, HEP1 only, both) trained on the
same outer folds; the pooled arm uses cohort-stratified AUC for early stopping and
epoch choice; late-fusion inputs z-scored on the fit set; seeds 42-46 (EEG
configurations 42-44); a size-matched Melbourne arm for the clinical model.

REVE standalone (`thesisStandalone/analysis/reve_standalone.py`): REVE features
mean-pooled per patient (512) plus ChemBERTa (768); head 1280 -> 64 -> 64 (BatchNorm,
ReLU, Dropout 0.3) -> 2; AdamW lr 1e-3, weight decay 1e-4, batch 16, maximum 80 epochs,
patience 15, no gradient clipping.


## 6. Clinical features written as text (exp19)

| Setting | Value |
|---|---|
| Encoders (frozen, pinned revisions) | PubMedBERT, ClinicalBERT (768, token mean, 510-token windows); Llama-3.1-8B (bf16, token mean primary, last token secondary); Qwen3-Embedding-8B (secondary; last token, L2-normalised, instruction prefix) |
| Estimator `mlp` | late-fusion MLP trained with the portable non-EEG loop (Section 5 settings), refit protocol |
| Estimator `pca32` | each input z-scored, PCA to min(32, n_fit - 1) on the fit rows, then the MLP |
| Estimator `lr` | L2 logistic regression, balanced class weights, max_iter 5000; C from {0.001, 0.01, 0.1, 1, 10} by pooled inner-fold AUC (earliest on ties), Youden threshold at that C, then refit |
| Clinical input | 23 dims (all 16 variables, including for HEP1 scoring); information-matched `clinical_full` 36 dims |
| Zero-shot baseline | Llama-3.1-8B-Instruct, probability of answering "Yes", computed in fp32 |


## 7. Inconsistencies found while compiling this table

These are differences between the code, its configs and descriptions elsewhere. None
was introduced by this document; they are listed so the paper and future runs can be
checked against them.

1. The main 27-channel EEG cache has no amplitude normalisation (signals in volts); only
   the 19-channel cross-cohort caches are z-scored per window. The paper's Methods says
   recordings were "z-scored per window".
2. The "LaBraM" arm is LaBraM's architecture trained from scratch, not the pretrained
   foundation model.
3. Inner folds stratify on outcome only (outcome x cohort in exp18), while the outer
   split is multilabel.
4. `findings/architecture.md` gives about 2M parameters for exp7a and about 4.7M for exp7b
   (the paper also says 4.7M for FuseMoE); built models have 1.70M and 6.26M.
5. Learning rates and schedulers differ between families: exp1 and exp2 use
   ReduceLROnPlateau, all others a constant rate; exp1a uses patience 15, the others 20.
6. EEG aggregator dropout is 0.1 in exp2, exp3, exp7b and exp9 but 0.3 in exp5c, exp6b
   and exp7a; SimpleCNN dropout is 0.1 in exp2 and exp9 but 0.3 in exp6b.
7. FuseMoE temperature is fixed at 1.0 in exp1, exp2 and exp7b but annealed to 0.5 in
   exp3; `num_moe_layers=2` is unused in exp3 and exp7b (one FuseMoE layer); the top-2
   expert outputs in the per-modality router are summed without gate weights; the
   auxiliary loss is a negative mutual-information router regulariser, although the
   docstrings call it load balancing.
8. exp2's `TRAIN_CONFIG` batch size of 4 is overridden to 8, and its SimpleCNN embedding
   is 200, not the 256 in `EEG_EMBED_DIMS`. exp9 imports exp2's learning rate (5e-5) but
   hard-codes 1e-3.
9. The portable cross-cohort loop (80 epochs, patience 15, no clipping) differs from the
   exp4-7 loops (100 epochs, patience 20, clipping 1.0) for configurations of the same
   name.
10. With ASM weighting the loss is a plain mean of weighted per-sample losses, while the
    unweighted class-weighted loss divides by the sum of weights, so loss scales differ
    between modes.
11. Some docstrings still give the clinical input as 20 dims (exp5, exp7) or 19 (exp15);
    the effective width is 23.
12. exp19 scores HEP1 with all 23 clinical dims, including variables constant in HEP1;
    the other cross-cohort analyses drop them (20 dims).
13. Seeding differs: exp7 seeds once per process; exp15, exp16 and exp17 reseed every fold.
14. `rerun_clean.sh:298` marks exp11 and the exp9 ablations as deferred; they were later
    run with `RUN_DEFERRED=1` and their outputs exist.

Not determined: parameter counts for exp4, exp5, exp6, exp9, exp11 and the HEP1 models
(logged only at run time); the braindecode version (exact EEGNet and Labram internals);
the input units REVE expects relative to the cache.
