# Plan: EEG input fix (channels and normalisation) and pretrained LaBraM

Status: draft for audit, 2026-10-09. Nothing in this plan has been run. Every EEG
result in the paper is superseded once this plan is executed; no non-EEG result changes.

Author of the defects: the EEG cache and encoder code written for Stage A
(January-February 2026). Found during the configuration audit of 2026-10-09
(findings/training_configurations.md, Section 7).


## 0. Summary

Three defects, two of which the paper misdescribes:

1. The EEG fed to every main-table EEG model is the wrong set of channels. The
   27-channel cache keeps every channel the EDF header types as EEG, which on the
   Alfred recordings includes EMG+, EMG-, PG1, PG2, ECG+, ECG- and the ear references
   A1, A2. Eight of the 27 "EEG channels" are not scalp EEG, and the ECG channels are
   recorded in mV (about 1000 times the EEG amplitude).
2. The same cache is never amplitude-normalised: signals are in volts (median
   per-window SD 1.07e-5). Every from-scratch encoder starts with a BatchNorm whose
   epsilon (1e-5, or 1e-3 in braindecode EEGNet) is five orders of magnitude larger than
   the signal variance (1e-10), so the first normalisation layer attenuates the signal
   by 540x (EEG2Vec, SimpleCNN) to 6000x (EEGNet) instead of standardising it. The paper
   says recordings were "z-scored per window"; they were not.
3. The "LaBraM" arm is the LaBraM architecture with 2 layers, 4 heads and a 128-dim
   embedding, initialised at random and trained from scratch on 27 channels. No
   pretrained weights were ever loaded. The paper presents it as a foundation model.

Defects 1 and 2 together explain why every from-scratch EEG encoder sits at or below
chance and why the full model does not beat the clinical model. They do not touch the
clinical, text or drug pipelines.

The fix: rebuild the EEG cache on the 19 standard 10-20 channels in microvolts with a
per-window z-score normalisation in microvolts (the supervisor's convention); route every EEG consumer through one loader that
refuses the old cache; add the real pretrained LaBraM (weights verified identical to the
official checkpoint) as a frozen-feature encoder, with an optional fine-tuned arm; realign
REVE with its own normalisation convention; pre-register all of this as Addendum C;
rerun every EEG-dependent item (about 100-120 GPU-hours for the primary set, about 1 to
1.5 days of wall-clock on the 4-GPU cap).


## 1. Evidence

All numbers below were measured on 2026-10-09; the scripts are reproducible from the
cited files. No patient identifiers are involved.

### 1.1 Channel content of the 27-channel cache

- `exp2_fusion/eeg_pipeline.py:620-626` (legacy path): `raw.pick_types(eeg=True,
  exclude=[])`, keeping every channel the EDF header types as EEG, in header order.
- All 142 Alfred EDF files share one channel order at 250 Hz, with header units mV and
  uV: `EMG+ PG1 Fp1 Fp2 PG2 EMG- F7 F3 Fz F4 F8 A1 T7 C3 Cz C4 T8 A2 P7 P3 Pz P4 P8 ECG+
  O1 O2 ECG-` (read from the headers with MNE, aggregate only).
- `outputs/eeg_cache/processed_eeg.pkl` (built 2026-01-16; 147 recordings; arrays
  (120, 27, 2000)) therefore holds 19 scalp channels plus EMG, PG (eye), ECG and ear
  references. The depthwise spatial convolution in EEG2Vec/EEGNet and the first
  BatchNorm (which normalises across all channels jointly) see the ECG channels as the
  dominant input.
- The 19-channel path (`filter_to_standard_19`, `eeg_pipeline.py:541-567`, added
  2026-05-21 for Stage C) renames T3/T4/T5/T6, picks the 19 standard channels in
  canonical order and drops everything else. The docstring itself says it "drops
  EMG/PG/A/ECG/sop".

### 1.2 Amplitude scale of the 27-channel cache

- `EEGPreprocessor(normalisation="none")` is the default (`eeg_pipeline.py:779`), and
  the cache builder passes no normalisation (`exp2_fusion/data_pipeline.py:160-163`).
  The `window_zscore` option was added on 2026-02-03 (exp9), after the cache was built,
  and is used only by the Stage C caches.
- Measured on the cache (first 40 recordings): median per-window per-channel SD
  1.07e-5 (volts), max |x| 0.207 V (an artefact 20 000 times the median SD), minimum SD
  1.2e-11 (flat channel).
- For comparison the 19-channel caches (`processed_eeg_std19_alfred.pkl`, 148
  recordings, 2026-05-21; `processed_eeg_std19_hep.pkl`, 95, 2026-07-20) are z-scored
  per window per channel with amplitude clipping at 5 SD
  (`thesisStandalone/analysis/hep_eeg_preprocess.py:66-71`): median SD 1.00.

### 1.3 Effect inside the encoders (synthetic test, random initialisation)

EEG-like noise of unit SD scaled to 1e-5 (volts) versus 1.0, through each encoder as
built by `exp2_fusion/models/eeg_encoders.py`:

| Encoder | BN1 input SD at volt scale | BN1 output SD at volt scale | BN1 output SD at unit scale | Attenuation |
|---|---|---|---|---|
| EEG2Vec (BatchNorm2d, eps 1e-5) | 5.7e-6 | 0.002 | 1.000 | 556x |
| SimpleCNN (BatchNorm1d, eps 1e-5) | 2.2e-2 | 0.002 | 1.000 | 540x |
| EEGNet (braindecode, eps 1e-3) | 5.1e-6 | 0.0002 | 0.998 | 6168x |

BatchNorm computes (x - mean) / sqrt(var + eps). With var about 1e-10 and eps 1e-5
the denominator is sqrt(eps), so the layer scales the signal by about 1/316 instead of
standardising it, and the downstream weights and the learned BatchNorm affine terms must
recover a signal 500 to 6000 times smaller than designed, under weight decay. Together
with the ECG channels, this is sufficient to explain chance-level EEG discrimination.

### 1.4 "LaBraM" arm

- `exp2_fusion/models/eeg_encoders.py:46-100`: `braindecode.models.Labram(n_chans=27,
  n_times=2000, emb_size=128, n_layers=2, att_num_heads=4, neural_tokenizer=True)`,
  no `load_state_dict`, no pretrained path. braindecode 1.2.0 in `.venv-others` has no
  `from_pretrained`. `exp9_eeg_investigation/run_experiments.py:400-403` sets
  `embed_dim: 128` "for memory".
- Official LaBraM (Jiang et al., ICLR 2024; github.com/935963004/LaBraM, MIT licence,
  copyright 2024 Weibang Jiang): `labram_base_patch200_200` = patch 200, embed 200,
  depth 12, 10 heads, MLP ratio 4, qk LayerNorm; checkpoint
  `checkpoints/labram-base.pth` (96.6 MB; sha256
  7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c).
- braindecode 1.6.0.dev1024 (`.venv-reve`) ships `Labram.from_pretrained(
  "braindecode/labram-pretrained")` (5,819,936 parameters; safetensors sha256
  53b752edb366fd6395dd3cb7d63ae3e1e16aabed040aa0901d64ef32e8f444f8, snapshot
  0563b6c626e7b40d9a36653b763715db94d945d7). Verified on 2026-10-09: all 221 hub tensors
  are numerically identical (allclose, atol 1e-6) to the `student.*` tensors of the
  official checkpoint. The hub weights are the official weights.
- Official input convention (`engine_for_finetuning.py`, train and evaluate):
  `samples.float() / 100` then `rearrange(samples, 'B N (A T) -> B N A T', T=200)`, i.e.
  microvolts divided by 100, 1-second patches; preprocessing (README): 0.1-75 Hz
  bandpass, 50 Hz notch, 200 Hz, unit uV. Channel embeddings are selected by 10-20
  name (`utils.get_input_chans` over `standard_1020`); braindecode does the same through
  `forward(x, ch_names=...)`. Verified: a 19-channel model built from the hub weights
  runs on (B, 19, 2000) input, and permuting the channel names changes the output, so the
  per-channel embeddings are applied by name. The only adjustment needed is slicing the
  temporal embedding from 16 to n_patches + 1 = 11 rows, which is how both the official
  code (`time_embed[:, 0:input_time_window]`) and braindecode index it.
- Official downstream pooling: `use_mean_pooling=True` by default, i.e. the mean over
  patch tokens through a fresh `fc_norm` LayerNorm (the pretrained `norm` is used only
  on the CLS path). The hub model is configured with `use_mean_pooling=False`.
- Fine-tuning recipe (official README, TUAB): AdamW, lr 5e-4, weight decay 0.05,
  layer decay 0.65, 50 epochs, 5 warm-up epochs, cosine schedule, drop path 0.1, batch
  64 windows, label smoothing 0.1 for multi-class (BCE for binary), `--abs_pos_emb
  --disable_rel_pos_bias --disable_qkv_bias`. The README notes that learning rate and
  warm-up "strongly affect results".

### 1.5 REVE

- `thesisStandalone/analysis/reve_extract_features.py` runs REVE-base
  (`brain-bzh/reve-base`, 19 channels, attention pooling, final layer stripped) on the
  per-window z-scored, 5-SD-clipped 19-channel cache.
- REVE's own convention (arXiv 2510.21585; braindecode REVE docstring): resample to
  200 Hz, band-pass 0.5-99.5 Hz, "Z-score normalization with statistics computed across
  the recording sessions", clipping at 15 standard deviations; the docstring warns that
  "users should apply similar preprocessing". The model does not normalise internally
  (the only normalisation in its forward pass is RMSNorm on tokens).
- Our per-window z-score differs from REVE's per-session statistics (it removes
  within-recording amplitude dynamics and whitens artefact windows). The effect is
  probably small, but the rebuild should align it.

### 1.6 What the paper currently says that is wrong

- Methods, EEG paragraph: "Recordings were resampled to 200 Hz on a 27-channel montage
  ... and z-scored per window." The montage includes non-EEG channels; nothing was
  z-scored.
- Methods, encoder list, S7 and the encoder table: LaBraM described alongside REVE as a
  pretrained foundation model.
- Discussion: "The REVE foundation model was the only EEG encoder above 0.5".
- findings/training_configurations.md Section 1 records the facts correctly (no
  normalisation; LaBraM from scratch) but not the channel content.


## 2. Decisions

Decisions D3 and D4 were taken on 2026-10-09 and are recorded in place.

D1. Montage: the 19 standard 10-20 channels for every EEG experiment, Melbourne and
    HEP1 alike. The 27-channel cache is retired. Rationale: it is the only channel set
    that is scalp-only, canonically ordered, shared by both cohorts and usable by the
    pretrained encoders. Consequence: the EEG cohort becomes the 148 recordings the
    19-channel pipeline accepts (the 27-channel pipeline accepted 147), so the EEG rows'
    n and the quad cohort (107 -> probably 108, as in exp15) change. The exact counts
    are computed by the new cache builder and frozen in Addendum C before any run.

D2. Storage: the new cache stores the 19-channel signal in microvolts, unnormalised
    (float32, (120, 19, 2000) with the padding mask, as now). Normalisation is applied by the loader, so conventions can be switched without
    rebuilding.

D3. Normalisation conventions (applied by the loader), following the supervisor's
    benchmark pipeline (code-fury/eeg-foundation-model,
    `benchmark/preprocessing/preprocess_multichannel.py`, `zscore_norm_epoch`, lines
    416-420: per-epoch, per-channel z-score over the time axis in microvolts, floor 1e-6,
    no clipping; comparison in findings/training_configurations.md, Section 1a).
    - `zscore_window` (from-scratch encoders and REVE): per 10-second window, per
      channel, (x - mean) / max(sd, 1e-6) over the window's 2000 samples, in microvolts,
      no clipping. The Stage C caches used the same z-score after clipping the raw
      signal at 5 SD; the clip is not applied.
    - `labram` (pretrained LaBraM): microvolts divided by 100, no z-score, because the
      pretrained weights expect that scale (official `engine_for_finetuning.py`); no
      clipping.
    - A window whose standard deviation is below the floor in a channel becomes zero
      in that channel; whole flat channels are counted in the sidecar.
    - Leading flat segment (supervisor's `find_signal_start`): the 300 s skip starts
      from the first 100 s chunk whose standard deviation exceeds the flat threshold.
      The number of recordings affected is reported in the sidecar.
    Statistics come from the window itself, never from other patients.

D4. LaBraM arms:
    - L1 (primary replacement for the "LaBraM" row): frozen pretrained LaBraM-base as a
      per-window feature extractor, mirroring the REVE arm. Features: the mean over the
      190 patch tokens of the last block, followed by a parameter-free LayerNorm (the
      official fine-tuning default, `use_mean_pooling=True` with a fresh `fc_norm`);
      the CLS-token feature (pretrained `norm`, `x[:, 0]`) is stored alongside for
      transparency but not analysed. 200 dims per window. Extraction in fp32 on CPU
      (about 2 minutes for all windows) for bit-reproducibility.
    - L2 (secondary, exploratory, pre-registered): full fine-tuning of LaBraM-base inside
      the exp9 harness with the official recipe adapted to patient-level training:
      AdamW, peak lr 5e-4 with 5 warm-up epochs then cosine, layer decay 0.65, weight
      decay 0.05, drop path 0.1, batch of 1 patient (windows chunked), gradient
      accumulation to 8 patients, at most 50 epochs, epoch chosen by the refit protocol
      as for every other model. About 80-100 GPU-hours for 5 seeds. Runs after the
      frozen-feature (L1) results are in.
    - The from-scratch LaBraM-architecture arm is rerun on the corrected inputs and
      kept in the encoder comparison, relabelled "LaBraM architecture, random
      initialisation (2 layers)".

D5. Fusion models: EEG2Vec trained from scratch stays the pre-specified EEG branch of
    the main table (no change to the pre-registered encoder set). Two feature-based
    full-model variants are added beside exp15 (REVE features): exp15-style quad with
    LaBraM L1 features. Both are reported with exp15 in the Supplementary.

D6. REVE: features re-extracted from the new cache under `zscore_window` (same code path,
    `reve_features_v2_*.npz`). exp15, the REVE encoder row, exp18 and the HEP EEG scripts
    all move to the new caches and features.

D7. Scope of the rerun (both protocols, so Table 2 and the pre-registered S10 table stay
    consistent): every item that reads an EEG cache or REVE features. Listed in Section 5.

D8. Everything else is frozen: outer and inner CV, refit protocol, seeds, hyperparameters,
    clinical/text/drug pipelines, analysis code. No non-EEG prediction file is touched.


## 3. Implementation

Work in the public `experiments` repo; no patient data in git. Each step is a gate:
after it is implemented, two fresh independent reviewers (10 minutes each) check that
it fully resolves its problem and that nothing is unimplemented or lazily done; the next
step starts only when both pass. The paper's EEG-dependent
statements are marked `\tbc` until all results are in.

### S1. `shared/eeg_cache.py` (new)

- `build_cache(cohort, edf_index, out_path, notch_hz)`: for each recording, the
  existing functions of `exp2_fusion/eeg_pipeline.py` (EDF reading, the 19-channel
  filter, resampling to 200 Hz, the 0.1-75 Hz bandpass and the notch) plus Duong's
  leading-flat detection, then
  the 300 s skip (after the leading-flat detection of D3), the 1200 s segment, the
  600 s minimum and the 10 s windows. Convert volts to microvolts (x 1e6) after
  `extract_time_window`, then window. Entry per patient:
  `{"windows_uv": float32 (120, 19, 2000), "padding_mask": bool (120,), "ch_names":
  STD_19, "sfreq": 200, "signal_start_s": float,
  "version": 2, "source_sha256": <sha256 of the EDF bytes>}`. Also write a sidecar
  `*.meta.json` with aggregate statistics only (n recordings, median SD in uV, n flat
  channels, n rejected and why).
  Output: `outputs/eeg_cache/eeg19_v2_alfred.pkl`, `outputs/eeg_cache/eeg19_v2_hep.pkl`
  (HEP1 at 60 Hz notch, as now).
- `load_cache(path, convention)` with `convention in {"zscore_window", "labram", "raw_uv"}`
  returning the same `{pid: (windows, padding_mask)}` structure every consumer expects
  today, normalised as in D3, plus `cache_info(path)` for `n_channels`, `ch_names`,
  cohort counts. It raises on a file without `version == 2` unless
  `allow_legacy=True`, which no production path sets.
- A short `python -m shared.eeg_cache build --cohort alfred|hep` CLI, and `stats` to
  print the sidecar.

### S2. Route every consumer through the loader

Replace direct pickle loads and hard-coded channel counts:

| Consumer | Today | Change |
|---|---|---|
| `exp2_fusion/data_pipeline.py:235` `preprocess_all_eeg` | builds/loads `processed_eeg.pkl` | `load_cache(ALFRED_V2, "zscore_window")` |
| `exp3_fusion/data_pipeline.py:292` | same | same |
| `exp5_clinical_fusion/data_pipeline.py:144`, `exp6_clinical_triple/...:144`, `exp7_all_modalities/...:126` `load_eeg_data` | pickle of `EEG_CACHE_PATH` | `load_cache` |
| `exp5/exp6/exp7/exp9/exp8 config.py` `EEG_CACHE_PATH`, `"n_channels": 27` | 27-channel | point at the v2 path; `n_channels` from `cache_info` |
| `exp9_eeg_investigation/run_experiments.py:236` `get_max_channels` | derived from data | keep (will give 19) |
| `exp11`, `exp16`, `exp17` | via exp7/exp9 pipelines | inherit |
| `exp18_mixed_cohort/config.py:79-80` | std19 caches | v2 caches, `zscore_window` |
| `shared/portable_models.py:268 load_eeg_cache`, `N_CHANNELS = 19` | std19 | `load_cache`; keep 19 |
| `thesisStandalone/analysis/hep_external_validation_eeg.py:74-75`, `hep_reduced_external_validation.py` | std19 | v2 |
| `thesisStandalone/analysis/reve_extract_features.py` | std19 | v2 with `zscore_window`; output `reve_features_v2_*.npz` |
| `exp15_reve_quad_mlp/config.py:22` | `reve_features_alfred.npz` | `--feature-set reve_v2|labram_v2` |
| Defaults `n_channels: int = 27` in `eeg_encoders.py`, `fusion.py`, `triple_mlp.py`, `triple_fusemoe.py`, `exp5/6/7 models.py`, `exp7/data_pipeline.py max_channels` | 27 | 19 (and always passed explicitly from `cache_info`) |

`hep_eeg_preprocess.py` is superseded by `shared.eeg_cache build`; keep it with a
deprecation note pointing to the new builder (or delete it; the caches it built are
archived, not deleted).

### S3. `shared/labram_pretrained.py` (new; runs in `.venv-reve`)

- Loads `Labram.from_pretrained("braindecode/labram-pretrained")`, asserts the
  safetensors sha256 above, and (once) asserts tensor-for-tensor equality with the
  official `labram-base.pth` (sha256 above; downloaded from the official repository,
  kept outside git) so the provenance check is part of the code, not a note.
- Builds `Labram(n_chans=19, n_times=2000, sfreq=200, n_outputs=0,
  chs_info=[{"ch_name": c} for c in STD_19], use_mean_pooling=False)`, loads the hub
  state dict with `temporal_embedding` sliced to the model's `n_patches + 1` rows,
  strict except the head, and runs `forward(x, ch_names=STD_19, return_all_tokens=...)`
  to obtain the token tensor; computes the mean-pooled patch feature with a
  parameter-free LayerNorm (D4) and the CLS feature.
- `extract(cache_path, out_npz)`: convention `labram` from `load_cache`; fp32 on CPU;
  writes `outputs/labram_features_v2_{alfred,hep}.npz` in the exp15 format
  (`features_mean (n, 120, 200)`, `features_cls`, `pids`, `valid_window_counts`,
  `ch_names`, `hub_snapshot`, `sha256`).
- Also saves the adapted 19-channel state dict to `outputs/labram_base_19ch.pt` for S4.

### S4. Pretrained LaBraM inside the training environment (for arm L2 and the encoder comparison)

`.venv-others` (torch 2.9.1, braindecode 1.2.0) cannot load the hub weights.
Vendor braindecode 1.6.0.dev1024's `models/labram.py` (BSD-3-Clause; keep the notice
and the upstream commit in the file header) as `shared/vendor/labram.py` with its two
small helpers, so the exact architecture runs in `.venv-others`. A unit test built
once in `.venv-reve` records the output of the hub model on a fixed random input
(`shared/tests/fixtures/labram_19ch_reference.npz`); the vendored model must reproduce
it to 1e-5. `exp2_fusion/models/eeg_encoders.py` gains `encoder_type="labram_pretrained"`
(vendored model + `outputs/labram_base_19ch.pt`, input convention `labram`, trainable
or frozen) and `encoder_type="precomputed"` (identity over (B, W, D) features), and
`get_eeg_encoder` stops defaulting to `n_channels=27`.

### S5. exp9 encoder comparison and feature consumers

- `exp9` arms on the v2 cache: `baseline_simplecnn_transformer`, `encoder_eegnet`,
  `encoder_eeg2vec`, `encoder_labram_scratch` (renamed from `encoder_labram`),
  `encoder_labram_pretrained_frozen` (precomputed L1 features, aggregator and head as
  the other arms), `encoder_reve_frozen` (precomputed REVE v2 features; replaces the
  separate `reve_standalone` row so every encoder shares one aggregator and head), and
  `encoder_labram_finetune` (L2). The `precomputed` encoder reads the npz through the
  exp15 loader generalised to a `feature_set` argument.
- Each precomputed arm declares its own input convention; the raw-EEG arms read
  `zscore_window`. One cache, one loader, one place to change.
- `exp15` gains `--feature-set {reve_v2, labram_v2}`.

### S6. Tests (`shared/tests/test_eeg_cache.py`, `test_labram_pretrained.py`)

- Builder: on a synthetic 25-channel RawArray exported to EDF (MNE `export_raw`), the
  cache has exactly `STD_19` in order, non-EEG channels are absent, units are
  microvolts (SD of a 20 uV synthetic sine is 20 within 5%), a leading flat segment
  is skipped, a flat channel is counted.
- Loader: `zscore_window` gives mean 0 and SD 1 per window and channel; `labram` equals
  x / 100; a version-1 pickle raises; `cache_info` reports 19 channels.
- Real-cache aggregate check (not in CI, run once and recorded in the sidecar): median
  per-window SD in microvolts between 3 and 60 for both cohorts; count of flat
  channels.
- LaBraM: hub sha256 matches; the 19-channel model loads with only the head missing;
  channel permutation changes the output; the vendored model matches the reference
  fixture; the npz pids and valid counts match the cache masks exactly.
- Consumers: each EEG entry point runs one fold for two epochs on the v2 cache
  (`rerun_clean.sh smoke` extended to every EEG task), and `shared/verify_oof.py`
  expected counts are updated to the frozen Addendum C numbers.

### S7. Addendum C (`docs/analysis_plan_clean_rerun_exp18.md`, append-only, committed before any run)

Records: Sections 1 and 2 of this plan in short form; the new expected cohort sizes per
configuration (from the sidecar); the list of superseded prediction files; the new arms
(L1, L2, LaBraM quad, REVE v2) and their status (L1 and REVE v2 primary within the
encoder comparison; L2 and the LaBraM quad exploratory); that the normalisation
conventions were chosen before any result under them was seen; that no non-EEG
result changes.

### S8. Audits before pushing

Two independent read-only subagent audits (as for Addendum B): (a) code end to end
(builder, loader, every consumer, feature extraction, vendored model), (b) Addendum C
against the implementation and against the official LaBraM and REVE conventions.
Fix, re-audit, then push both repositories.

### S9. M3 run (commands handed to Carter; I do not submit)

1. `git pull` on M3; `uv pip install` nothing new in `.venv-others` (vendored model
   needs no new package); copy `outputs/labram_features_v2_*.npz`,
   `reve_features_v2_*.npz` and `outputs/labram_base_19ch.pt` from the laptop
   (`rsync`, as for the REVE features).
2. Build the two caches on M3 (CPU, about 1-2 h; `python -m shared.eeg_cache build
   --cohort alfred` and `--cohort hep`) and print `stats`; compare the sidecar with the
   laptop build (counts and medians must match).
3. Archive, do not delete: move every EEG-dependent prediction and results file to
   `outputs/_archive_eeg27_2026-10-xx/` (list generated by a new
   `bash rerun_clean.sh list-eeg-outputs`), and the old caches to the same folder.
4. `bash rerun_clean.sh preflight` (extended to check the v2 caches and feature files),
   then the new array. Done markers move to `outputs/_clean_rerun_v3/<protocol>/` so
   old markers cannot mask the rerun.
5. `bash rerun_clean.sh verify` with the new expected counts; sync to the laptop;
   regenerate tables and figures.

### S10. Paper, after the results

Methods EEG paragraph (19 channels, microvolts, per-window z-score normalisation and
clipping, LaBraM-base pretrained and frozen, REVE convention); Figures 1 and 2
(`figures/tikz`: EEG row text); encoder table and S7/S8; Results and Discussion EEG
statements; Supplementary note on the superseded 27-channel results;
`findings/training_configurations.md` Section 1 and 4.


## 4. Order of work and critical path

| Step | Depends on | Effort |
|---|---|---|
| Carter confirms D3, D4 (L2 timing), D4 (keep scratch arm) | - | 10 min |
| S1 builder + loader + tests; build the Alfred and HEP caches on the laptop (CPU) | - | 0.5 day; cache build about 1-2 h |
| S3 feature extraction (LaBraM L1, REVE v2) on the laptop CPU | S1 | 1 h |
| S2 consumers, S4 vendoring, S5 arms, smoke runs | S1, S3 | 0.5-1 day |
| S7 Addendum C with the frozen counts | S1 sidecar | 1 h |
| S8 two audits, fixes, push | all above | 2-3 h |
| S9 phase A on M3 (Section 5) | S8 | about 1-1.5 days wall-clock |
| S9 phase B | A running | about 1.5 days |
| S9 phase C (L2 fine-tuning) if confirmed | A | about 1 day |
| Analysis regeneration and S10 | A | 0.5 day |


## 5. Rerun scope and cost

Per-item medians below are the completed items of the September-October arrays on M3
(`sacct`, GPU partition, both protocols pooled; items run both balance modes where the
driver does). Items not depending on EEG are not rerun.

| Item (per seed) | Median h | Notes |
|---|---|---|
| exp2 | 1.1 | EEG + SMILES, MLP and MoE, SimpleCNN and EEG2Vec |
| exp3 | 1.3 | text + EEG + SMILES |
| exp5 (only 5c needed) | 0.6 | add a `--only 5c` filter to `exp5_clinical_fusion.run_experiments`, as exp2 has `--eeg-encoder` |
| exp6 (only 6b needed) | 0.8 | same filter for 6b |
| exp7a, exp7b, exp7a_stratbatch | 2.2, 0.7, 0.3 | |
| exp15 (features; CPU) | 0.1 | plus the LaBraM-feature quad |
| exp16, exp17 | 1.5, 0.2 | |
| exp9 encoders (simplecnn, eegnet, eeg2vec, labram_scratch, frozen) | 0.1, 0.3, 0.3, 1.5, 0.1 | plus L1 and REVE v2 precomputed arms (CPU, about 0.1 each) |
| hep_eeg, hep_eeg_h12, hep_reduced | 1.4, 1.4, 0.4 | |
| exp18 Exp5c/Exp6b/Exp7a (+h12), seeds 42-44 | 1.7, 1.7, 1.1 each | |

Phase A (primary, refit protocol): about 13 GPU-hours per seed x 5 seeds, plus about
27 GPU-hours for exp18, about 95-120 GPU-hours in total. On the 4-GPU-per-user cap that
is about 1-1.5 days of wall-clock, before queue time.

Phase B (secondary): exp9 aggregator and embedding ablations (about 8 h per seed, 40 h),
the exp11 grid (about 14 h per seed, 70 h), and the inner-split protocol for every EEG
item (about 25 h): about 135 GPU-hours, about 1.5 days.

Phase C (if confirmed): L2 fine-tuning, 80-100 GPU-hours, about 1 day.

Everything in the clinical, text and drug pipelines, and every non-EEG prediction
file, stays as it is.


## 6. Risks and open points

- The EEG cohort changes from 147 to 148 (and the quad from 107 to about 108). The
  analysis plan's expected counts, `verify_oof.py` and the paper's n values change; the
  patient who differs is identified by the builder log (aggregate only) and the reason
  recorded in Addendum C.
- HEP1 EDF units: the builder must read the header units per channel (as MNE does) so
  the microvolt conversion is correct for both cohorts; the sidecar medians are the
  check (3-60 uV expected).
- The hub LaBraM weights are a dev build of braindecode (1.6.0.dev1024). The snapshot
  hash and the official checkpoint hash are pinned in code; vendoring the model file
  removes the dependency on that build at training time.
- Pooling for L1 (mean over patch tokens with a parameter-free LayerNorm) follows the
  official fine-tuning default, but no official frozen-feature protocol exists; the CLS
  feature is stored so the choice can be checked later without re-extraction.
- L2 memory: 120 windows x 191 tokens x 200 dims x 12 layers at batch 1 patient is a
  few GB of activations; chunking must keep the whole patient in the graph (no
  checkpointing across chunks) or use gradient checkpointing.
- Per-window z-scoring differs from REVE's own per-session z-score clipped at 15. A
  per-session convention is a one-line addition to the loader.
- The Stage C builder matched EDF files by pid prefix (`f"{pid}*.edf"`), which can
  assign another patient's file to a short pid; the new builder matches the exact id
  parsed from the file name (`extract_patient_id`), and the sidecar reports the counts.
- Artefact windows are not masked; a window-level artefact flag is a possible later
  addition and is not pre-registered here.
- The driver's EEG items also train non-EEG configurations inside the same experiment
  modules (exp2 trains text-free models only, fine; exp5 and exp6 train 5a/5b/6a as well),
  hence the `--only` filters, which must leave the existing non-EEG prediction files
  untouched (the new array must not overwrite them).
- Timeline: implementation and audits about two working days; phase A results about
  two days after submission.


## 7. Reproducibility notes

- Measurements in Section 1 used `.venv-others` (torch 2.9.1, braindecode 1.2.0) for the
  encoders and the cache statistics, and `.venv-reve` (torch 2.12.0, braindecode
  1.6.0.dev1024, huggingface_hub 1.15.0, mne 1.12.1) for the hub and official
  checkpoints. The official checkpoint was downloaded from
  `https://raw.githubusercontent.com/935963004/LaBraM/main/checkpoints/labram-base.pth`.
- Sources: LaBraM README and `engine_for_finetuning.py`, `utils.py`,
  `modeling_finetune.py` (github.com/935963004/LaBraM, MIT); braindecode Labram
  documentation and source; REVE paper arXiv 2510.21585 and the braindecode REVE
  docstring; the `brain-bzh/reve-base` model card (REVE Responsible Use License v1.0).
