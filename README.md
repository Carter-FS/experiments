# experiments

Multimodal fusion experiments for predicting anti-seizure medication (ASM) treatment outcomes. They combine clinical features with embeddings from EEG reports, EEG signals and drug molecular structures (SMILES). This is the experiment code for my Honours research at Monash University. Patient data is not included.

**Status:** Active

## Experiments

| # | Folder | What it tests | Notes |
| --- | --- | --- | --- |
| 1 | `exp1_fusion` | Report text + SMILES | [exp1](findings/exp1_notes.md) |
| 2 | `exp2_fusion` | EEG signal + SMILES | [exp2](findings/exp2_notes.md) |
| 3 | `exp3_fusion` | Text + EEG + SMILES, MLP and FuseMoE fusion | [exp3](findings/exp3_notes.md) |
| 4 | `exp4_baseline` | Clinical features only (baseline) | [exp4](findings/exp4_notes.md) |
| 5 | `exp5_clinical_fusion` | Clinical + one modality | [exp5](findings/exp5_notes.md) |
| 6 | `exp6_clinical_triple` | Clinical + SMILES + text or EEG | [exp6](findings/exp6_notes.md) |
| 7 | `exp7_all_modalities` | All four modalities | [exp7](findings/exp7_notes.md) |
| 8 | `exp8_stratification` | Stratification analysis | [exp8](findings/exp8_notes.md) |
| 9 | `exp9_eeg_investigation` | EEG variance ablations | [exp9](findings/exp9_notes.md) |
| 10 | `exp10_direct_llm` | LLM run at training time instead of frozen embeddings | [exp10](findings/exp10_notes.md) |
| 11 | `exp11_eeg_upgrade` | EEG2Vec 128D with aggregator variants | [exp11](findings/exp11_notes.md) |
| 12 | `exp12_moe_hparam` | FuseMoE hyperparameters | [exp12](findings/exp12_notes.md) |
| 13 | `exp13_qwen_finetune` | Qwen 2.5 0.5B fine-tuning | |
| 14 | `exp14_optuna_tuning` | Optuna hyperparameter search | [exp14](findings/exp14_notes.md) |
| 15 | `exp15_reve_quad_mlp` | Quad-modal fusion with REVE EEG embeddings | |
| 16 | `exp16_reduced_capacity` | Reduced-capacity quad-modal model | |
| 17 | `exp17_focal_only` | Quad-modal model on the focal subset | |
| 18 | `exp18_mixed_cohort` | Mixed-cohort training, per-cohort evaluation | |
| 19 | `exp19_serialised_clinical` | Clinical features as serialised text vs tabular | |

Cross-experiment results are in [findings/experiment_findings.md](findings/experiment_findings.md).

## Requirements

- Python 3.10 and [uv](https://docs.astral.sh/uv/)
- A CUDA GPU with 8 GB or more of VRAM for the full experiments
- About 10 GB of disk space for dependencies

## Install

The project uses two virtual environments, because MoLeR needs TensorFlow and everything else uses PyTorch.

```sh
git clone --recurse-submodules https://github.com/Carter-FS/experiments.git
cd experiments

# Main environment (PyTorch), used by every experiment
uv venv --python 3.10 .venv-others
source .venv-others/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu118   # or cu121, or plain torch for CPU
uv pip install transformers scikit-learn numpy pandas mne
uv pip install braindecode==1.2.0 --no-deps   # see docs/troubleshooting.md

# MoLeR environment (TensorFlow), only for MoLeR SMILES embeddings
uv venv --python 3.10 .venv-moler
source .venv-moler/bin/activate
uv pip install rdkit "tensorflow<2.10" numpy molecule-generation
```

Installing braindecode normally replaces the CUDA 11.8 build of torch with a CUDA 12 one. [docs/troubleshooting.md](docs/troubleshooting.md) has the full fix, along with GPU, memory and EDF loading issues.

## Data

The data is not included, for privacy and ethics reasons. The code expects it next to the repo:

```
../asm_data/
├── alfred_1st_regimen.csv   # one row per patient: pid, outcome, ASM, eeg_report, ...
└── Alfred/EEG/*.edf         # EEG recordings
```

`outcome` is 1 for success and 2 for failure. `shared/cohort.py` maps it to 1/0.

## Usage

Each experiment is a module with a `run_experiments.py` entry point:

```sh
source .venv-others/bin/activate
python -m exp1_fusion.run_experiments --dry-run    # list the configurations
python -m exp1_fusion.run_experiments              # run them with 5-fold CV
python -m exp5_clinical_fusion.run_experiments --exp 5c
python -m exp3_fusion.run_experiments --fusion fusemoe
```

Run a module with `--help` for its options. Results are written as JSON under `outputs/` (for example `outputs/exp1_results/`), with the mean, standard deviation and per-fold values of accuracy, AUC and F1.

On the Monash M3 cluster, `submit_job.sh` runs an experiment as a SLURM job. Check the `#SBATCH` lines at the top, preview with `DRY_RUN=true bash submit_job.sh`, then submit with `EXPERIMENT=exp2 sbatch submit_job.sh`. Logs go to `logs/asm_<jobid>.out`.

## Reproducing the results table

An earlier version of the results table mixed runs from different code versions and had a data leak: a few patients appeared twice in the CSV and could land in both the train and test folds. `shared/cohort.py` now builds every cohort:

- It deduplicates by `pid` before every CV split. If two rows disagree on the outcome, the patient is dropped (pid 954). If they disagree only on features, the first row is kept and the conflict is logged (pid N009).
- It uses one outcome mapping and one SMILES resolver for all experiments.
- The corrected cohort sizes are clinical 198, text 117, EEG 147, text + EEG 107 and quad-modal 107.

To regenerate every out-of-fold prediction file and check it:

```sh
bash rerun_all_oof.sh               # archive old predictions, rerun unweighted and weighted, then verify
bash rerun_all_oof.sh --skip-exp11  # skip the slow EEG2Vec configuration
```

`python -m shared.verify_oof outputs` checks that every file is deduplicated, that no patient spans two folds, and that each cohort has the expected size. Unit tests for the cohort code run with `pytest shared/tests/`.

Older notes on prediction logging and ASM-balanced training are in [docs/STAGE_A_README.md](docs/STAGE_A_README.md) and [docs/STAGE_B_README.md](docs/STAGE_B_README.md).

## Repo layout

```
exp*/                 # one folder per experiment (see the table above)
exp1_misc/            # scripts that pre-compute text and SMILES embeddings
shared/               # cohort building, OOF verification and shared helpers
findings/             # per-experiment notes, architecture docs and overall findings
docs/                 # troubleshooting, analysis plans and older rerun notes
smiles-transformer/   # upstream SMILES Transformer code (git submodule)
MoLeR_checkpoint/, smiles_transformer/   # pre-trained SMILES model weights
```

## Licence

Copyright (c) 2025 Carter Facey-Smith. All rights reserved.

This code is shared for reference as part of Honours research at Monash University. Please [contact me](mailto:carterfaceysmith@gmail.com) before reusing it. Third-party components, such as the pre-trained MoLeR checkpoint and the SMILES Transformer submodule, remain under their original licences.
