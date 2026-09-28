# ASM Outcome Prediction Experiments

Multimodal fusion experiments for predicting anti-seizure medication (ASM) treatment outcomes by combining embeddings from clinical text reports, EEG signals, and drug molecular structures (SMILES).

## Consistent OOF rerun (data-leakage fix)

The published results table was assembled from runs on inconsistent code/data
vintages and was affected by a **duplicate-patient leakage bug**: a handful of
pids appeared twice in the CSV and landed in different CV folds, putting the
same patient in train and test. `shared/cohort.py` is now the single source of
truth that fixes this and unifies the previously-copied logic:

- **Dedupe by pid before every CV split.** Fuller row wins; on an
  outcome-label conflict the pid is **dropped** (pid 954 only -> clinical
  cohorts 199->198); on a feature-only conflict keep-first and log it (pid
  N009, stays in the quad). Every `data_pipeline.py` calls `dedupe_by_pid`
  (exp1 applies the same mask to its row-aligned embedding matrix).
- **One outcome map** (`{1:0, 2:1}`) and **one SMILES resolver**
  (`smiles_vector`, keep-with-mean-fallback; the 15-drug index covers 100% of
  the cohort's ASMs, so no patient is dropped for SMILES).
- **Corrected cohorts:** clinical 198, text 117, EEG 147, text+EEG 107,
  quad 107.
- **Four bug fixes:** exp4 distinct per-config prediction filenames; exp5
  `asm_drugs` guard so `--asm-balance none` no longer crashes; exp5/exp7
  per-config try/except; exp3 run without the `--fusion mlp` filter so exp3b
  is produced.
- **`--asm-balance weighted`** (inverse-sqrt sample weighting) wired into every
  table experiment so the unbalanced and weighted columns are both complete.

Regenerate the whole prediction set and gate it in one command:

```bash
bash rerun_all_oof.sh              # archive old preds, rerun none+weighted, gate
bash rerun_all_oof.sh --skip-exp11 # skip the slow EEG2Vec encoder config
```

The gate is `python -m shared.verify_oof outputs`: it asserts every OOF file is
deduped, has no pid spanning two folds, and matches the expected cohort count.
Unit tests: `pytest shared/tests/`.

## Prerequisites

- Python 3.10
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-capable GPU (recommended, 8GB+ VRAM for full experiments)
- ~10GB disk space for dependencies

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ThousandEyedLibrarian/experiments
cd experiments

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Environment Setup

This project uses **two separate virtual environments** due to TensorFlow/PyTorch dependency conflicts.

### Main Environment (PyTorch-based)

Used for all experiments (Exp 1-7):

```bash
uv venv --python 3.10 .venv-others
source .venv-others/bin/activate

# For GPU support (recommended) - check CUDA version with: nvidia-smi
# CUDA 11.8:
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
# uv pip install torch

# Install remaining dependencies
uv pip install transformers scikit-learn numpy pandas mne braindecode
```

### MoLeR Environment (TensorFlow-based)

Only needed for MoLeR molecular embeddings:

```bash
uv venv --python 3.10 .venv-moler
source .venv-moler/bin/activate
uv pip install "rdkit" "tensorflow<2.10" numpy molecule-generation
```

## Data Setup

Data is not included in this repository for privacy reasons. The expected data structure:

```
../asm_data/
├── alfred_1st_regimen.csv      # Patient metadata with outcomes
└── Alfred/
    └── EEG/
        └── *.edf               # EEG recordings (157 files)
```

The CSV should contain columns: `pid`, `outcome` (1 = success, 2 = failure; mapped to 1/0 by `shared/cohort.py`), `ASM`, `eeg_report`, etc.

## Running Experiments

### HPC Setup (GPU Nodes)

On HPC systems, experiments must run on GPU compute nodes, not login nodes.

**Using the submission script (recommended):**

A pre-configured SLURM script `submit_job.sh` is included for M3/Monash HPC:

```bash
# 1. Edit configuration in submit_job.sh:
#    - EXPERIMENT="exp1" or "exp2"
#    - EXTRA_ARGS for additional flags
#    - SBATCH directives (time, memory, GPU type)

# 2. Test without running (dry-run mode):
DRY_RUN=true bash submit_job.sh

# 3. Submit the job:
sbatch submit_job.sh

# 4. Monitor:
squeue -u $USER

# 5. Check logs:
cat logs/asm_<jobid>.out
```

**Script configuration options:**
- Default: 4 hours, 32GB RAM, 1 GPU (any type)
- For specific GPU: edit `#SBATCH --gres=gpu:A100:1` or `--gres=gpu:A40:1`
- For email alerts: uncomment `--mail-user` and `--mail-type` lines

**Interactive session (alternative):**
```bash
srun --gres=gpu:1 --partition=gpu --mem=32G --time=4:00:00 --pty bash
source .venv-others/bin/activate
python -m exp1_fusion.run_experiments
```

### Experiment 1: LLM + SMILES Fusion

Combines clinical text embeddings with drug molecular embeddings.

```bash
source .venv-others/bin/activate

# Preview what experiments will run
python -m exp1_fusion.run_experiments --dry-run

# Run all Experiment 1 combinations (8 experiments, 5-fold CV each)
python -m exp1_fusion.run_experiments
```

### Experiment 2: EEG + SMILES Fusion

Combines EEG signal embeddings with drug molecular embeddings.

```bash
source .venv-others/bin/activate

# Preview experiments
python -m exp2_fusion.run_experiments --dry-run

# Run with SimpleCNN encoder (works on 8GB GPU)
python -m exp2_fusion.run_experiments --eeg-encoder simplecnn

# Run with specific SMILES model
python -m exp2_fusion.run_experiments --eeg-encoder simplecnn --smiles-model chemberta
```

### Experiment 3: Triple Modality Fusion (LLM + EEG + SMILES)

Combines all three modalities: clinical text, EEG signals, and drug molecular embeddings.

```bash
source .venv-others/bin/activate

# Run all Experiment 3 combinations (8 experiments, 5-fold CV each)
python -m exp3_fusion.run_experiments

# Run only MLP fusion experiments (exp3a)
python -m exp3_fusion.run_experiments --fusion mlp

# Run only FuseMoE experiments (exp3b)
python -m exp3_fusion.run_experiments --fusion fusemoe

# Run a specific configuration
python -m exp3_fusion.run_experiments --experiments exp3a_clinicalbert_chemberta
```

**Experiment matrix (8 configurations):**

| ID | Text Encoder | SMILES Encoder | Fusion |
|----|--------------|----------------|--------|
| exp3a-1 | ClinicalBERT | ChemBERTa | MLP |
| exp3a-2 | ClinicalBERT | SMILES-Trf | MLP |
| exp3a-3 | PubMedBERT | ChemBERTa | MLP |
| exp3a-4 | PubMedBERT | SMILES-Trf | MLP |
| exp3b-1 | ClinicalBERT | ChemBERTa | FuseMoE |
| exp3b-2 | ClinicalBERT | SMILES-Trf | FuseMoE |
| exp3b-3 | PubMedBERT | ChemBERTa | FuseMoE |
| exp3b-4 | PubMedBERT | SMILES-Trf | FuseMoE |

### Experiment 4: Clinical Features Baseline

Uses only clinical features (demographics, medical history) as a baseline.

```bash
source .venv-others/bin/activate
python -m exp4_baseline.run_experiments
```

### Experiment 5: Clinical + Single Modality

Combines clinical features with one embedding modality.

```bash
source .venv-others/bin/activate

# Run all Exp5 variants
python -m exp5_clinical_fusion.run_experiments

# Run specific variant
python -m exp5_clinical_fusion.run_experiments --exp 5a  # Clinical + SMILES
python -m exp5_clinical_fusion.run_experiments --exp 5b  # Clinical + Text
python -m exp5_clinical_fusion.run_experiments --exp 5c  # Clinical + EEG
```

### Experiment 6: Clinical + SMILES + Third Modality

Combines clinical features, SMILES embeddings, and either text or EEG.

```bash
source .venv-others/bin/activate

# Run all Exp6 variants
python -m exp6_clinical_triple.run_experiments

# Run specific variant
python -m exp6_clinical_triple.run_experiments --exp 6a  # Clinical + SMILES + Text
python -m exp6_clinical_triple.run_experiments --exp 6b  # Clinical + SMILES + EEG
```

### Experiment 7: All Four Modalities

Combines all modalities: Clinical + Text + EEG + SMILES.

```bash
source .venv-others/bin/activate

# Run all Exp7 variants
python -m exp7_all_modalities.run_experiments

# Run specific variant
python -m exp7_all_modalities.run_experiments --exp 7a  # MLP fusion
python -m exp7_all_modalities.run_experiments --exp 7b  # FuseMoE
```

### Embedding Generation (Optional)

Generate embeddings separately using scripts in `exp1_misc/`:

```bash
source .venv-others/bin/activate

# Generate ChemBERTa SMILES embeddings
python exp1_misc/e1_LLM+SMILES_ChemBERTa.py

# Generate SMILES Transformer embeddings
python exp1_misc/e1_LLM+SMILES_SMILESTransformer.py

# Generate text embeddings with ClinicalBERT/PubMedBERT
python exp1_misc/e1_LLM+SMILES_Bert.py
python exp1_misc/e1_LLM+SMILES_PubMedBert.py
```

## Output Structure

```
outputs/
├── chemberta_asm_embeddings.npy    # SMILES embeddings (ChemBERTa)
├── smilestrf_asm_embeddings.npy    # SMILES embeddings (SMILES Transformer)
├── exp1_results/                    # Experiment 1 results
│   ├── summary.json                # All experiments summary
│   └── exp1a_*.json               # Individual experiment results
├── exp2_results/                    # Experiment 2 results
│   ├── summary.json
│   └── exp2_*.json
├── exp3_results/                    # Experiment 3 results (triple modality)
│   └── exp3_results_*.json        # Results with timestamp
└── eeg_cache/                       # Cached preprocessed EEG data
    └── processed_eeg.pkl
```

### Results Format

Each experiment JSON contains:
```json
{
  "experiment": "exp1b_clinicalbert_chemberta",
  "accuracy": {"mean": 0.61, "std": 0.08, "per_fold": [...]},
  "auc": {"mean": 0.65, "std": 0.08, "per_fold": [...]},
  "f1": {"mean": 0.62, "std": 0.13, "per_fold": [...]}
}
```

## Results

High-level findings and analysis are in the `findings/` folder.

## Troubleshooting

### PyTorch Not Using GPU

If experiments show "Device: cpu" instead of "cuda":

**1. On HPC systems: You're likely on a login node**

Login nodes don't have GPUs. Request a GPU compute node:

```bash
# Interactive GPU session (adjust for your HPC system)
# For SLURM-based systems:
srun --gres=gpu:1 --partition=gpu --time=2:00:00 --pty bash

# Or using smux (M3/Monash):
smux new-session --gres=gpu:1 --partition=gpu --mem=32G --time=2:00:00

# Verify GPU is accessible:
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

**2. PyTorch installed without CUDA support**

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False on a GPU node, reinstall PyTorch with CUDA:
source .venv-others/bin/activate
uv pip uninstall torch
uv pip install torch --index-url https://download.pytorch.org/whl/cu118  # or cu121
```

### CUDA Out of Memory

Reduce batch size in the config files:
- `exp1_fusion/config.py` - `TRAIN_CONFIG["batch_size"]`
- `exp2_fusion/config.py` - `BATCH_SIZE_BY_ENCODER`

### EEG Loading Errors

The pipeline automatically handles encoding issues in EDF files by trying multiple encodings (UTF-8, Latin-1).

### braindecode Breaks PyTorch CUDA (libcudart.so.12 / libcudnn.so.9)

Running `uv pip install braindecode` resolves `torch` against PyPI's default index, which ships the **CUDA 12 variant**. This silently replaces your cu118 torch with the cu12 version and pulls in `nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`, etc. - causing `libcudart.so.12: cannot open shared object file` errors on CUDA 11.8 systems. Removing those nvidia packages then breaks torch entirely (`libcudnn.so.9` missing).

**Fix - install braindecode without touching torch:**

```bash
source .venv-others/bin/activate

# 1. Reinstall torch from the cu118 index (bundles its own CUDA libs)
uv pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall

# 2. Install braindecode without dependency resolution
uv pip install braindecode==1.2.0 --no-deps

# 3. Install braindecode's non-torch dependencies manually
uv pip install "mne>=1.10.0" "pandas<3.0.0" h5py "skorch>=1.2.0" joblib torchinfo einops docstring-inheritance

# 4. Remove any leftover nvidia-cu12 packages
uv pip uninstall nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 \
  nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 2>/dev/null

# 5. Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import braindecode; print(braindecode.__version__)"
```

**Note:** braindecode 1.2.0 is the latest version supporting Python 3.10. Versions 1.3.0+ require Python 3.11.

### Missing Dependencies

If you encounter import errors, ensure you've installed all packages:

```bash
source .venv-others/bin/activate
uv pip install torch transformers scikit-learn numpy pandas mne braindecode
```

## Project Structure

```
experiments/
├── exp1_fusion/          # Experiment 1: LLM + SMILES fusion
│   ├── run_experiments.py
│   ├── training.py
│   └── models/
├── exp2_fusion/          # Experiment 2: EEG + SMILES fusion
│   ├── run_experiments.py
│   ├── eeg_pipeline.py
│   └── models/
├── exp3_fusion/          # Experiment 3: LLM + EEG + SMILES fusion
│   ├── run_experiments.py
│   ├── data_pipeline.py
│   ├── training.py
│   └── models/
│       ├── triple_mlp.py      # ~2.5M params
│       └── triple_fusemoe.py  # ~4.7M params
├── exp4_baseline/        # Experiment 4: Clinical features baseline
│   ├── run_experiments.py
│   └── data_pipeline.py
├── exp5_clinical_fusion/ # Experiment 5: Clinical + single modality
│   ├── run_experiments.py
│   ├── models.py
│   └── training.py
├── exp6_clinical_triple/ # Experiment 6: Clinical + SMILES + third modality
│   ├── run_experiments.py
│   ├── models.py
│   └── training.py
├── exp7_all_modalities/  # Experiment 7: All four modalities
│   ├── run_experiments.py
│   ├── models.py
│   └── training.py
├── exp1_misc/            # Embedding generation scripts
├── outputs/              # Generated embeddings and results
├── findings/             # Analysis and architecture docs
├── smiles-transformer/   # External SMILES transformer code
└── MoLeR_checkpoint/     # Pre-trained MoLeR model
```
