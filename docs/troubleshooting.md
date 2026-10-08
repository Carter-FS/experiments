# Troubleshooting

## PyTorch Not Using GPU

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

## CUDA Out of Memory

Reduce batch size in the config files:
- `exp1_fusion/config.py` - `TRAIN_CONFIG["batch_size"]`
- `exp2_fusion/config.py` - `BATCH_SIZE_BY_ENCODER`

## EEG Loading Errors

The pipeline automatically handles encoding issues in EDF files by trying multiple encodings (UTF-8, Latin-1).

## braindecode Breaks PyTorch CUDA (libcudart.so.12 / libcudnn.so.9)

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

## Missing Dependencies

If you encounter import errors, ensure you've installed all packages:

```bash
source .venv-others/bin/activate
uv pip install torch transformers scikit-learn numpy pandas mne braindecode
```
