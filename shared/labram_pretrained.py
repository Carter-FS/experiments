"""Pretrained LaBraM-base as a frozen per-window EEG feature extractor.

Runs in ``.venv-reve`` (braindecode with Hugging Face Hub support). The weights are
``braindecode/labram-pretrained`` at a pinned snapshot, verified by file hash and,
optionally, tensor for tensor against the official ``labram-base.pth``.

    python -m shared.labram_pretrained extract --cohort alfred         # outputs/labram_features_v2_alfred.npz
    python -m shared.labram_pretrained extract --cohort hep
    python -m shared.labram_pretrained verify-official [CHECKPOINT]    # default outputs/labram/labram-base.pth
    python -m shared.labram_pretrained export-weights                  # outputs/labram_base_19ch.pt

Input convention: the version-2 EEG cache loaded with ``convention="labram"``
(microvolts divided by 100, as in the official ``engine_for_finetuning.py``), 19 standard
10-20 channels whose embeddings the model selects by name (the four temporal channels
under their legacy TUH names T3/T4/T5/T6, as in the official clinical fine-tuning),
10-second windows at 200 Hz (ten 1-second patches). Two features per window, both
200-dimensional and computed in fp32 with no dropout:

- ``features`` (primary): the mean over the 190 patch tokens of the last block followed
  by a parameter-free LayerNorm, which is ``use_mean_pooling=True`` with a freshly
  initialised ``fc_norm``, the official fine-tuning default.
- ``features_cls``: the ``[CLS]`` token after the pretrained final LayerNorm
  (``use_mean_pooling=False``).

The npz layout matches the REVE features used by exp15: ``features (n, 120, 200)``,
``features_cls``, ``pids``, ``valid_window_counts``, plus a JSON ``meta`` field that is
also written beside the npz as ``<name>.npz.meta.json``. Padded windows hold zeros.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from exp2_fusion.config import MAX_WINDOWS
from shared.eeg_cache import CACHE_PATHS, CH_NAMES, N_CHANNELS, SAMPLES_PER_WINDOW, TARGET_SFREQ, cache_info, load_cache
from shared.paths import EXPERIMENTS_ROOT

logger = logging.getLogger(__name__)

HUB_REPO = "braindecode/labram-pretrained"
HUB_REVISION = "0563b6c626e7b40d9a36653b763715db94d945d7"
HUB_SAFETENSORS_SHA256 = "53b752edb366fd6395dd3cb7d63ae3e1e16aabed040aa0901d64ef32e8f444f8"
OFFICIAL_CHECKPOINT_SHA256 = "7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c"
OFFICIAL_URL = "https://raw.githubusercontent.com/935963004/LaBraM/main/checkpoints/labram-base.pth"
EMBED_DIM = 200
PATCH_SIZE = 200
N_PATCHES = SAMPLES_PER_WINDOW // PATCH_SIZE
INPUT_CONVENTION = "labram"
# LaBraM-base without a classification head: the hub snapshot holds 5,819,936 parameters
# with a 16-row temporal embedding; sliced to 10 patches + [CLS] it is 1,000 fewer.
EXPECTED_N_PARAMS = 5_818_936
# Channel names given to the model. The cache stores the modern 10-20 names; LaBraM
# keeps separate embeddings for the legacy temporal names (T3/T4/T5/T6, canonical
# indices 88-91) and the modern ones (T7/T8/P7/P8, indices 37/45/59/67). The official
# clinical fine-tuning runs (TUAB and TUEV in run_class_finetuning.py) name the TUH
# channels T3/T4/T5/T6, so those embeddings are used for routine clinical EEG here.
# Names are matched case-insensitively against braindecode's canonical list, whose 128
# entries equal the first 128 of the official ``standard_1020`` list position for position.
CACHE_TO_LABRAM_NAME = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
LABRAM_CH_NAMES: Tuple[str, ...] = tuple(CACHE_TO_LABRAM_NAME.get(c, c) for c in CH_NAMES)

# Constructor arguments of the 19-channel LaBraM-base (braindecode ``Labram``); the
# export carries the same dict so a vendored copy rebuilds the identical architecture.
MODEL_KWARGS = {
    "n_chans": N_CHANNELS, "n_times": SAMPLES_PER_WINDOW, "sfreq": TARGET_SFREQ, "n_outputs": 0,
    "patch_size": PATCH_SIZE, "embed_dim": EMBED_DIM, "num_layers": 12, "num_heads": 10, "mlp_ratio": 4.0,
    "qkv_bias": False, "init_values": 0.1, "conv_in_channels": 1, "conv_out_channels": 8,
    "use_abs_pos_emb": True, "neural_tokenizer": True, "learned_patcher": False,
    "drop_prob": 0.0, "attn_drop_prob": 0.0, "drop_path_prob": 0.0,
}
# Official checkpoint tensors that belong to pretraining only (tokenizer targets and the
# symmetric-masking projection); every other official tensor must match a hub tensor.
OFFICIAL_PRETRAINING_ONLY = ("lm_head.", "logit_scale", "projection_head.", "student.lm_head.", "student.mask_token")
# sha256 of the first 128 names of the official ``standard_1020`` list (utils.py), upper
# case and comma-joined; the 129-row position embedding never reaches the 8 bipolar names
# that follow them. braindecode's canonical list must hash to the same value.
OFFICIAL_CANONICAL_128_SHA256 = "25c40292a1bcf0cd7ffafa4d57261a06a0f06214726ee8c63025a057787949e1"

OUT_DIR = EXPERIMENTS_ROOT / "outputs"
FEATURE_PATHS = {"alfred": OUT_DIR / "labram_features_v2_alfred.npz", "hep": OUT_DIR / "labram_features_v2_hep.npz"}
WEIGHTS_19CH_PATH = OUT_DIR / "labram_base_19ch.pt"
# The official checkpoint is kept outside git (outputs/ is ignored); override with
# $LABRAM_OFFICIAL_CHECKPOINT.
OFFICIAL_CHECKPOINT_PATH = Path(os.environ.get("LABRAM_OFFICIAL_CHECKPOINT") or OUT_DIR / "labram" / "labram-base.pth")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hub_state_dict() -> Dict[str, torch.Tensor]:
    """The pinned hub weights, after checking the safetensors file hash."""
    from braindecode.models import Labram
    from huggingface_hub import hf_hub_download

    weights = Path(hf_hub_download(HUB_REPO, "model.safetensors", revision=HUB_REVISION))
    digest = _sha256(weights)
    if digest != HUB_SAFETENSORS_SHA256:
        raise RuntimeError(f"{weights} has sha256 {digest}, expected {HUB_SAFETENSORS_SHA256}")
    model = Labram.from_pretrained(HUB_REPO, revision=HUB_REVISION)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def adapt_state_dict(state: Dict[str, torch.Tensor], n_patches: int = N_PATCHES) -> Dict[str, torch.Tensor]:
    """Slice the temporal embedding to ``n_patches + 1`` rows (the official code indexes
    ``time_embed[:, 0:n_patches]``); every other tensor is unchanged."""
    out = dict(state)
    te = out["temporal_embedding"]
    if te.shape[1] < n_patches + 1:
        raise ValueError(f"temporal embedding has {te.shape[1]} rows, need {n_patches + 1}")
    out["temporal_embedding"] = te[:, : n_patches + 1, :].clone()
    return out


def build_model(pooling: str, state: Optional[Dict[str, torch.Tensor]] = None):
    """A 19-channel LaBraM-base with the pretrained weights and no classification head.

    ``pooling="mean"``: mean over patch tokens through a fresh LayerNorm (the official
    fine-tuning default). ``pooling="cls"``: the [CLS] token after the pretrained norm.
    Every pretrained tensor must load; only the pooling-specific LayerNorm may differ.
    """
    from braindecode.models import Labram

    if pooling not in ("mean", "cls"):
        raise ValueError(f"pooling must be 'mean' or 'cls', not {pooling!r}")
    state = adapt_state_dict(hub_state_dict() if state is None else state)
    with warnings.catch_warnings():
        # braindecode notes that 19 channels are not its 128-channel layout; the
        # channel names are passed to every forward call, which is the supported path
        warnings.filterwarnings("ignore", message="Labram chs_info does not match")
        model = Labram(**MODEL_KWARGS, use_mean_pooling=(pooling == "mean"),
                       chs_info=[{"ch_name": c} for c in LABRAM_CH_NAMES])
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"fc_norm.weight", "fc_norm.bias"} if pooling == "mean" else set()
    allowed_unexpected = {"norm.weight", "norm.bias"} if pooling == "mean" else set()
    if set(missing) != allowed_missing or set(unexpected) != allowed_unexpected:
        raise RuntimeError(f"unexpected weight mismatch: missing {missing}, unexpected {unexpected}")
    if pooling == "mean":
        # a fresh LayerNorm is parameter-free until trained: weight 1, bias 0
        if not (torch.all(model.fc_norm.weight == 1) and torch.all(model.fc_norm.bias == 0)):
            raise RuntimeError("fc_norm is not at its initial values")
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != EXPECTED_N_PARAMS:
        raise RuntimeError(f"model has {n_params} parameters; expected {EXPECTED_N_PARAMS} (LaBraM-base, no head)")
    return model.eval()


@torch.no_grad()
def window_features(model, windows: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    """(n_windows, 19, 2000) in the LaBraM convention -> (n_windows, 200), fp32."""
    if windows.ndim != 3 or windows.shape[1:] != (N_CHANNELS, SAMPLES_PER_WINDOW):
        raise ValueError(f"windows must be (n, {N_CHANNELS}, {SAMPLES_PER_WINDOW}), got {tuple(windows.shape)}")
    device = next(model.parameters()).device
    out = []
    for start in range(0, windows.shape[0], batch_size):
        batch = windows[start : start + batch_size].to(device=device, dtype=torch.float32)
        out.append(model(batch, ch_names=list(LABRAM_CH_NAMES)).float().cpu())
    return torch.cat(out) if out else torch.zeros((0, EMBED_DIM))


def canonical_channel_list() -> list:
    """braindecode's canonical channel list, checked against the first 128 names of the
    official ``standard_1020`` list by hash."""
    from braindecode.models.labram import LABRAM_CHANNEL_ORDER

    canonical = [n.upper() for n in LABRAM_CHANNEL_ORDER]
    digest = hashlib.sha256(",".join(canonical).encode()).hexdigest()
    if digest != OFFICIAL_CANONICAL_128_SHA256:
        raise RuntimeError("braindecode's LABRAM_CHANNEL_ORDER differs from the official standard_1020 list")
    return canonical


def input_chans() -> list:
    """Rows of the pretrained position embedding for ``LABRAM_CH_NAMES``: the official
    ``utils.get_input_chans`` convention (0 for [CLS], then index in ``standard_1020`` + 1),
    which braindecode's ``forward(ch_names=...)`` applies to its canonical list."""
    canonical = canonical_channel_list()
    return [0] + [canonical.index(n.upper()) + 1 for n in LABRAM_CH_NAMES]


def _source_cache_provenance(cache_path: Path) -> dict:
    """Version, cohort, counts and build time of the cache the features come from."""
    info = cache_info(cache_path)
    keep = ("version", "cohort", "n_recordings", "n_channels", "ch_names", "sfreq", "built_at",
            "experiments_commit", "n_edf_files", "n_skipped", "skipped")
    return {k: info[k] for k in keep if k in info}


def extract_features(cache_path: Path, out_path: Path, device: str = "cpu", batch_size: int = 64,
                     state: Optional[Dict[str, torch.Tensor]] = None) -> dict:
    """Per-window features for every recording in a version-2 cache; returns the metadata.

    Recordings are processed in sorted pid order; the feature rows follow ``pids``.
    """
    cache_path, out_path = Path(cache_path), Path(out_path)
    import braindecode

    eeg = load_cache(cache_path, INPUT_CONVENTION)
    pids = sorted(eeg)
    provenance = _source_cache_provenance(cache_path)
    if provenance.get("n_recordings", len(pids)) != len(pids):
        raise RuntimeError(f"{cache_path.name} metadata lists {provenance['n_recordings']} recordings, loaded {len(pids)}")
    state = hub_state_dict() if state is None else state
    model_mean = build_model("mean", state).to(device)
    model_cls = build_model("cls", state).to(device)
    features = np.zeros((len(pids), MAX_WINDOWS, EMBED_DIM), dtype=np.float32)
    features_cls = np.zeros_like(features)
    valid_counts = np.zeros(len(pids), dtype=np.int32)
    t0 = _dt.datetime.now()
    for i, pid in enumerate(pids):
        windows, padding_mask = eeg[pid]
        valid_idx = np.flatnonzero(~padding_mask)
        if not np.array_equal(valid_idx, np.arange(len(valid_idx))):
            # consumers rebuild the mask as ``j >= valid_window_counts``
            raise ValueError(f"recording {i} has padded windows before valid ones")
        valid_counts[i] = len(valid_idx)
        if not len(valid_idx):
            continue
        x = torch.from_numpy(windows[valid_idx])
        features[i, valid_idx] = window_features(model_mean, x, batch_size).numpy()
        features_cls[i, valid_idx] = window_features(model_cls, x, batch_size).numpy()
        if (i + 1) % 10 == 0 or i + 1 == len(pids):
            logger.info("%s: %d/%d recordings", cache_path.name, i + 1, len(pids))
    if not (np.isfinite(features).all() and np.isfinite(features_cls).all()):
        raise RuntimeError("non-finite feature values")
    with_windows = valid_counts > 0
    norms = [np.linalg.norm(features[i, : valid_counts[i]], axis=-1).mean() for i in np.flatnonzero(with_windows)]
    meta = {
        "source_cache": cache_path.name, "source_cache_meta": provenance,
        "n_recordings": len(pids), "n_recordings_without_windows": int((~with_windows).sum()),
        "embed_dim": EMBED_DIM, "max_windows": MAX_WINDOWS, "n_patches": N_PATCHES,
        "input_convention": INPUT_CONVENTION, "cache_ch_names": list(CH_NAMES), "labram_ch_names": list(LABRAM_CH_NAMES),
        "hub_repo": HUB_REPO, "hub_revision": HUB_REVISION, "hub_safetensors_sha256": HUB_SAFETENSORS_SHA256,
        "braindecode_version": braindecode.__version__, "torch_version": torch.__version__,
        "pooling": {"features": "mean over patch tokens + parameter-free LayerNorm", "features_cls": "[CLS] token"},
        "device": device, "dtype": "float32", "built_at": t0.isoformat(timespec="seconds"),
        "elapsed_s": round((_dt.datetime.now() - t0).total_seconds(), 1),
        "valid_windows": {"min": int(valid_counts.min()), "median": float(np.median(valid_counts)),
                          "max": int(valid_counts.max())} if len(pids) else None,
        "feature_norm_mean": float(np.mean(norms)) if norms else None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, features=features, features_cls=features_cls, pids=np.array(pids),
                        valid_window_counts=valid_counts, meta=json.dumps(meta))
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta


def official_name(hub_key: str) -> str:
    """The official ``labram-base.pth`` key holding the tensor behind a hub key."""
    if hub_key == "position_embedding":
        return "student.pos_embed"
    if hub_key == "temporal_embedding":
        return "student.time_embed"
    key = hub_key.replace("patch_embed.temporal_conv.", "patch_embed.")
    key = re.sub(r"^(blocks\.\d+\.mlp\.)0\.", r"\1fc1.", key)
    key = re.sub(r"^(blocks\.\d+\.mlp\.)2\.", r"\1fc2.", key)
    return "student." + key


def verify_against_official(checkpoint: Path = OFFICIAL_CHECKPOINT_PATH,
                            state: Optional[Dict[str, torch.Tensor]] = None) -> dict:
    """Check that the hub weights are the official encoder weights: every hub tensor is
    bit-identical to the official tensor of the corresponding name, the mapping is one to
    one, and the official tensors left over are the pretraining-only ones."""
    checkpoint = Path(checkpoint)
    digest = _sha256(checkpoint)
    if digest != OFFICIAL_CHECKPOINT_SHA256:
        raise RuntimeError(f"{checkpoint} has sha256 {digest}, expected {OFFICIAL_CHECKPOINT_SHA256}")
    official = torch.load(checkpoint, map_location="cpu", weights_only=False)
    official = official.get("model", official)
    state = hub_state_dict() if state is None else state
    mapping = {k: official_name(k) for k in state}
    missing = sorted(k for k, o in mapping.items() if o not in official)
    if missing:
        raise RuntimeError(f"{len(missing)} hub tensors have no official counterpart: {missing[:5]}")
    if len(set(mapping.values())) != len(mapping):
        raise RuntimeError("hub-to-official name mapping is not one to one")
    different = sorted(k for k, o in mapping.items()
                       if state[k].shape != official[o].shape or state[k].dtype != official[o].dtype
                       or not torch.equal(state[k], official[o]))
    if different:
        raise RuntimeError(f"{len(different)} hub tensors differ from the official checkpoint: {different[:5]}")
    unused = sorted(set(official) - set(mapping.values()))
    stray = [k for k in unused if not k.startswith(OFFICIAL_PRETRAINING_ONLY)]
    if stray:
        raise RuntimeError(f"official encoder tensors with no hub counterpart: {stray[:5]}")
    return {"hub_tensors": len(state), "identical": len(state), "unused_official": unused, "checkpoint_sha256": digest}


def export_weights(out_path: Path = WEIGHTS_19CH_PATH, state: Optional[Dict[str, torch.Tensor]] = None) -> Path:
    """The adapted 19-channel state dict plus the constructor arguments, embedding rows
    and pooling conventions a vendored copy of the architecture needs."""
    import braindecode

    adapted = adapt_state_dict(hub_state_dict() if state is None else state)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": adapted, "hub_repo": HUB_REPO, "hub_revision": HUB_REVISION,
        # the key names and constructor arguments are this braindecode version's Labram
        "braindecode_version": braindecode.__version__,
        "hub_safetensors_sha256": HUB_SAFETENSORS_SHA256, "n_patches": N_PATCHES,
        "model_kwargs": dict(MODEL_KWARGS), "input_chans": input_chans(),
        "cache_ch_names": list(CH_NAMES), "labram_ch_names": list(LABRAM_CH_NAMES),
        "input_convention": INPUT_CONVENTION,
        # with use_mean_pooling=True the pretrained final norm is unused and fc_norm starts fresh
        "pooling": {"mean": {"use_mean_pooling": True, "drop_keys": ["norm.weight", "norm.bias"],
                             "fresh_keys": ["fc_norm.weight", "fc_norm.bias"]},
                    "cls": {"use_mean_pooling": False, "drop_keys": [], "fresh_keys": []}},
    }, out_path)
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract", help="per-window features for a cohort's version-2 cache")
    e.add_argument("--cohort", choices=sorted(CACHE_PATHS), required=True)
    e.add_argument("--cache", type=Path, default=None, help="cache path (default the cohort's version-2 cache)")
    e.add_argument("--out", type=Path, default=None, help="npz path (default outputs/labram_features_v2_<cohort>.npz)")
    e.add_argument("--device", default="cpu")
    e.add_argument("--batch-size", type=int, default=64)
    v = sub.add_parser("verify-official", help="compare the hub weights with the official checkpoint")
    v.add_argument("checkpoint", type=Path, nargs="?", default=OFFICIAL_CHECKPOINT_PATH)
    x = sub.add_parser("export-weights", help="write the adapted 19-channel state dict")
    x.add_argument("--out", type=Path, default=WEIGHTS_19CH_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "extract":
        meta = extract_features(args.cache or CACHE_PATHS[args.cohort], args.out or FEATURE_PATHS[args.cohort],
                                device=args.device, batch_size=args.batch_size)
        print(json.dumps(meta, indent=2, sort_keys=True))
    elif args.command == "verify-official":
        print(json.dumps(verify_against_official(args.checkpoint)))
    else:
        print(export_weights(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
