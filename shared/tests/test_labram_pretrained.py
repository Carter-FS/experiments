"""Tests for shared.labram_pretrained (run in .venv-reve; skipped where braindecode has
no Hub support). The pinned hub weights are read from the local Hugging Face cache."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pytest.importorskip("braindecode")
try:
    from braindecode.models import Labram
    if not hasattr(Labram, "from_pretrained"):
        pytest.skip("braindecode without Hub support", allow_module_level=True)
except ImportError:
    pytest.skip("braindecode not installed", allow_module_level=True)

from shared import eeg_cache as C
from shared import labram_pretrained as LP


@pytest.fixture(scope="module")
def hub_state():
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
    try:
        return LP.hub_state_dict()
    except (OSError, HfHubHTTPError, LocalEntryNotFoundError) as exc:  # no network and no cached weights
        pytest.skip(f"hub weights unavailable: {exc}")
    # a hash mismatch raises RuntimeError and fails the suite


def test_adapt_state_dict_slices_only_the_temporal_embedding(hub_state):
    adapted = LP.adapt_state_dict(hub_state)
    assert adapted["temporal_embedding"].shape == (1, LP.N_PATCHES + 1, LP.EMBED_DIM)
    assert torch.equal(adapted["temporal_embedding"][0], hub_state["temporal_embedding"][0, : LP.N_PATCHES + 1])
    same = [k for k in hub_state if k != "temporal_embedding"]
    assert all(torch.equal(adapted[k], hub_state[k]) for k in same)
    with pytest.raises(ValueError):
        LP.adapt_state_dict(hub_state, n_patches=hub_state["temporal_embedding"].shape[1] + 5)


def test_models_load_with_the_expected_keys_and_shapes(hub_state):
    mean = LP.build_model("mean", hub_state)
    cls = LP.build_model("cls", hub_state)
    assert sum(p.numel() for p in mean.parameters()) == LP.EXPECTED_N_PARAMS
    assert not mean.training and not cls.training
    x = torch.randn(2, LP.N_CHANNELS, LP.SAMPLES_PER_WINDOW, generator=torch.Generator().manual_seed(0)) * 0.3
    with torch.no_grad():
        fm, fc = mean(x, ch_names=list(LP.LABRAM_CH_NAMES)), cls(x, ch_names=list(LP.LABRAM_CH_NAMES))
    assert fm.shape == (2, LP.EMBED_DIM) and fc.shape == (2, LP.EMBED_DIM)
    assert not torch.allclose(fm, fc)
    # the mean feature is the parameter-free LayerNorm of the mean over the patch
    # tokens (CLS excluded) of the last block's raw output, as in the official code
    raw = {}
    handle = mean.blocks[-1].register_forward_hook(lambda m, i, o: raw.setdefault("x", o.detach()))
    with torch.no_grad():
        fm2 = mean(x, ch_names=list(LP.LABRAM_CH_NAMES))
    handle.remove()
    assert raw["x"].shape == (2, 1 + LP.N_CHANNELS * LP.N_PATCHES, LP.EMBED_DIM)
    manual = torch.nn.functional.layer_norm(raw["x"][:, 1:, :].mean(1), (LP.EMBED_DIM,), eps=mean.fc_norm.eps)
    assert torch.allclose(manual, fm2, atol=1e-5) and torch.equal(fm, fm2)
    with pytest.raises(ValueError):
        LP.build_model("max", hub_state)


def test_a_missing_pretrained_tensor_is_refused(hub_state):
    incomplete = {k: v for k, v in hub_state.items() if k != "blocks.0.attn.qkv.weight"}
    with pytest.raises(RuntimeError, match="weight mismatch"):
        LP.build_model("mean", incomplete)


def test_temporal_channels_use_the_legacy_tuh_embeddings():
    from braindecode.models.labram import _LABRAM_CANONICAL_INDEX as index
    assert len(LP.LABRAM_CH_NAMES) == 19 and len(set(LP.LABRAM_CH_NAMES)) == 19
    for cache_name, labram_name in LP.CACHE_TO_LABRAM_NAME.items():
        assert LP.LABRAM_CH_NAMES[C.CH_NAMES.index(cache_name)] == labram_name
    # the official standard_1020 list orders the legacy names T3, T5, T4, T6
    assert [index[n] for n in ("T3", "T5", "T4", "T6")] == [88, 89, 90, 91]
    assert [index[n] for n in ("T7", "T8", "P7", "P8")] == [37, 45, 59, 67]
    assert all(n.upper() in index for n in LP.LABRAM_CH_NAMES)
    assert not {"T7", "T8", "P7", "P8"} & {n.upper() for n in LP.LABRAM_CH_NAMES}


def test_canonical_list_matches_the_official_standard_1020():
    canonical = LP.canonical_channel_list()
    assert len(canonical) == 128 and canonical[:3] == ["FP1", "FPZ", "FP2"] and canonical[88:92] == ["T3", "T5", "T4", "T6"]


def test_forward_uses_the_exported_embedding_rows(hub_state):
    model = LP.build_model("mean", hub_state)
    x = torch.randn(2, LP.N_CHANNELS, LP.SAMPLES_PER_WINDOW, generator=torch.Generator().manual_seed(1)) * 0.3
    seen = {}
    original = model.forward_features

    def spy(inp, input_chans, **kwargs):
        seen["rows"] = input_chans.tolist()
        return original(inp, input_chans=input_chans, **kwargs)

    model.forward_features = spy
    with torch.no_grad():
        a = model(x, ch_names=list(LP.LABRAM_CH_NAMES))
        rows = seen["rows"]
        b = model(x, ch_names=list(reversed(LP.LABRAM_CH_NAMES)))
    assert rows == LP.input_chans() and rows[0] == 0 and rows[1 + list(C.CH_NAMES).index("T7")] == 89
    assert not torch.allclose(a, b)


def test_window_features_rejects_the_wrong_layout(hub_state):
    model = LP.build_model("mean", hub_state)
    with pytest.raises(ValueError, match="windows must be"):
        LP.window_features(model, torch.zeros(2, 27, LP.SAMPLES_PER_WINDOW))
    assert LP.window_features(model, torch.zeros(0, LP.N_CHANNELS, LP.SAMPLES_PER_WINDOW)).shape == (0, LP.EMBED_DIM)


def _tiny_cache(tmp_path, counts=(5, 0, 3)):
    rng = np.random.default_rng(0)
    recs = {}
    for i, n_valid in enumerate(counts):
        w = np.zeros((C.MAX_WINDOWS, 19, 2000), dtype=np.float32)
        m = np.ones(C.MAX_WINDOWS, dtype=bool)
        w[:n_valid] = rng.normal(0.0, 20.0, (n_valid, 19, 2000))
        m[:n_valid] = False
        recs[f"p{i}"] = {"windows_uv": w, "padding_mask": m, "ch_names": list(C.CH_NAMES), "sfreq": 200.0,
                         "signal_start_s": 0.0, "duration_s": 600.0, "version": 2, "source_sha256": None}
    path = tmp_path / "eeg19_v2_alfred.pkl"
    C.write_cache(path, {"version": C.CACHE_VERSION, "cohort": "alfred"}, recs)
    return path


def test_extract_features_layout_and_determinism(tmp_path, hub_state):
    cache = _tiny_cache(tmp_path)
    out = tmp_path / "labram_features_v2_alfred.npz"
    meta = LP.extract_features(cache, out, device="cpu", batch_size=4, state=hub_state)
    d = np.load(out)
    assert d["features"].shape == (3, C.MAX_WINDOWS, LP.EMBED_DIM) and d["features_cls"].shape == d["features"].shape
    assert d["pids"].tolist() == ["p0", "p1", "p2"] and d["valid_window_counts"].tolist() == [5, 0, 3]
    f = d["features"]
    assert np.isfinite(f).all() and not f[0, 5:].any() and not f[1].any() and f[2, :3].all()
    assert meta["n_recordings"] == 3 and meta["n_recordings_without_windows"] == 1
    assert meta["embed_dim"] == LP.EMBED_DIM and meta["input_convention"] == "labram"
    assert meta["source_cache_meta"]["version"] == C.CACHE_VERSION and meta["source_cache_meta"]["n_recordings"] == 3
    assert meta["braindecode_version"] and meta["torch_version"]
    stored = json.loads(d["meta"].item())
    assert stored["hub_revision"] == LP.HUB_REVISION and stored["labram_ch_names"] == list(LP.LABRAM_CH_NAMES)
    # the stored feature equals a direct forward on the labram-convention windows
    w, m = C.load_cache(cache, "labram")["p0"]
    direct = LP.window_features(LP.build_model("mean", hub_state), torch.from_numpy(w[:5]))
    assert np.allclose(direct.numpy(), f[0, :5], atol=1e-5)
    direct_cls = LP.window_features(LP.build_model("cls", hub_state), torch.from_numpy(w[:5]))
    assert np.allclose(direct_cls.numpy(), d["features_cls"][0, :5], atol=1e-5)
    # second run with another batch size is identical
    out2 = tmp_path / "again.npz"
    LP.extract_features(cache, out2, device="cpu", batch_size=2, state=hub_state)
    assert np.array_equal(np.load(out2)["features"], f)
    side = json.loads((tmp_path / "labram_features_v2_alfred.npz.meta.json").read_text())
    assert side["hub_safetensors_sha256"] == LP.HUB_SAFETENSORS_SHA256


def test_extract_refuses_padding_before_valid_windows(tmp_path, hub_state):
    cache = _tiny_cache(tmp_path, counts=(4,))
    payload = C._read_payload(cache, allow_legacy=False)
    rec = payload["recordings"]["p0"]
    rec["padding_mask"][1] = True              # a hole inside the valid prefix
    rec["windows_uv"][1] = 0.0
    C.write_cache(cache, payload["meta"], payload["recordings"])
    with pytest.raises(ValueError, match="padded windows before valid ones"):
        LP.extract_features(cache, tmp_path / "out.npz", state=hub_state)


def test_extract_refuses_a_legacy_cache(tmp_path, hub_state):
    import pickle
    legacy = tmp_path / "processed_eeg.pkl"
    with open(legacy, "wb") as fh:
        pickle.dump({"1": (np.zeros((120, 27, 2000), np.float32), np.ones(120, bool))}, fh)
    with pytest.raises(ValueError, match="not a version-2"):
        LP.extract_features(legacy, tmp_path / "out.npz", state=hub_state)


def test_export_weights_round_trip(tmp_path, hub_state):
    path = LP.export_weights(tmp_path / "labram_base_19ch.pt", hub_state)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["hub_revision"] == LP.HUB_REVISION and saved["n_patches"] == LP.N_PATCHES
    assert saved["model_kwargs"]["n_chans"] == 19 and saved["model_kwargs"]["n_times"] == 2000
    assert saved["model_kwargs"] == LP.MODEL_KWARGS and saved["input_convention"] == "labram"
    assert saved["braindecode_version"] and saved["input_chans"] == LP.input_chans()
    assert saved["labram_ch_names"] == list(LP.LABRAM_CH_NAMES) and saved["cache_ch_names"] == list(C.CH_NAMES)
    assert saved["state_dict"]["temporal_embedding"].shape == (1, LP.N_PATCHES + 1, LP.EMBED_DIM)
    assert set(saved["state_dict"]) == set(hub_state)
    model = LP.build_model("cls", saved["state_dict"])
    assert torch.equal(model.temporal_embedding, saved["state_dict"]["temporal_embedding"])
    # the constructor arguments rebuild the architecture and the state dict loads strictly
    rebuilt = Labram(**saved["model_kwargs"], use_mean_pooling=False,
                     chs_info=[{"ch_name": c} for c in saved["labram_ch_names"]])
    rebuilt.load_state_dict(saved["state_dict"], strict=True)
    assert sum(p.numel() for p in rebuilt.parameters()) == LP.EXPECTED_N_PARAMS
    # the embedding rows follow the official get_input_chans convention
    chans = saved["input_chans"]
    assert chans[0] == 0 and len(chans) == 20
    assert chans[1 + list(C.CH_NAMES).index("T7")] == 89 and chans[1 + list(C.CH_NAMES).index("FP1")] == 1
    assert saved["pooling"]["mean"]["drop_keys"] == ["norm.weight", "norm.bias"]


def test_official_name_map():
    assert LP.official_name("position_embedding") == "student.pos_embed"
    assert LP.official_name("temporal_embedding") == "student.time_embed"
    assert LP.official_name("patch_embed.temporal_conv.conv1.weight") == "student.patch_embed.conv1.weight"
    assert LP.official_name("blocks.11.mlp.0.weight") == "student.blocks.11.mlp.fc1.weight"
    assert LP.official_name("blocks.3.mlp.2.bias") == "student.blocks.3.mlp.fc2.bias"
    assert LP.official_name("blocks.0.attn.qkv.weight") == "student.blocks.0.attn.qkv.weight"
    assert LP.official_name("cls_token") == "student.cls_token" and LP.official_name("norm.weight") == "student.norm.weight"


def test_hub_weights_match_the_official_checkpoint(hub_state):
    if not LP.OFFICIAL_CHECKPOINT_PATH.exists():
        pytest.skip(f"official checkpoint not at {LP.OFFICIAL_CHECKPOINT_PATH}")
    result = LP.verify_against_official(LP.OFFICIAL_CHECKPOINT_PATH, hub_state)
    assert result["identical"] == result["hub_tensors"] == len(hub_state) == 221
    assert result["checkpoint_sha256"] == LP.OFFICIAL_CHECKPOINT_SHA256
    assert result["unused_official"] == ["lm_head.bias", "lm_head.weight", "logit_scale", "projection_head.0.bias",
                                         "projection_head.0.weight", "student.lm_head.bias", "student.lm_head.weight",
                                         "student.mask_token"]
    # a changed value is rejected
    perturbed = dict(hub_state)
    perturbed["blocks.0.attn.qkv.weight"] = hub_state["blocks.0.attn.qkv.weight"] + 1e-3
    with pytest.raises(RuntimeError, match="differ from the official checkpoint"):
        LP.verify_against_official(LP.OFFICIAL_CHECKPOINT_PATH, perturbed)
    # two blocks swapped (identical shapes) are rejected
    swapped = dict(hub_state)
    for k in hub_state:
        if k.startswith("blocks.0."):
            other = "blocks.1." + k[len("blocks.0."):]
            swapped[k], swapped[other] = hub_state[other], hub_state[k]
    with pytest.raises(RuntimeError, match="differ from the official checkpoint"):
        LP.verify_against_official(LP.OFFICIAL_CHECKPOINT_PATH, swapped)
    # a tensor under an unknown name is rejected
    renamed = dict(hub_state)
    renamed["blocks.0.attn.extra.weight"] = renamed.pop("blocks.0.attn.proj.weight")
    with pytest.raises(RuntimeError, match="no official counterpart"):
        LP.verify_against_official(LP.OFFICIAL_CHECKPOINT_PATH, renamed)
    # a hub state missing an encoder tensor leaves an official encoder tensor unmatched
    short = {k: v for k, v in hub_state.items() if k != "blocks.0.attn.proj.weight"}
    with pytest.raises(RuntimeError, match="no hub counterpart"):
        LP.verify_against_official(LP.OFFICIAL_CHECKPOINT_PATH, short)
    # a dtype change is rejected even when the values round-trip
    halved = dict(hub_state)
    halved["cls_token"] = hub_state["cls_token"].half()
    with pytest.raises(RuntimeError, match="differ from the official checkpoint"):
        LP.verify_against_official(LP.OFFICIAL_CHECKPOINT_PATH, halved)
