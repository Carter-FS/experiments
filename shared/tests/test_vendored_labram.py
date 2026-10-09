"""The vendored LaBraM (shared/vendor/labram.py) reproduces the upstream model in the
training environment, and the pretrained and precomputed encoders behave as specified.

Runs in ``.venv-others`` (no braindecode Hub support needed). The reference outputs in
``fixtures/labram_19ch_reference.npz`` were recorded from the upstream model in
``.venv-reve`` (``fixtures/make_labram_reference.py``); the weights come from
``outputs/labram_base_19ch.pt`` (``python -m shared.labram_pretrained export-weights``),
so the tests that need them skip where that file is absent.
"""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from shared import labram_pretrained as LP
from shared.vendor.labram import Labram, _SignalParamsMixin

REPO = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("fixtures") / "labram_19ch_reference.npz"
# The fixture was recorded under a different torch release (2.12 against 2.9.1 here), whose
# fp32 kernels accumulate in a different order. Under the fixture's own torch the vendored
# and upstream models are bit-identical; under torch 2.9.1 the measured error is 1.4e-7 of
# scale for the mean feature and 3.9e-6 for the [CLS] feature, so each output is compared
# against the largest magnitude of its reference at 1e-5 (an element-wise relative test
# fails on entries near zero).
SCALE_TOL = 1e-5
TOL = {"rtol": 1e-4, "atol": 1e-5}  # same-environment comparisons


def _matches(actual, reference):
    actual, reference = np.asarray(actual, dtype=np.float32), np.asarray(reference, dtype=np.float32)
    err = float(np.abs(actual - reference).max())
    scale = max(1.0, float(np.abs(reference).max()))
    assert err <= SCALE_TOL * scale, f"max error {err:.3g} on a reference of scale {scale:.3g}"
needs_weights = pytest.mark.skipif(not LP.WEIGHTS_19CH_PATH.exists(), reason=f"no exported weights at {LP.WEIGHTS_19CH_PATH}")


@pytest.fixture(scope="module")
def reference():
    d = np.load(FIXTURE)
    meta = json.loads(d["meta"].item())
    return {"x": torch.from_numpy(d["x"]), "mean": d["mean"], "cls": d["cls"], "last_block": d["last_block"], "meta": meta}


@pytest.fixture(scope="module")
def payload():
    if not LP.WEIGHTS_19CH_PATH.exists():
        pytest.skip(f"no exported weights at {LP.WEIGHTS_19CH_PATH}")
    return torch.load(LP.WEIGHTS_19CH_PATH, map_location="cpu", weights_only=True)


def _build(payload, pooling):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Labram chs_info does not match")
        model = Labram(**payload["model_kwargs"], use_mean_pooling=(pooling == "mean"),
                       chs_info=[{"ch_name": c} for c in payload["labram_ch_names"]])
    conventions = payload["pooling"][pooling]
    state = {k: v for k, v in payload["state_dict"].items() if k not in conventions["drop_keys"]}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert set(missing) == set(conventions["fresh_keys"]) and not unexpected
    return model.eval()


def test_vendored_module_imports_without_braindecode_or_mne():
    code = "import sys; import shared.vendor.labram; print('braindecode' in sys.modules, 'mne' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False"


def test_vendored_header_names_the_upstream_release():
    head = (REPO / "shared" / "vendor" / "labram.py").read_text()[:4000]
    assert "braindecode 1.6.0.dev1024" in head and "BSD 3-Clause License" in head
    assert "_SignalParamsMixin" in head and "InterpolatedLaBraM" in head
    assert (REPO / "shared" / "vendor" / "LICENSE_braindecode").exists()


def test_signal_params_mixin_infers_and_checks():
    class M(_SignalParamsMixin, torch.nn.Module):
        pass

    m = M(n_chans=19, input_window_seconds=10.0, sfreq=200.0)
    assert m.n_times == 2000 and m.n_chans == 19 and m.sfreq == 200.0 and m.input_window_seconds == 10.0
    m = M(chs_info=[{"ch_name": "FP1"}, {"ch_name": "O2"}], n_times=2000, sfreq=200.0)
    assert m.n_chans == 2 and m.input_window_seconds == 10.0
    with pytest.raises(ValueError):
        M(n_chans=3, chs_info=[{"ch_name": "FP1"}])
    with pytest.raises(ValueError):
        M(n_times=1000, input_window_seconds=10.0, sfreq=200.0)
    with pytest.raises(ValueError):
        M(n_chans=19).n_times


def test_fixture_provenance_matches_the_module_constants(reference):
    meta = reference["meta"]
    assert meta["hub_revision"] == LP.HUB_REVISION and meta["hub_safetensors_sha256"] == LP.HUB_SAFETENSORS_SHA256
    assert meta["labram_ch_names"] == list(LP.LABRAM_CH_NAMES) and meta["model_kwargs"] == LP.MODEL_KWARGS
    assert reference["x"].shape == (2, LP.N_CHANNELS, LP.SAMPLES_PER_WINDOW)


@needs_weights
def test_vendored_model_reproduces_the_upstream_outputs(reference, payload):
    assert payload["hub_revision"] == reference["meta"]["hub_revision"]
    assert payload["model_kwargs"] == reference["meta"]["model_kwargs"]
    mean, cls = _build(payload, "mean"), _build(payload, "cls")
    assert sum(p.numel() for p in mean.parameters()) == LP.EXPECTED_N_PARAMS
    captured = {}
    handle = mean.blocks[-1].register_forward_hook(lambda m, i, o: captured.setdefault("x", o.detach()))
    with torch.no_grad():
        fm = mean(reference["x"], ch_names=payload["labram_ch_names"])
        fc = cls(reference["x"], ch_names=payload["labram_ch_names"])
    handle.remove()
    _matches(fm.numpy(), reference["mean"])
    _matches(fc.numpy(), reference["cls"])
    _matches(captured["x"][:1].numpy(), reference["last_block"])
    # the same rows of the position embedding as the exported input_chans
    seen = {}
    original = mean.forward_features

    def spy(inp, input_chans, **kwargs):
        seen["rows"] = input_chans.tolist()
        return original(inp, input_chans=input_chans, **kwargs)

    mean.forward_features = spy
    with torch.no_grad():
        mean(reference["x"][:1], ch_names=payload["labram_ch_names"])
    assert seen["rows"] == payload["input_chans"] == reference["meta"]["input_chans"]


@needs_weights
def test_vendored_model_is_batch_invariant_and_deterministic(reference, payload):
    model = _build(payload, "mean")
    x = reference["x"]
    with torch.no_grad():
        both = model(x, ch_names=payload["labram_ch_names"])
        one = model(x[1:], ch_names=payload["labram_ch_names"])
        again = model(x, ch_names=payload["labram_ch_names"])
    assert torch.allclose(both[1:], one, **TOL) and torch.equal(both, again)


@needs_weights
def test_pretrained_encoder_frozen_matches_the_extracted_features(reference):
    from exp2_fusion.models.eeg_encoders import PretrainedLaBraMEncoder, get_eeg_encoder

    enc = get_eeg_encoder("labram_pretrained")
    assert isinstance(enc, PretrainedLaBraMEncoder) and enc.INPUT_CONVENTION == "labram" and enc.frozen
    assert enc.provenance["hub_revision"] == LP.HUB_REVISION and enc.input_chans == reference["meta"]["input_chans"]
    assert not any(p.requires_grad for p in enc.parameters())
    enc.train()
    assert not enc.model.training  # frozen features never see dropout or drop path
    out = enc(reference["x"])
    assert out.shape == (2, LP.EMBED_DIM) and not out.requires_grad
    _matches(out.numpy(), reference["mean"])
    cls_enc = get_eeg_encoder("labram_pretrained", pooling="cls")
    _matches(cls_enc(reference["x"]).numpy(), reference["cls"])
    with pytest.raises(ValueError, match="expected"):
        enc(torch.zeros(2, 27, 2000))
    with pytest.raises(ValueError, match="not supported"):
        get_eeg_encoder("labram_pretrained", emb_size=128)
    with pytest.raises(ValueError, match="pooling"):
        get_eeg_encoder("labram_pretrained", pooling="max")
    with pytest.raises(FileNotFoundError):
        get_eeg_encoder("labram_pretrained", weights_path=Path("/nonexistent/labram.pt"))


@needs_weights
def test_pretrained_encoder_trainable_mode(reference):
    from exp2_fusion.models.eeg_encoders import get_eeg_encoder

    enc = get_eeg_encoder("labram_pretrained", frozen=False, drop_path_prob=0.1)
    trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    assert trainable == LP.EXPECTED_N_PARAMS
    enc.train()
    assert enc.model.training
    out = enc(reference["x"])
    assert out.requires_grad and out.shape == (2, LP.EMBED_DIM)
    enc.eval()
    with torch.no_grad():
        _matches(enc(reference["x"]).numpy(), reference["mean"])


def test_precomputed_encoder_is_an_identity_with_shape_checks():
    from exp2_fusion.models.eeg_encoders import PrecomputedFeatureEncoder, flatten_windows, get_eeg_encoder

    enc = get_eeg_encoder("precomputed", emb_size=512)
    assert isinstance(enc, PrecomputedFeatureEncoder) and enc.INPUT_CONVENTION == "precomputed"
    assert sum(p.numel() for p in enc.parameters()) == 0
    x = torch.randn(7, 512)
    assert torch.equal(enc(x), x)
    with pytest.raises(ValueError, match="precomputed features"):
        enc(torch.randn(7, 200))
    with pytest.raises(ValueError, match="precomputed features"):
        enc(torch.randn(7, 19, 2000))
    flat, b, w = flatten_windows(torch.zeros(3, 120, 512))
    assert flat.shape == (360, 512) and (b, w) == (3, 120)
    flat, b, w = flatten_windows(torch.zeros(3, 120, 19, 2000))
    assert flat.shape == (360, 19, 2000)


def test_fusion_models_accept_precomputed_features():
    from exp2_fusion.models.fusion import EEGSMILESFuseMoE, EEGSMILESMLPFusion
    from exp9_eeg_investigation.run_experiments import AblationModel

    eeg = torch.randn(2, 120, 512)
    mask = torch.zeros(2, 120, dtype=torch.bool)
    mask[1, 90:] = True
    smiles = torch.randn(2, 768)
    mlp = EEGSMILESMLPFusion(eeg_encoder_type="precomputed", eeg_embed_dim=512, window_chunk_size=120).eval()
    moe = EEGSMILESFuseMoE(eeg_encoder_type="precomputed", eeg_embed_dim=512, window_chunk_size=120).eval()
    abl = AblationModel(encoder_type="precomputed", embed_dim=512, window_chunk_size=120).eval()
    with torch.no_grad():
        assert mlp(eeg, mask, smiles).shape == (2, 2)
        logits, _aux = moe(eeg, mask, smiles)
        assert logits.shape == (2, 2)
        assert abl(eeg, mask, smiles).shape == (2, 2)


def _dim(cls, name, default):
    param = inspect.signature(cls.__init__).parameters.get(name)
    return default if param is None or param.default is inspect.Parameter.empty else param.default


@pytest.mark.parametrize("spec", [
    ("exp3_fusion.models.triple_mlp", "TripleModalityMLP", {"eeg_embed_dim": 512}, "text,eeg,mask,smiles", False),
    ("exp3_fusion.models.triple_fusemoe", "TripleModalityFuseMoE", {"eeg_embed_dim": 512}, "text,eeg,mask,smiles", True),
    ("exp5_clinical_fusion.models", "ClinicalEEGFusion", {}, "encode", False),
    ("exp6_clinical_triple.models", "ClinicalSMILESEEGFusion", {}, "encode", False),
    ("exp7_all_modalities.models", "QuadFusionMLP", {}, "encode", False),
    ("exp7_all_modalities.models", "QuadFusionMoE", {"eeg_embed_dim": 512}, "clinical,text,eeg,mask,smiles", True),
    ("exp11_eeg_upgrade.models", "ClinicalEEGFusionv2", {"eeg_embed_dim": 512}, "clinical,smiles,eeg,mask", False),
    ("exp11_eeg_upgrade.models", "QuadMLPv2", {"eeg_embed_dim": 512}, "clinical,text,eeg,mask,smiles", False),
])
def test_every_fusion_model_accepts_precomputed_features(spec):
    """Stored per-window features (batch, windows, dim) go through each model's EEG path.
    Models whose EEG width is fixed at 256 take 256-wide features; the others take 512."""
    import importlib
    module, name, kwargs, call, returns_tuple = spec
    cls = getattr(importlib.import_module(module), name)
    model = cls(eeg_encoder_type="precomputed", window_chunk_size=120, **kwargs).eval()
    width = kwargs.get("eeg_embed_dim", 256)
    eeg = torch.randn(2, 120, width)
    mask = torch.zeros(2, 120, dtype=torch.bool)
    mask[1, 90:] = True
    inputs = {"eeg": eeg, "mask": mask, "text": torch.randn(2, _dim(cls, "text_dim", 768)),
              "smiles": torch.randn(2, _dim(cls, "smiles_dim", 768)), "clinical": torch.randn(2, _dim(cls, "clinical_dim", 23))}
    with torch.no_grad():
        if call == "encode":
            out = model.encode_eeg_windows(eeg, mask)
            assert out.shape[0] == 2 and out.ndim == 2
        else:
            out = model(*[inputs[k] for k in call.split(",")])
            logits = out[0] if returns_tuple else out
            assert logits.shape == (2, 2)
    with pytest.raises(ValueError, match="precomputed features"):
        with torch.no_grad():
            if call == "encode":
                model.encode_eeg_windows(torch.randn(2, 120, 19, 2000), mask)
            else:
                model(*[torch.randn(2, 120, 19, 2000) if k == "eeg" else inputs[k] for k in call.split(",")])


def test_unknown_encoder_type_lists_the_options():
    from exp2_fusion.models.eeg_encoders import ENCODER_TYPES, get_eeg_encoder

    assert {"labram_pretrained", "precomputed"} <= set(ENCODER_TYPES)
    with pytest.raises(ValueError, match="labram_pretrained"):
        get_eeg_encoder("reve")
