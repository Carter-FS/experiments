"""Stored per-window feature sets (shared.eeg_features) and their consumers: the exp9
encoder arms that read them, the exp15 feature-set switch, and the datasets that pass
(windows, dim) features through."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from exp2_fusion.config import MAX_WINDOWS
from shared import eeg_cache as C
from shared import eeg_features as F


def _write_features(path, pids, counts, dim, seed=0, max_windows=MAX_WINDOWS, finite=True):
    rng = np.random.default_rng(seed)
    feats = np.zeros((len(pids), max_windows, dim), dtype=np.float32)
    for i, n in enumerate(counts):
        feats[i, :min(n, max_windows)] = rng.normal(size=(min(n, max_windows), dim))
    if not finite:
        feats[0, 0, 0] = np.nan
    np.savez_compressed(path, features=feats, pids=np.array(pids), valid_window_counts=np.array(counts, dtype=np.int32),
                        meta=json.dumps({"made_by": "test"}))
    return path


def _write_cache(path, pids, counts):
    recs = {}
    for pid, n in zip(pids, counts):
        w = np.zeros((MAX_WINDOWS, 19, 2000), dtype=np.float32)
        m = np.ones(MAX_WINDOWS, dtype=bool)
        w[:n] = 1.0
        m[:n] = False
        recs[pid] = {"windows_uv": w, "padding_mask": m, "ch_names": list(C.CH_NAMES), "sfreq": 200.0,
                     "signal_start_s": 0.0, "duration_s": 600.0, "version": 2, "source_sha256": None}
    C.write_cache(path, {"version": C.CACHE_VERSION, "cohort": "alfred"}, recs)
    return path


def test_registry_paths_and_widths():
    assert F.feature_path("reve_v2", "alfred").name == "reve_features_v2_alfred.npz"
    assert F.feature_path("labram_v2", "hep").name == "labram_features_v2_hep.npz"
    assert F.FEATURE_SETS["reve_v2"]["dim"] == 512 and F.FEATURE_SETS["labram_v2"]["dim"] == 200
    assert F.FEATURE_SETS["labram_v2"]["input_convention"] == "labram"
    with pytest.raises(ValueError, match="unknown feature set"):
        F.feature_path("reve_v1", "alfred")
    with pytest.raises(ValueError, match="unknown cohort"):
        F.feature_path("reve_v2", "melbourne")


def test_load_features_rebuilds_prefix_masks_and_zeroes_padding(tmp_path):
    path = _write_features(tmp_path / "labram_features_v2_alfred.npz", ["p2", "p1", "p3"], [5, 0, MAX_WINDOWS], 200)
    out = F.load_features("labram_v2", path=path)
    assert list(out) == ["p2", "p1", "p3"]
    w, m = out["p2"]
    assert w.shape == (MAX_WINDOWS, 200) and w.dtype == np.float32 and m.dtype == bool
    assert m.sum() == MAX_WINDOWS - 5 and not m[:5].any() and not w[5:].any() and w[:5].all()
    assert out["p1"][1].all() and not out["p3"][1].any()
    assert F.feature_meta("labram_v2", path=path) == {"made_by": "test"}


@pytest.mark.parametrize("bad", ["dim", "max_windows", "duplicate", "count", "nan", "missing_key"])
def test_load_features_rejects_malformed_files(tmp_path, bad):
    path = tmp_path / "reve_features_v2_alfred.npz"
    if bad == "dim":
        _write_features(path, ["a"], [3], 200)
        msg = "expected"
    elif bad == "max_windows":
        _write_features(path, ["a"], [3], 512, max_windows=60)
        msg = "expected"
    elif bad == "duplicate":
        _write_features(path, ["a", "a"], [3, 3], 512)
        msg = "duplicate"
    elif bad == "count":
        _write_features(path, ["a"], [MAX_WINDOWS + 1], 512)
        msg = "counts outside"
    elif bad == "nan":
        _write_features(path, ["a"], [3], 512, finite=False)
        msg = "non-finite"
    else:
        np.savez_compressed(path, features=np.zeros((1, MAX_WINDOWS, 512), np.float32), pids=np.array(["a"]))
        msg = "lacks"
    with pytest.raises(ValueError, match=msg):
        F.load_features("reve_v2", path=path)
    with pytest.raises(FileNotFoundError, match="produce them"):
        F.load_features("reve_v2", path=tmp_path / "absent.npz")


def test_check_against_cache_requires_the_same_patients_and_masks(tmp_path):
    cache = _write_cache(tmp_path / "eeg19_v2_alfred.pkl", ["a", "b"], [10, 4])
    good = F.load_features("reve_v2", path=_write_features(tmp_path / "good.npz", ["b", "a"], [4, 10], 512))
    assert F.check_features_against_cache(good, cache) == {"n_recordings": 2, "n_valid_windows": 14}
    fewer = F.load_features("reve_v2", path=_write_features(tmp_path / "fewer.npz", ["a"], [10], 512))
    with pytest.raises(ValueError, match="without features"):
        F.check_features_against_cache(fewer, cache)
    shifted = F.load_features("reve_v2", path=_write_features(tmp_path / "shifted.npz", ["a", "b"], [10, 5], 512))
    with pytest.raises(ValueError, match="different padding mask"):
        F.check_features_against_cache(shifted, cache)


def test_check_cli(tmp_path, monkeypatch):
    cache = _write_cache(tmp_path / "eeg19_v2_alfred.pkl", ["a"], [7])
    _write_features(tmp_path / "labram_features_v2_alfred.npz", ["a"], [7], 200)
    monkeypatch.setattr(F, "OUT_DIR", tmp_path)
    monkeypatch.setitem(F.CACHE_PATHS, "alfred", cache)
    assert F.main(["check", "--feature-set", "labram_v2", "--cohort", "alfred"]) == 0


def test_exp2_dataset_passes_feature_windows_through(tmp_path):
    from exp2_fusion.data_pipeline import EEGSMILESDataset
    feats = F.load_features("labram_v2", path=_write_features(tmp_path / "f.npz", ["7"], [3], 200))
    ds = EEGSMILESDataset(patient_ids=["7"], eeg_data=feats, smiles_embeddings=np.ones((1, 8), np.float32),
                          smiles_indices={"LEV": 0}, labels={"7": 1}, asm_drugs={"7": "LEV"}, max_channels=19)
    eeg, mask, smiles, label = ds[0]
    assert eeg.shape == (MAX_WINDOWS, 200) and mask.shape == (MAX_WINDOWS,) and int(mask.sum()) == MAX_WINDOWS - 3
    assert smiles.shape == (8,) and int(label) == 1


# ---------------------------------------------------------------------------
# exp9 encoder arms
# ---------------------------------------------------------------------------


def test_exp9_arms_declare_their_eeg_inputs():
    import exp9_eeg_investigation.run_experiments as r
    arms = {e["name"]: e for e in r.define_ablation_experiments()}
    assert "encoder_labram" not in arms and "encoder_labram_scratch" in arms
    assert arms["encoder_labram_scratch"]["encoder_type"] == "labram"
    for name, fs in (("encoder_labram_pretrained_frozen", "labram_v2"), ("encoder_reve_frozen", "reve_v2")):
        arm = arms[name]
        assert arm["encoder_type"] == "precomputed" and arm["input"] == {"kind": "features", "feature_set": fs}
        assert arm["embed_dim"] == F.FEATURE_SETS[fs]["dim"] and arm["output_dim"] == 256
    specs = {r.input_key(r.input_spec(e)) for e in arms.values()}
    assert len(specs) == 3  # the raw cache under the window z-score, and the two feature sets
    raw = [e for e in arms.values() if e["encoder_type"] != "precomputed"]
    assert all(r.input_spec(e) == r.RAW_EEG_INPUT for e in raw)
    with pytest.raises(ValueError, match="needs a feature-set input"):
        r.input_spec({"name": "x", "encoder_type": "precomputed"})
    with pytest.raises(ValueError, match="needs a cache input"):
        r.input_spec({"name": "x", "encoder_type": "simplecnn", "input": {"kind": "features", "feature_set": "reve_v2"}})
    with pytest.raises(ValueError, match="unknown feature set"):
        r.input_spec({"name": "x", "encoder_type": "precomputed", "input": {"kind": "features", "feature_set": "reve_v1"}})
    with pytest.raises(ValueError, match="unsupported cache convention"):
        r.input_spec({"name": "x", "encoder_type": "simplecnn", "input": {"kind": "cache", "convention": "raw_uv"}})


def test_exp9_rejects_a_width_mismatch_for_a_feature_arm():
    import exp9_eeg_investigation.run_experiments as r
    arm = {"name": "bad", "encoder_type": "precomputed", "input": {"kind": "features", "feature_set": "reve_v2"},
           "embed_dim": 256}
    with pytest.raises(ValueError, match="512-dimensional"):
        r.run_ablation_experiment(arm, {}, np.zeros((1, 8)), {}, pd.DataFrame({"outcome": [0, 1]}), torch.device("cpu"),
                                  splitter="legacy", use_multilabel_stratification=False)


def _synthetic_inputs(n=12, seed=0):
    """A tiny cohort: raw windows for the cache spec, 200-wide features for labram_v2."""
    rng = np.random.default_rng(seed)
    pids = [str(100 + i) for i in range(n)]
    counts = [6] * n
    raw, feat = {}, {}
    for pid, c in zip(pids, counts):
        w = np.zeros((MAX_WINDOWS, 19, 2000), np.float32)
        w[:c] = rng.normal(size=(c, 19, 2000))
        f = np.zeros((MAX_WINDOWS, 200), np.float32)
        f[:c] = rng.normal(size=(c, 200))
        m = np.arange(MAX_WINDOWS) >= c
        raw[pid], feat[pid] = (w, m), (f, m)
    df = pd.DataFrame({"pid": pids, "outcome": [i % 2 for i in range(n)], "ASM": ["LEV"] * n,
                       "focal": [1] * n, "sex": [0] * n})
    return raw, feat, df


def test_exp9_runs_raw_and_feature_arms_on_one_cohort(tmp_path, monkeypatch):
    """Two arms with different EEG inputs run end to end on a tiny synthetic cohort and
    write their prediction files; the inputs are loaded once per spec."""
    import exp9_eeg_investigation.run_experiments as r
    raw, feat, df = _synthetic_inputs()
    loads = []

    def fake_load(spec, cohort="alfred"):
        loads.append(spec)
        return (raw if spec["kind"] == "cache" else feat), df.copy()

    monkeypatch.setattr(r, "load_eeg_input", fake_load)
    monkeypatch.setattr(r, "load_smiles_embeddings", lambda m: (np.ones((1, 8), np.float32), {"LEV": 0}))
    monkeypatch.setitem(r.CV_CONFIG, "n_splits", 2)
    monkeypatch.setattr(r, "RESULTS_DIR", tmp_path / "results")
    arms = [
        {"name": "baseline_simplecnn_transformer", "encoder_type": "simplecnn", "aggregator_type": "transformer",
         "embed_dim": 32, "output_dim": 32, "num_layers": 1},
        {"name": "encoder_labram_pretrained_frozen", "encoder_type": "precomputed",
         "input": {"kind": "features", "feature_set": "labram_v2"}, "aggregator_type": "transformer",
         "embed_dim": 200, "output_dim": 32, "num_layers": 1},
    ]
    results = r.run_all_ablations(experiments=arms, use_multilabel=False, log_predictions=True,
                                  predictions_dir=tmp_path / "pred", splitter="legacy", inner_val=0.0)
    assert [x["name"] for x in results] == [a["name"] for a in arms]
    assert all("error" not in x for x in results), [x.get("error") for x in results]
    assert [s["kind"] for s in loads] == ["cache", "features"]
    for a in arms:
        payload = json.loads((tmp_path / "pred" / f"predictions_oof_exp9_{a['name']}.json").read_text())
        assert payload["metadata"]["eeg_input"]["kind"] == ("features" if a["encoder_type"] == "precomputed" else "cache")
        assert sum(len(f["pids"]) for f in payload["folds"]) == len(df)


def test_exp9_refuses_inputs_with_different_cohorts(tmp_path, monkeypatch):
    import exp9_eeg_investigation.run_experiments as r
    raw, feat, df = _synthetic_inputs()

    def fake_load(spec, cohort="alfred"):
        if spec["kind"] == "cache":
            return raw, df.copy()
        return feat, df.iloc[:-1].copy()

    monkeypatch.setattr(r, "load_eeg_input", fake_load)
    monkeypatch.setattr(r, "load_smiles_embeddings", lambda m: (np.ones((1, 8), np.float32), {"LEV": 0}))
    arms = [{"name": "a", "encoder_type": "simplecnn", "embed_dim": 32, "output_dim": 32, "num_layers": 1},
            {"name": "b", "encoder_type": "precomputed", "input": {"kind": "features", "feature_set": "labram_v2"},
             "embed_dim": 200, "output_dim": 32, "num_layers": 1}]
    monkeypatch.setitem(r.CV_CONFIG, "n_splits", 2)
    monkeypatch.setattr(r, "RESULTS_DIR", tmp_path / "results")
    with pytest.raises(RuntimeError, match="different cohort"):
        r.run_all_ablations(experiments=arms, use_multilabel=False, splitter="legacy")


# ---------------------------------------------------------------------------
# exp15 feature-set switch
# ---------------------------------------------------------------------------


def test_exp15_dataset_records_the_feature_width_and_the_model_follows_it():
    from exp15_reve_quad_mlp.data_pipeline import ReveQuadDataset
    from exp15_reve_quad_mlp.models import get_model
    n = 3
    windows = [np.zeros((MAX_WINDOWS, 200), np.float32) for _ in range(n)]
    masks = [np.arange(MAX_WINDOWS) >= 6 for _ in range(n)]   # six valid windows each
    ds = ReveQuadDataset(clinical_features=np.zeros((n, 23), np.float32), text_embeddings=np.zeros((n, 768), np.float32),
                         reve_windows=windows, padding_masks=masks, smiles_embeddings=np.ones((1, 768), np.float32),
                         smiles_indices={"LEV": 0}, asm_drugs=["LEV"] * n, labels=np.array([0, 1, 0]))
    assert ds.feature_dim == 200
    model = get_model(reve_dim=ds.feature_dim).eval()
    clinical, text, eeg, mask, smiles, label = ds[0]
    with torch.no_grad():
        out = model(clinical[None], text[None], eeg[None], mask[None], smiles[None])
    assert out.shape == (1, 2)
    with pytest.raises(ValueError, match="widths differ"):
        ReveQuadDataset(clinical_features=np.zeros((2, 23), np.float32), text_embeddings=np.zeros((2, 768), np.float32),
                        reve_windows=[windows[0], np.zeros((MAX_WINDOWS, 512), np.float32)], padding_masks=masks[:2],
                        smiles_embeddings=np.ones((1, 768), np.float32), smiles_indices={"LEV": 0},
                        asm_drugs=["LEV"] * 2, labels=np.array([0, 1]))


def test_exp15_cli_and_output_names():
    import inspect
    import exp15_reve_quad_mlp.run_experiments as r
    from exp15_reve_quad_mlp import config as cfg
    assert cfg.DEFAULT_FEATURE_SET == "reve_v2" and cfg.REVE_DIM == 512
    assert inspect.signature(r.run_exp15_with_predictions).parameters["feature_set"].default == "reve_v2"
    src = inspect.getsource(r.run_exp15_with_predictions)
    assert 'f"predictions_oof_{feature_set}{suffix_part}.json"' in src and '"feature_set": feature_set' in src
