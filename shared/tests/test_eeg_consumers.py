"""Every EEG consumer reads the version-2 cache through shared.eeg_cache.

Guards: no module references the superseded caches; every configuration points at the
version-2 Melbourne cache with the 19-channel montage; the experiments' loaders apply the
per-window z-score and refuse a legacy pickle; model and dataset defaults carry the
19-channel count; the Melbourne EEG cohort is the CSV rows with a cached recording.
"""
from __future__ import annotations

import importlib
import inspect
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shared import eeg_cache as C

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_DIRS = sorted(p for p in REPO.glob("exp*") if p.is_dir()) + [REPO / "shared", REPO / "thesisStandalone" / "analysis"]
SUPERSEDED = re.compile(r"processed_eeg(?!\*)|preprocess_all_eeg\(|get_valid_patient_eeg_pairs\(|cache_eeg=|force_reprocess=")
# A 27-channel default or literal input shape anywhere in the experiment code.
OLD_MONTAGE = re.compile(r"(n_channels|n_eeg_channels|max_channels)\s*(:\s*int)?\s*=\s*27\b|,\s*27\s*,\s*2000\s*\)")
# Files allowed to mention the old names: the superseded Stage C builder, the loader
# (its refusal message names the legacy pickle) and tests.
ALLOWED = {"thesisStandalone/analysis/hep_eeg_preprocess.py", "shared/eeg_cache.py"}


def _python_files():
    for d in EXPERIMENT_DIRS:
        for f in d.rglob("*.py"):
            rel = f.relative_to(REPO).as_posix()
            if "/tests/" in rel or rel in ALLOWED:
                continue
            yield rel, f


def test_no_consumer_references_the_superseded_caches():
    hits = [f"{rel}:{i}" for rel, f in _python_files()
            for i, line in enumerate(f.read_text().splitlines(), 1) if SUPERSEDED.search(line)]
    assert hits == [], hits


def test_no_27_channel_default_remains():
    hits = [f"{rel}:{i}" for rel, f in _python_files()
            for i, line in enumerate(f.read_text().splitlines(), 1) if OLD_MONTAGE.search(line)]
    assert hits == [], hits


@pytest.mark.parametrize("module", sorted(p.relative_to(REPO).with_suffix("").as_posix().replace("/", ".")
                                          for p in REPO.glob("exp*/run_experiments.py")))
def test_every_run_script_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("script", ["hep_external_validation_eeg", "hep_reduced_external_validation", "reve_extract_features"])
def test_thesis_eeg_scripts_import(script):
    import sys
    root = str(REPO / "thesisStandalone")
    if root not in sys.path:
        sys.path.insert(0, root)
    importlib.import_module(f"analysis.{script}")


@pytest.mark.parametrize("module", ["exp3_fusion.config", "exp5_clinical_fusion.config", "exp6_clinical_triple.config",
                                    "exp7_all_modalities.config", "exp8_stratification.config", "exp9_eeg_investigation.config",
                                    "exp11_eeg_upgrade.config", "exp12_moe_hparam.config"])
def test_configs_use_the_v2_cache_and_19_channels(module):
    cfg = importlib.import_module(module)
    if hasattr(cfg, "EEG_CACHE_PATH"):
        assert cfg.EEG_CACHE_PATH == C.CACHE_PATHS["alfred"]
    for name in ("EEG_ENCODER_CONFIG", "EEG_CONFIG"):
        conf = getattr(cfg, name, None)
        if isinstance(conf, dict) and "n_channels" in conf:
            assert conf["n_channels"] == 19, (module, name)
    assert cfg.N_CHANNELS == 19


def test_exp2_and_exp3_prepare_through_the_loader():
    for module in ("exp2_fusion.data_pipeline", "exp3_fusion.data_pipeline"):
        dp = importlib.import_module(module)
        assert dp.EEG_CONVENTION == "zscore_window" and dp.EEG_CACHE_PATH == C.CACHE_PATHS["alfred"]
        src = inspect.getsource(dp.prepare_data)
        assert "load_cache(EEG_CACHE_PATH, EEG_CONVENTION)" in src and "eeg_patient_frame(eeg_data.keys())" in src


def test_exp18_and_portable_paths_are_v2():
    from exp18_mixed_cohort import config as cfg18
    from shared import portable_models as pm
    assert cfg18.MEL_EEG_CACHE == C.CACHE_PATHS["alfred"] and cfg18.HEP_EEG_CACHE == C.CACHE_PATHS["hep"]
    assert pm.N_CHANNELS == 19 and pm.EEG_CONVENTION == "zscore_window"


def _v2_cache(tmp_path, n_valid=30, sd_uv=20.0):
    rng = np.random.default_rng(0)
    w = np.zeros((C.MAX_WINDOWS, 19, 2000), dtype=np.float32)
    mask = np.ones(C.MAX_WINDOWS, dtype=bool)
    w[:n_valid] = rng.normal(5.0, sd_uv, (n_valid, 19, 2000))
    mask[:n_valid] = False
    recs = {"101": {"windows_uv": w, "padding_mask": mask, "ch_names": list(C.CH_NAMES), "sfreq": 200.0,
                    "signal_start_s": 0.0, "duration_s": 600.0, "version": 2, "source_sha256": None}}
    path = tmp_path / "eeg19_v2_alfred.pkl"
    C.write_cache(path, {"version": C.CACHE_VERSION, "cohort": "alfred"}, recs)
    return path


@pytest.mark.parametrize("module", ["exp5_clinical_fusion.data_pipeline", "exp6_clinical_triple.data_pipeline",
                                    "exp7_all_modalities.data_pipeline"])
def test_experiment_loaders_apply_the_window_zscore(tmp_path, module):
    dp = importlib.import_module(module)
    assert dp.EEG_CONVENTION == "zscore_window"
    assert inspect.signature(dp.load_eeg_data).parameters["cache_path"].default == C.CACHE_PATHS["alfred"]
    eeg = dp.load_eeg_data(_v2_cache(tmp_path))
    w, m = eeg["101"]
    assert w.shape == (C.MAX_WINDOWS, 19, 2000) and m.sum() == 90
    assert abs(float(w[~m].mean())) < 1e-4 and abs(float(w[~m].std(axis=-1).mean()) - 1.0) < 1e-3
    legacy = tmp_path / "processed_eeg.pkl"
    with open(legacy, "wb") as fh:
        pickle.dump({"101": (np.zeros((120, 27, 2000), np.float32), np.ones(120, bool))}, fh)
    with pytest.raises(ValueError, match="not a version-2"):
        dp.load_eeg_data(legacy)
    with pytest.raises(FileNotFoundError):
        dp.load_eeg_data(tmp_path / "missing.pkl")


def test_portable_loader_applies_the_window_zscore(tmp_path):
    from shared import portable_models as pm
    w, m = pm.load_eeg_cache(_v2_cache(tmp_path))["101"]
    assert abs(float(w[~m].mean())) < 1e-4 and abs(float(w[~m].std(axis=-1).mean()) - 1.0) < 1e-3


def test_eeg_patient_frame_is_the_csv_rows_with_a_cached_recording(tmp_path):
    csv = tmp_path / "alfred_1st_regimen.csv"
    pd.DataFrame({"pid": ["7", "8", "8", "9", "10"], "outcome": [1, 2, 2, 3, 1], "ASM": ["LEV"] * 5,
                  "focal": [1, 0, 0, 1, 1], "sex": [0, 1, 1, 0, 1]}).to_csv(csv, index=False)
    df = C.eeg_patient_frame(["8", "10", "11"], csv_path=csv)
    assert df["pid"].tolist() == ["8", "10"]           # 7 has no recording, 9 no usable outcome, 11 not in the CSV
    assert df["outcome"].tolist() == [0, 1]             # raw 2 -> 0, raw 1 -> 1 (OUTCOME_MAPPING)
    assert df["pid"].dtype == object and list(df.columns) == ["pid", "outcome", "ASM", "focal", "sex"]


def test_model_and_dataset_defaults_are_19_channels():
    from exp2_fusion.models import eeg_encoders, fusion
    from exp3_fusion.models import triple_mlp, triple_fusemoe
    from exp5_clinical_fusion import models as m5
    from exp6_clinical_triple import models as m6
    from exp7_all_modalities import models as m7
    from exp11_eeg_upgrade import models as m11
    checked = 0
    for mod in (eeg_encoders, fusion, triple_mlp, triple_fusemoe, m5, m6, m7, m11):
        for _name, obj in inspect.getmembers(mod, lambda o: inspect.isclass(o) or inspect.isfunction(o)):
            if getattr(obj, "__module__", None) != mod.__name__:
                continue
            try:
                params = inspect.signature(obj).parameters
            except (TypeError, ValueError):
                continue
            for pname in ("n_channels", "n_eeg_channels", "max_channels"):
                if pname in params and params[pname].default is not inspect.Parameter.empty:
                    assert params[pname].default == 19, (mod.__name__, _name, pname)
                    checked += 1
    assert checked >= 8


def test_importing_the_loader_does_not_import_mne_or_torch():
    import subprocess, sys
    code = "import sys; import shared.eeg_cache; print('mne' in sys.modules, 'torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False"


def test_cache_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ASM_EEG_CACHE_DIR", str(tmp_path))
    mod = importlib.reload(C)
    try:
        assert mod.CACHE_PATHS["alfred"] == tmp_path / "eeg19_v2_alfred.pkl"
    finally:
        monkeypatch.delenv("ASM_EEG_CACHE_DIR")
        importlib.reload(C)
