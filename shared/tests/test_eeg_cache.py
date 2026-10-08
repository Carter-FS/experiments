"""Tests for shared.eeg_cache: channel selection, units, windowing, the leading-flat
skip, the load-time conventions, the legacy refusal and the build round trip.

Recordings are synthetic MNE RawArrays, so no EDF file or patient data is needed.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import mne
import numpy as np
import pytest

from shared import eeg_cache as C

ALFRED_HEADER_CHANNELS = ("EMG+ PG1 Fp1 Fp2 PG2 EMG- F7 F3 Fz F4 F8 A1 T7 C3 Cz C4 T8 A2 "
                          "P7 P3 Pz P4 P8 ECG+ O1 O2 ECG-").split()
NOISE_SD_UV = 10.0
ECG_SD_UV = 1000.0


def synthetic_raw(duration_s: float = 900.0, flat_lead_s: float = 0.0, sfreq: float = 250.0,
                  names=ALFRED_HEADER_CHANNELS, seed: int = 0) -> mne.io.RawArray:
    """27 header channels as in the Melbourne EDFs, white noise of 10 uV SD, ECG at 1 mV."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sfreq)
    data_uv = rng.normal(0.0, NOISE_SD_UV, (len(names), n))
    for i, name in enumerate(names):
        if name.startswith("ECG"):
            data_uv[i] = rng.normal(0.0, ECG_SD_UV, n)
    data_uv[:, : int(flat_lead_s * sfreq)] = 0.0
    info = mne.create_info(list(names), sfreq, ch_types="eeg")
    return mne.io.RawArray(data_uv * 1e-6, info, verbose=False)   # MNE stores volts


def test_process_raw_keeps_only_the_19_standard_channels_in_order():
    rec = C.process_raw(synthetic_raw(), notch_hz=50.0)
    assert rec.windows_uv.shape == (C.MAX_WINDOWS, 19, 2000)
    assert list(C.CH_NAMES) == ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "C3", "CZ", "C4", "T8",
                                "P7", "P3", "PZ", "P4", "P8", "O1", "O2"]
    valid = rec.windows_uv[~rec.padding_mask]
    # no channel carries the millivolt-scale ECG signal
    assert valid.std(axis=(0, 2)).max() < 100.0


def test_process_raw_units_are_microvolts_and_windows_are_counted():
    rec = C.process_raw(synthetic_raw(duration_s=900.0), notch_hz=50.0)
    assert rec.n_valid == 60                       # (900 - 300) / 10
    assert rec.padding_mask[:60].sum() == 0 and rec.padding_mask[60:].all()
    assert np.all(rec.windows_uv[rec.padding_mask] == 0)
    sd = rec.windows_uv[~rec.padding_mask].std(axis=-1)
    # white noise of 10 uV, band-passed 0.1-75 Hz at 200 Hz, keeps most of its power
    assert 6.0 < np.median(sd) < 12.0
    assert rec.windows_uv.dtype == np.float32 and rec.signal_start_s == 0.0
    assert rec.duration_s == pytest.approx(900.0)


def test_leading_flat_segment_is_skipped_in_100s_chunks():
    rec = C.process_raw(synthetic_raw(duration_s=1000.0, flat_lead_s=150.0), notch_hz=50.0)
    assert rec.signal_start_s == pytest.approx(100.0)   # first non-flat 100 s chunk, native rate
    assert rec.n_valid == 60                            # (1000 - 100 - 300) / 10
    assert rec.duration_s == pytest.approx(1000.0)
    # no window is flat once the skip has been applied
    assert rec.windows_uv[~rec.padding_mask].std(axis=-1).min() > 1.0


def test_find_signal_start_values():
    sfreq = 200.0
    data = np.zeros((19, int(350 * sfreq)))
    assert C.find_signal_start(data, sfreq) is None
    data[:, int(220 * sfreq):] = 5.0 * np.random.default_rng(1).standard_normal((19, int(130 * sfreq)))
    assert C.find_signal_start(data, sfreq) == int(200 * sfreq)


def test_entirely_flat_recording_is_skipped():
    with pytest.raises(C.SkipRecording) as exc:
        C.process_raw(synthetic_raw(duration_s=700.0, flat_lead_s=700.0), notch_hz=50.0)
    assert exc.value.reason == "flat"


def test_short_recording_is_skipped():
    with pytest.raises(C.SkipRecording) as exc:
        C.process_raw(synthetic_raw(duration_s=500.0), notch_hz=50.0)
    assert exc.value.reason == "too_short"


def test_missing_standard_channel_is_skipped():
    names = [n for n in ALFRED_HEADER_CHANNELS if n != "Cz"]
    with pytest.raises(C.SkipRecording) as exc:
        C.process_raw(synthetic_raw(names=names), notch_hz=50.0)
    assert exc.value.reason == "missing_channels"


def test_old_channel_names_are_mapped():
    names = [{"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}.get(n, n) for n in ALFRED_HEADER_CHANNELS]
    rec = C.process_raw(synthetic_raw(names=names, duration_s=700.0), notch_hz=60.0)
    assert rec.n_valid == 40


def _entry(seed: int, n_valid: int = 30, sd_uv: float = 20.0) -> dict:
    rng = np.random.default_rng(seed)
    w = np.zeros((C.MAX_WINDOWS, 19, 2000), dtype=np.float32)
    mask = np.ones(C.MAX_WINDOWS, dtype=bool)
    w[:n_valid] = rng.normal(3.0, sd_uv, (n_valid, 19, 2000))
    mask[:n_valid] = False
    return {"windows_uv": w, "padding_mask": mask, "ch_names": list(C.CH_NAMES), "sfreq": 200.0,
            "signal_start_s": 0.0, "duration_s": 600.0, "version": 2, "source_sha256": None}


def test_zscore_window_convention():
    e = _entry(0)
    e["windows_uv"][5, 3, :] = 7.0                      # a flat channel in one window
    out = C.normalise_windows(e["windows_uv"], e["padding_mask"], "zscore_window")
    valid = out[~e["padding_mask"]].astype(np.float64)
    assert np.allclose(valid.mean(axis=-1), 0.0, atol=1e-5)
    sd = valid.std(axis=-1)
    sd[5, 3] = 1.0                                       # flat channel becomes zeros
    assert np.allclose(sd, 1.0, atol=1e-4)
    assert np.all(out[5, 3] == 0.0)
    assert np.all(out[e["padding_mask"]] == 0.0)
    assert out.dtype == np.float32


def test_labram_and_raw_conventions():
    e = _entry(1)
    out = C.normalise_windows(e["windows_uv"], e["padding_mask"], "labram")
    assert np.allclose(out[~e["padding_mask"]], e["windows_uv"][~e["padding_mask"]] / 100.0)
    raw = C.normalise_windows(e["windows_uv"], e["padding_mask"], "raw_uv")
    assert np.array_equal(raw, e["windows_uv"])
    with pytest.raises(ValueError):
        C.normalise_windows(e["windows_uv"], e["padding_mask"], "zscore")


def test_write_load_and_info_round_trip(tmp_path):
    path = tmp_path / "eeg19_v2_test.pkl"
    recs = {"a": _entry(0, n_valid=30), "b": _entry(1, n_valid=120)}
    meta = {"version": C.CACHE_VERSION, "cohort": "alfred", "ch_names": list(C.CH_NAMES), **C.summarise(recs)}
    C.write_cache(path, meta, recs)
    assert C.sidecar_path(path).exists()
    side = json.loads(C.sidecar_path(path).read_text())
    assert side["n_recordings"] == 2 and side["valid_windows"] == {"min": 30, "median": 75.0, "max": 120}
    assert side["n_full_length"] == 1 and 15.0 < side["median_window_sd_uv"] < 25.0
    loaded = C.load_cache(path, "zscore_window")
    assert set(loaded) == {"a", "b"}
    w, m = loaded["a"]
    assert w.shape == (C.MAX_WINDOWS, 19, 2000) and m.dtype == bool and m.sum() == 90
    assert abs(float(w[~m].mean())) < 1e-4
    info = C.cache_info(path)
    assert info["version"] == 2 and info["n_recordings"] == 2 and info["n_channels"] == 19
    assert info["samples_per_window"] == 2000
    C.sidecar_path(path).unlink()                                   # pickle fallback
    assert C.cache_info(path) == info
    assert not list(tmp_path.glob(".*.partial"))


def test_legacy_cache_is_refused_unless_allowed(tmp_path):
    path = tmp_path / "processed_eeg.pkl"
    legacy = {"p": (np.zeros((120, 27, 2000), np.float32), np.ones(120, bool))}
    with open(path, "wb") as fh:
        pickle.dump(legacy, fh)
    with pytest.raises(ValueError, match="not a version-2"):
        C.load_cache(path, "zscore_window")
    with pytest.raises(ValueError):
        C.cache_info(path)
    out = C.load_cache(path, "raw_uv", allow_legacy=True)
    assert out["p"][0].shape == (120, 27, 2000)
    with pytest.raises(ValueError, match="raw_uv"):
        C.load_cache(path, "zscore_window", allow_legacy=True)


def test_map_patients_to_edf_matches_exact_ids(tmp_path):
    for name in ["10_1_1-1-2019.edf", "1002,EEG,7565706.edf", "N009_15-3-2018.EDF", "1002,EEG,later.edf",
                 "777.edf", "88-15-3-2018.edf", "15-3-2018.edf", "notes.txt"]:
        (tmp_path / name).write_bytes(b"")
    files = C.scan_edf_files(tmp_path)
    assert len(files) == 7
    mapping, counts = C.map_patients_to_edf(files)
    assert set(mapping) == {"10", "1002", "N009", "777", "88"}
    assert mapping["1002"].name == "1002,EEG,7565706.edf"       # first in sorted order
    assert counts == {"multiple_files": 1, "unmatched": 1}        # the date-only stem
    assert "100" not in mapping                                  # no prefix matching
    mapping, counts = C.map_patients_to_edf(files, known_pids=["10", "1002", "15", "100"])
    assert set(mapping) == {"10", "1002"}                        # "15" never matches a date-only stem
    assert counts == {"multiple_files": 1, "unmatched": 4}


def test_patient_id_from_filename_forms():
    assert C.patient_id_from_filename("083_7085712_15-2-2019.edf") == "083"
    assert C.patient_id_from_filename("093,EEG,07022018.edf") == "093"
    assert C.patient_id_from_filename("1234.edf") == "1234"
    assert C.patient_id_from_filename("N7-5-11-2019.edf") == "N7"
    assert C.patient_id_from_filename("15-3-2018.edf") is None
    assert C.patient_id_from_filename("a b.edf") is None


def test_build_cache_with_synthetic_reader(tmp_path):
    specs = {"a.edf": dict(duration_s=900.0), "b.edf": dict(duration_s=500.0),
             "c.edf": dict(duration_s=800.0, flat_lead_s=150.0)}
    for name in specs:
        (tmp_path / name).write_bytes(b"x")
    pairs = [(name[0], tmp_path / name) for name in specs]

    def reader(path: Path):
        return synthetic_raw(**specs[path.name], seed=sorted(specs).index(path.name))

    out = tmp_path / "cache.pkl"
    meta = C.build_cache("alfred", out, reader=reader, pairs=pairs)
    assert meta["n_processed"] == 3 and meta["n_recordings"] == 2
    assert meta["skipped"] == {"missing_channels": 0, "units": 0, "flat": 0, "too_short": 1,
                               "non_finite": 0, "read_error": 0}
    assert meta["incomplete"] is False and meta["n_flat_valid_windows"] == 0
    assert meta["samples_per_window"] == 2000 and meta["n_channels"] == 19
    assert meta["n_recordings_with_leading_flat_segment"] == 1
    assert meta["notch_hz"] == 50.0 and meta["units"] == "uV" and meta["ch_names"] == list(C.CH_NAMES)
    loaded = C.load_cache(out, "labram")
    assert set(loaded) == {"a", "c"}
    assert np.isfinite(loaded["a"][0]).all()
    entry = pickle.load(open(out, "rb"))["recordings"]["a"]
    assert entry["source_sha256"] is not None and entry["version"] == 2


def test_non_finite_samples_skip_the_recording():
    raw = synthetic_raw(duration_s=700.0)
    data = raw.get_data()
    data[3, 1000] = np.nan
    raw = mne.io.RawArray(data, raw.info, verbose=False)
    with pytest.raises(C.SkipRecording) as exc:
        C.process_raw(raw, notch_hz=50.0)
    assert exc.value.reason == "non_finite"


def test_non_voltage_channel_skips_the_recording():
    raw = synthetic_raw(duration_s=700.0)
    raw._orig_units = {ch: " UV " for ch in raw.ch_names}        # case and whitespace are tolerated
    raw._orig_units["Cz"] = "mmHg"
    with pytest.raises(C.SkipRecording) as exc:
        C.process_raw(raw, notch_hz=50.0)
    assert exc.value.reason == "units"


def test_mis_typed_voltage_channel_is_still_scaled():
    raw = synthetic_raw(duration_s=700.0)
    raw.set_channel_types({"Cz": "misc"}, verbose=False)
    rec = C.process_raw(raw, notch_hz=50.0)
    sd = rec.windows_uv[~rec.padding_mask].std(axis=-1)
    assert 6.0 < np.median(sd[:, C.CH_NAMES.index("CZ")]) < 12.0


def test_flat_windows_inside_a_recording_are_counted():
    raw = synthetic_raw(duration_s=900.0)
    data = raw.get_data()
    data[:, int(700 * 250):] = 0.0                          # flat for the last 200 s
    rec = C.process_raw(mne.io.RawArray(data, raw.info, verbose=False), notch_hz=50.0)
    side = C.summarise({"x": rec.to_entry(None)})
    assert rec.n_valid == 60 and side["n_flat_valid_windows"] >= 18


def test_edf_round_trip_scales_to_microvolts_per_channel(tmp_path):
    """A 20 uV sine on Fp1 and a 5 uV sine on O2, written to EDF and read back, give
    window SDs of 14.1 and 3.5 uV in rows 0 and 18 (amplitude / sqrt 2)."""
    pytest.importorskip("edfio")
    sfreq, duration_s = 250.0, 700.0
    t = np.arange(int(duration_s * sfreq)) / sfreq
    rng = np.random.default_rng(3)
    data_uv = rng.normal(0.0, 2.0, (len(ALFRED_HEADER_CHANNELS), t.size))
    data_uv[ALFRED_HEADER_CHANNELS.index("Fp1")] = 20.0 * np.sin(2 * np.pi * 10.0 * t)
    data_uv[ALFRED_HEADER_CHANNELS.index("O2")] = 5.0 * np.sin(2 * np.pi * 7.0 * t)
    data_uv[ALFRED_HEADER_CHANNELS.index("ECG+")] = rng.normal(0.0, ECG_SD_UV, t.size)
    raw = mne.io.RawArray(data_uv * 1e-6, mne.create_info(list(ALFRED_HEADER_CHANNELS), sfreq, ch_types="eeg"),
                          verbose=False)
    path = tmp_path / "77_1_1-1-2020.edf"
    mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose=False)
    rec = C.process_edf(path, notch_hz=50.0)
    sd = np.median(rec.windows_uv[~rec.padding_mask].std(axis=-1), axis=0)
    assert sd[0] == pytest.approx(20.0 / np.sqrt(2), rel=0.05)      # FP1 is row 0
    assert sd[18] == pytest.approx(5.0 / np.sqrt(2), rel=0.05)      # O2 is row 18
    assert sd.max() < 20.0                                           # the 1 mV ECG is gone
    assert rec.n_valid == 40


def test_expect_files_marks_an_incomplete_copy(tmp_path, caplog):
    (tmp_path / "a_1_1-1-2020.edf").write_bytes(b"x")
    out = tmp_path / "cache.pkl"
    with caplog.at_level("WARNING", logger="shared.eeg_cache"):
        meta = C.build_cache("hep", out, reader=lambda p: synthetic_raw(duration_s=700.0),
                             pairs=[("a", tmp_path / "a_1_1-1-2020.edf")], expect_files=5)
    assert meta["incomplete"] is True and meta["expect_files"] == 5 and meta["notch_hz"] == 60.0
    assert any("expected 5" in rec.message for rec in caplog.records)


def test_flat_channel_is_counted_in_the_sidecar():
    e = _entry(4, n_valid=20)
    e["windows_uv"][:, 7, :] = 0.0                         # one channel flat in every window
    side = C.summarise({"x": e})
    assert side["n_recordings_with_flat_channel"] == 1 and side["n_flat_valid_windows"] == 0


def test_corrupt_sidecar_falls_back_to_the_pickle(tmp_path):
    path = tmp_path / "c.pkl"
    recs = {"a": _entry(0)}
    C.write_cache(path, {"version": C.CACHE_VERSION, "cohort": "alfred"}, recs)
    C.sidecar_path(path).write_text("{not json")
    info = C.cache_info(path)
    assert info["n_recordings"] == 1 and info["n_channels"] == 19
