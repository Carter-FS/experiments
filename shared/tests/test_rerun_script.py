"""rerun_clean.sh: item lists, the pending-array index list and the EEG archive dry run."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def run(args, **env):
    e = {**os.environ, **{k: str(v) for k, v in env.items()}}
    return subprocess.run(["bash", "rerun_clean.sh", *args], cwd=REPO, env=e, capture_output=True, text=True)


def test_script_parses_and_lists_items():
    assert subprocess.run(["bash", "-n", "rerun_clean.sh"], cwd=REPO).returncode == 0
    items = run(["list"]).stdout.split()
    assert len(items) == 373 and items[0] == "exp4:42" and not any(i.startswith("reve:") for i in items)
    assert "exp9_encoder_labram_pretrained_frozen:42" in items and "exp9_encoder_reve_frozen:46" in items
    assert "exp9_encoder_labram:42" not in items
    cpu = run(["list-cpu"]).stdout.split()
    assert len(cpu) == 190 and "exp9_encoder_reve_frozen:42" in cpu and "exp2:42" not in cpu
    assert len(run(["list"], PROTOCOL="innersplit").stdout.split()) == 289


def test_pending_array_skips_done_and_deferred_items(tmp_path):
    out = tmp_path / "outputs"
    (out / "_clean_rerun_v2" / "refit").mkdir(parents=True)
    (out / "_clean_rerun_v2" / "refit" / "exp4_s42.done").touch()
    idx = [int(i) for i in run(["pending-array"], OUT=out).stdout.strip().split(",")]
    items = run(["list"]).stdout.split()
    assert 0 not in idx                                        # exp4:42 is done
    assert items.index("exp4:43") in idx and items.index("exp2:42") in idx
    assert items.index("exp9_encoder_frozen:42") not in idx    # deferred
    # deferred under the refit protocol: exp9 extras, exp11, and the harmonised-label EEG items
    deferred = [i for i in items if i.split(":")[0].startswith(("exp9_encoder_frozen", "exp9_aggregator_", "exp9_embed_dim_",
                                                                  "exp11_", "hep_eeg_h12", "exp18_h12_Exp5c", "exp18_h12_Exp6b",
                                                                  "exp18_h12_Exp7a"))]
    assert len(idx) == len(items) - 1 - len(deferred)


def test_archive_eeg_dry_run_lists_eeg_outputs_only(tmp_path):
    out = tmp_path / "outputs"
    for d in ("exp9_predictions", "exp5_predictions", "eeg_cache", "_clean_rerun_v2/refit"):
        (out / d).mkdir(parents=True)
    keep = out / "exp5_predictions" / "predictions_oof_exp5a_chemberta_sp-multilabel_rf5_s42.json"
    eeg = [out / "exp9_predictions" / "predictions_oof_exp9_encoder_eeg2vec_sp-multilabel_rf5_s42.json",
           out / "exp5_predictions" / "predictions_oof_exp5c_eeg2vec_sp-multilabel_rf5_s42.json",
           out / "eeg_cache" / "processed_eeg.pkl",
           out / "_clean_rerun_v2" / "refit" / "exp9_encoder_eeg2vec_s42.done",
           out / "_clean_rerun_v2" / "refit" / "exp5_s42.done"]
    for f in [keep, *eeg, out / "_clean_rerun_v2" / "refit" / "exp4_s42.done"]:
        f.touch()
    res = run(["archive-eeg"], OUT=out, DRY_RUN=1)
    assert res.returncode == 0, res.stderr
    listed = [line.split()[2] for line in res.stdout.splitlines() if line.startswith("would move")]
    for f in eeg:
        assert str(f) in listed, f
    assert str(keep) not in listed and str(out / "_clean_rerun_v2" / "refit" / "exp4_s42.done") not in listed
    assert all(f.exists() for f in [keep, *eeg])             # a dry run moves nothing
