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


def _eeg_tree(tmp_path, guard_age="old"):
    """A synthetic outputs tree plus a thesis output directory: EEG-dependent files, files
    that must stay, the version-2 cache sidecar (the date guard) and done markers."""
    import time
    out = tmp_path / "outputs"
    thesis = tmp_path / "thesis_output"
    for d in ("exp9_predictions", "exp5_predictions", "exp6_predictions", "exp18_mixed_cohort", "eeg_cache",
              "exp9_results", "exp15_reve_quad/seed42_none", "_clean_rerun_v2/refit", "_clean_rerun_v2/innersplit"):
        (out / d).mkdir(parents=True)
    thesis.mkdir()
    eeg = [out / "exp9_predictions" / "predictions_oof_exp9_encoder_eeg2vec_sp-multilabel_rf5_s42.json",
           out / "exp5_predictions" / "predictions_oof_exp5c_eeg2vec_sp-multilabel_rf5_s42.json",
           out / "exp6_predictions" / "predictions_oof_exp6b_simplecnn_sp-multilabel_rf5_s42.json",
           out / "exp18_mixed_cohort" / "predictions_Exp7a_rf5_seed42.csv",
           out / "exp18_mixed_cohort" / "per_seed.csv",
           out / "exp9_results" / "ablation_results_20260901_000000.json",
           out / "exp15_reve_quad" / "seed42_none" / "predictions_oof.json",
           out / "eeg_cache" / "processed_eeg.pkl",
           out / "reve_features_alfred.npz",
           thesis / "hep_external_summary_eeg_sp-multilabel_rf5_s42.csv",
           thesis / "hep_external_cis_rf5.csv",
           thesis / "reve_summary.csv",
           thesis / "metrics_oof.csv",
           out / "_clean_rerun_v2" / "refit" / "exp9_encoder_eeg2vec_s42.done",
           out / "_clean_rerun_v2" / "refit" / "exp5_s42.done",
           out / "_clean_rerun_v2" / "innersplit" / "hep_eeg_s42.done"]
    keep = [out / "exp5_predictions" / "predictions_oof_exp5a_chemberta_sp-multilabel_rf5_s42.json",
            out / "exp6_predictions" / "predictions_oof_exp6a_chemberta_sp-multilabel_rf5_s42.json",
            out / "exp18_mixed_cohort" / "predictions_Exp4a_rf5_seed42.csv",
            out / "exp18_mixed_cohort" / "overlap_candidates.csv",
            out / "eeg_cache" / "eeg19_v2_alfred.pkl",
            out / "labram_features_v2_alfred.npz",
            thesis / "hep_external_summary_sp-multilabel_rf5_s42.csv",
            thesis / "hep_reverse_summary_sp-multilabel_rf5_s42.csv",
            thesis / "metrics_decomposition.csv",
            out / "_clean_rerun_v2" / "refit" / "exp4_s42.done"]
    for f in eeg + keep:
        f.touch()
    old = time.time() - 3600
    for f in eeg + keep:
        os.utime(f, (old, old))
    guard = out / "eeg_cache" / "eeg19_v2_alfred.pkl.meta.json"
    guard.write_text("{}")                      # newer than every file above
    fresh = out / "exp9_predictions" / "predictions_oof_exp9_encoder_reve_frozen_sp-multilabel_rf5_s42.json"
    fresh.touch()                               # written after the cache: a rerun output, must stay
    now = time.time() + 60
    os.utime(fresh, (now, now))
    return out, thesis, eeg, keep + [fresh], guard


def test_archive_eeg_moves_eeg_outputs_only_and_only_older_than_the_cache(tmp_path):
    out, thesis, eeg, keep, guard = _eeg_tree(tmp_path)
    env = {"OUT": out, "ASM_ANALYSIS_OUTPUT_DIR": thesis}
    dry = run(["archive-eeg"], DRY_RUN=1, **env)
    assert dry.returncode == 0, dry.stderr
    listed = [line.split()[2] for line in dry.stdout.splitlines() if line.startswith("would move")]
    assert sorted(listed) == sorted(str(f) for f in eeg)
    assert all(f.exists() for f in eeg)                          # a dry run moves nothing
    real = run(["archive-eeg"], **env)
    assert real.returncode == 0, real.stderr
    dest = out / "_archive_eeg_defect_20261009"
    assert not any(f.exists() for f in eeg) and all(f.exists() for f in keep)
    assert (dest / "exp9_predictions" / eeg[0].name).exists()
    assert (dest / "thesis_output" / "hep_external_cis_rf5.csv").exists()
    assert (dest / "_clean_rerun_v2" / "innersplit" / "hep_eeg_s42.done").exists()
    assert (dest / "exp15_reve_quad" / "seed42_none" / "predictions_oof.json").exists()
    again = run(["archive-eeg"], **env)
    assert again.returncode == 2 and "already made" in again.stderr


def test_archive_eeg_refuses_without_the_version2_cache(tmp_path):
    out = tmp_path / "outputs"
    (out / "exp9_predictions").mkdir(parents=True)
    res = run(["archive-eeg"], DRY_RUN=1, OUT=out, ASM_ANALYSIS_OUTPUT_DIR=tmp_path / "t")
    assert res.returncode == 2 and "build the version-2 caches first" in res.stderr


def test_pending_array_reports_nothing_pending(tmp_path):
    out = tmp_path / "outputs"
    (out / "_clean_rerun_v2" / "refit").mkdir(parents=True)
    for item in run(["list"]).stdout.split():
        task, seed = item.split(":")
        (out / "_clean_rerun_v2" / "refit" / f"{task}_s{seed}.done").touch()
    res = run(["pending-array"], OUT=out)
    assert res.returncode == 1 and "nothing pending" in res.stderr and res.stdout.strip() == ""
