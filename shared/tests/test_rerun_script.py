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
           thesis / "cis_tier1.csv",
           thesis / "all_pairs_stats.csv",
           thesis / "best_asm_simulation_summary.json",
           thesis / "clinical_reliability_ranking.csv",
           thesis / "stage_b_comparison.csv",
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
            thesis / "hep_external_predictions_sp-multilabel_rf5_s42_h12.csv",
            thesis / "hep_reverse_summary_sp-multilabel_rf5_s42.csv",
            thesis / "hep_focal_external_predictions_sp-multilabel_rf5_s42.csv",
            thesis / "hep_external_oov_breakdown.csv",
            thesis / "metrics_decomposition.csv",
            thesis / "eeg_report_keywords.csv",
            thesis / "asm_first_prescription_counts.csv",
            out / "_clean_rerun_v2" / "refit" / "exp4_s42.done"]
    for f in eeg + keep:
        f.touch()
    old = time.time() - 3600
    for f in eeg + keep:
        os.utime(f, (old, old))
    guard = out / "eeg_cache" / "eeg19_v2_alfred.pkl.meta.json"
    guard.write_text("{}")                      # newer than every file above
    return out, thesis, eeg, keep, guard


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
    assert (dest / ".complete").exists()
    assert not any(f.exists() for f in eeg) and all(f.exists() for f in keep)
    assert (dest / "exp9_predictions" / eeg[0].name).exists()
    assert (dest / "thesis_output" / "hep_external_cis_rf5.csv").exists()
    assert (dest / "_clean_rerun_v2" / "innersplit" / "hep_eeg_s42.done").exists()
    assert (dest / "exp15_reve_quad" / "seed42_none" / "predictions_oof.json").exists()
    again = run(["archive-eeg"], **env)
    assert again.returncode == 2 and "already made" in again.stderr
    (dest / ".complete").unlink()                                # an interrupted archive resumes
    resumed = run(["archive-eeg"], **env)
    assert resumed.returncode == 0 and (dest / ".complete").exists()


def test_archive_eeg_halts_on_an_eeg_file_newer_than_the_cache(tmp_path):
    """A rerun output (newer than the cache sidecar) in an EEG family stops the archive
    before anything moves; ALLOW_NEWER=1 archives it with the rest."""
    import time
    out, thesis, eeg, keep, guard = _eeg_tree(tmp_path)
    env = {"OUT": out, "ASM_ANALYSIS_OUTPUT_DIR": thesis}
    newer = out / "exp9_predictions" / "predictions_oof_exp9_encoder_reve_frozen_sp-multilabel_rf5_s42.json"
    newer.touch()
    now = time.time() + 60
    os.utime(newer, (now, now))
    halted = run(["archive-eeg"], **env)
    assert halted.returncode == 2 and newer.name in halted.stderr and "nothing was moved" in halted.stderr
    assert all(f.exists() for f in eeg + [newer])
    listed = run(["archive-eeg"], DRY_RUN=1, ALLOW_NEWER=1, **env)
    assert listed.returncode == 0 and str(newer) in listed.stdout and all(str(f) in listed.stdout for f in eeg)
    assert all(f.exists() for f in eeg + [newer])


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


def test_pending_array_cpu_indexes_the_cpu_list(tmp_path):
    out = tmp_path / "outputs"
    (out / "_clean_rerun_v2" / "refit").mkdir(parents=True)
    cpu = run(["list-cpu"]).stdout.split()
    (out / "_clean_rerun_v2" / "refit" / f"{cpu[0].replace(':', '_s')}.done").touch()
    idx = [int(i) for i in run(["pending-array-cpu"], OUT=out).stdout.strip().split(",")]
    assert 0 not in idx and 1 in idx and max(idx) < len(cpu)


def test_every_thesis_script_takes_the_output_dir_from_the_helper():
    import re
    offenders = []
    for path in sorted((REPO / "thesisStandalone" / "analysis").glob("*.py")):
        if path.name == "_asm_paths.py":
            continue
        if re.search(r'REPO_ROOT / "analysis" / "output"|resolve\(\)\.parent / "output"', path.read_text()):
            offenders.append(path.name)
    assert offenders == [], offenders


def test_archive_eeg_checks_destinations_before_moving(tmp_path):
    """A destination that already holds different content stops the run before any move;
    one identical to its source (an interrupted earlier move) is treated as done."""
    out, thesis, eeg, keep, guard = _eeg_tree(tmp_path)
    env = {"OUT": out, "ASM_ANALYSIS_OUTPUT_DIR": thesis}
    dest = out / "_archive_eeg_defect_20261009"
    clash = dest / "exp9_predictions" / eeg[0].name
    clash.parent.mkdir(parents=True)
    clash.write_text("different")
    res = run(["archive-eeg"], **env)
    assert res.returncode == 1 and "different content" in res.stderr and all(f.exists() for f in eeg)
    clash.write_text(eeg[0].read_text())             # identical: a move interrupted after the copy
    res = run(["archive-eeg"], **env)
    assert res.returncode == 0, res.stderr
    assert not eeg[0].exists() and clash.exists() and (dest / ".complete").exists()
    assert not any(p.name.endswith(".partial") for p in dest.rglob("*"))
    (thesis / ".gitkeep").touch()
    assert ".gitkeep" not in run(["archive-eeg"], DRY_RUN=1, OUT=out, ASM_ANALYSIS_OUTPUT_DIR=thesis).stdout


def test_stale_markers_lists_eeg_markers_older_than_the_cache(tmp_path):
    out, thesis, eeg, keep, guard = _eeg_tree(tmp_path)
    listed = run(["stale-markers"], OUT=out).stdout.split()
    names = sorted(Path(f).name for f in listed)
    assert names == ["exp5_s42.done", "exp9_encoder_eeg2vec_s42.done", "hep_eeg_s42.done"]   # not exp4
    run(["archive-eeg"], OUT=out, ASM_ANALYSIS_OUTPUT_DIR=thesis)
    assert run(["stale-markers"], OUT=out).stdout.strip() == ""
