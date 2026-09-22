"""Tests for the OOF verification gate (shared/verify_oof.py).

The cohort-count map is a list of regexes; a mis-anchored pattern fails
*silently* (never matches -> count check skipped, not a failure), so the
mapping is pinned here against the real prediction filenames every
experiment produces. Also checks the leakage detector actually fires.
"""
import json

import pytest

from shared.verify_oof import expected_for, missing_expected, verify_file

# (real prediction-file rel path -> expected unique-pid cohort). exp1-6 names
# come from each config; exp7 --mode predictions writes the bare headline file;
# exp11 config names already start "exp11_" and run_experiments prepends another,
# so its files carry a doubled prefix. Suffixes match --asm-balance.
FILENAME_COHORTS = {
    "exp1_predictions/predictions_oof_exp1a_clinicalbert_chemberta.json": 117,
    "exp1_predictions/predictions_oof_exp1b_pubmedbert_smilestrf_asmweighted.json": 117,
    "exp2_predictions/predictions_oof_exp2_simplecnn_chemberta_mlp.json": 147,
    "exp2_predictions/predictions_oof_exp2_simplecnn_smilestrf_fusemoe_asmweighted.json": 147,
    "exp3_predictions/predictions_oof_exp3a_clinicalbert_chemberta.json": 107,
    "exp3_predictions/predictions_oof_exp3b_pubmedbert_smilestrf.json": 107,
    "exp4_predictions/predictions_oof_exp4a_mlp.json": 198,
    "exp4_predictions/predictions_oof_exp4b_attention_asmweighted.json": 198,
    "exp5_predictions/predictions_oof_exp5a_chemberta.json": 198,
    "exp5_predictions/predictions_oof_exp5b_clinicalbert.json": 117,
    "exp5_predictions/predictions_oof_exp5c_eeg2vec.json": 147,
    "exp6_predictions/predictions_oof_exp6a_clinicalbert_chemberta.json": 117,
    "exp6_predictions/predictions_oof_exp6b_simplecnn_chemberta.json": 147,
    "exp7_predictions/predictions_oof.json": 107,
    "exp7_predictions/predictions_oof_asmweighted.json": 107,
    "exp11_predictions/predictions_oof_exp11_exp11_3a_clinicalbert_chemberta_trf.json": 107,
    "exp11_predictions/predictions_oof_exp11_exp11_6b_chemberta_meanmax_asmweighted.json": 147,
    "exp11_predictions/predictions_oof_exp11_exp11_7a_pubmedbert_chemberta_meanmax.json": 107,
    "exp19_predictions/predictions_oof_exp19_A_llama31_8b_mean_mlp_sp-multilabel_iv20_s42.json": 198,
    "exp19_predictions/predictions_oof_exp19_T5a-full_lr_sp-multilabel_iv20_s43.json": 198,
    "exp19_predictions/predictions_oof_exp19_B-imp_pubmedbert_mean_pca32_sp-multilabel_iv20_s43.json": 198,
    "exp19_predictions/predictions_oof_exp19_D_pubmedbert_mean_mlp_sp-multilabel_iv20_s44.json": 117,
    "exp19_predictions/predictions_oof_exp19_D-split_llama31_8b_last_mlp_sp-multilabel_iv20_s44.json": 117,
    "exp19_predictions/predictions_oof_exp19_T6a_pca32_sp-multilabel_iv20_s45.json": 117,
}


@pytest.mark.parametrize("rel,cohort", FILENAME_COHORTS.items())
def test_expected_for_matches_real_filenames(rel, cohort):
    assert expected_for(rel) == cohort, f"{rel} -> {expected_for(rel)}, want {cohort}"


def test_exp1_pattern_does_not_match_exp11():
    # exp1's pattern must not swallow an exp11 file (real doubled-prefix name).
    assert expected_for("exp11_predictions/predictions_oof_exp11_exp11_3a_x_y_z.json") == 107


def test_verify_file_flags_cross_fold_leakage(tmp_path):
    # Same pid in two folds is the exact bug the gate exists to catch.
    payload = {"folds": [{"pids": ["a", "b"]}, {"pids": ["b", "c"]}]}
    f = tmp_path / "predictions_oof_leaky.json"
    f.write_text(json.dumps(payload))
    problems = verify_file(f)
    assert any("fold" in p.lower() or "leak" in p.lower() or "twice" in p.lower()
               for p in problems), problems


def test_verify_file_passes_clean_disjoint_folds(tmp_path):
    payload = {"folds": [{"pids": [f"{k}a", f"{k}b"]} for k in range(5)]}
    f = tmp_path / "predictions_oof_clean.json"  # unmapped name -> no count check
    f.write_text(json.dumps(payload))
    assert verify_file(f) == []


def test_verify_file_flags_wrong_fold_count(tmp_path):
    payload = {"folds": [{"pids": ["a", "b"]}, {"pids": ["c", "d"]}]}
    f = tmp_path / "predictions_oof_twofold.json"
    f.write_text(json.dumps(payload))
    assert any("folds != expected" in p for p in verify_file(f))


def test_missing_expected_reports_only_absent_patterns(tmp_path):
    (tmp_path / "exp4_predictions").mkdir()
    (tmp_path / "exp4_predictions" / "predictions_oof_exp4a_mlp_sp-multilabel_iv20.json").write_text("{}")
    manifest = tmp_path / "expect.txt"
    manifest.write_text(
        "# required clean files\n"
        "exp4_predictions/predictions_oof_exp4a_mlp_sp-multilabel_iv20.json\n"
        "exp5_predictions/predictions_oof_exp5a_*_sp-multilabel_iv20.json\n"
    )
    assert missing_expected(tmp_path, manifest) == [
        "exp5_predictions/predictions_oof_exp5a_*_sp-multilabel_iv20.json"
    ]
