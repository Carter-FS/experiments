"""Dry-run (_smoke) files are never taken for results: the pooled-metrics loader and the
verify gate ignore them, and the loader refuses to parse one."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _thesis():
    root = str(REPO / "thesisStandalone")
    if root not in sys.path:
        sys.path.insert(0, root)
    import analysis.metrics_from_predictions as m
    return m


def test_parser_refuses_a_smoke_file():
    m = _thesis()
    assert m.parse_config_mode(Path("predictions_oof_exp9_encoder_eegnet_sp-multilabel_rf5_s42.json"))[2] == "sp-multilabel_rf5"
    with pytest.raises(ValueError, match="smoke-run file"):
        m.parse_config_mode(Path("predictions_oof_exp9_encoder_eegnet_sp-multilabel_rf5_s42_smoke.json"))


def test_collect_skips_smoke_files(tmp_path):
    m = _thesis()
    d = tmp_path / "exp9_predictions"
    d.mkdir()
    (d / "predictions_oof_exp9_encoder_eegnet_sp-multilabel_rf5_s42_smoke.json").write_text(json.dumps({"folds": []}))
    assert m.collect(tmp_path) == []


def test_verify_oof_ignores_smoke_files(tmp_path, capsys):
    from shared import verify_oof as v
    d = tmp_path / "exp9_predictions"
    d.mkdir()
    (d / "predictions_oof_exp9_encoder_eegnet_sp-multilabel_rf5_s42_smoke.json").write_text(json.dumps({"folds": []}))
    assert v.is_smoke_file(d / "predictions_oof_x_smoke.json") and not v.is_smoke_file(d / "predictions_oof_x_s42.json")
    assert v.main(["verify_oof", str(tmp_path)]) == 1          # nothing real to verify
    err = capsys.readouterr().err
    assert "1 smoke-run file(s) ignored" in err and "no prediction files" in err


def test_thesis_output_dir_override_and_result_files(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / "thesisStandalone" / "analysis"))
    import _asm_paths as ap
    monkeypatch.setenv("ASM_ANALYSIS_OUTPUT_DIR", str(tmp_path / "thesis_output"))
    out = ap.analysis_output_dir()
    assert out == tmp_path / "thesis_output" and out.is_dir()
    (out / "hep_external_summary_sp-multilabel_rf5_s42.csv").write_text("a\n")
    (out / "hep_external_summary_sp-multilabel_rf5_s42_smoke.csv").write_text("a\n")
    assert [p.name for p in ap.result_files("hep_external_summary_sp-multilabel_rf5_s*.csv")] == \
        ["hep_external_summary_sp-multilabel_rf5_s42.csv"]
    assert ap.is_smoke_file("x_rf5_s42_smoke.csv") and not ap.is_smoke_file("x_rf5_s42.csv")
    monkeypatch.delenv("ASM_ANALYSIS_OUTPUT_DIR")
    assert ap.analysis_output_dir() == REPO / "thesisStandalone" / "analysis" / "output"


def test_hep_scripts_write_under_the_overridable_output_dir():
    """Every thesis script that writes result CSVs names them through analysis_output_dir()."""
    import re
    for name in ("hep_external_validation.py", "hep_external_validation_eeg.py", "hep_reduced_external_validation.py",
                 "hep_reverse_validation.py", "hep_focal_external_validation.py"):
        src = (REPO / "thesisStandalone" / "analysis" / name).read_text()
        assert "from _asm_paths import analysis_output_dir" in src, name
        assert not re.search(r'REPO_ROOT / "analysis" / "output"', src), f"{name}: output path bypasses analysis_output_dir()"


def test_thesis_result_globs_skip_smoke_files():
    """A glob over seed-tagged result files must exclude dry runs."""
    import re
    offenders = []
    for path in sorted((REPO / "thesisStandalone" / "analysis").glob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"\.glob\(|glob\.glob\(", line) and re.search(r"_s\*|predictions_oof", line):
                if not re.search(r"result_files\(|_smoke", line):
                    offenders.append(f"{path.name}:{i}")
    assert offenders == [], offenders


def test_the_two_smoke_file_predicates_agree():
    """shared.verify_oof and the thesis _asm_paths carry the same predicate (separate repos)."""
    sys.path.insert(0, str(REPO / "thesisStandalone" / "analysis"))
    import _asm_paths as ap
    from shared import verify_oof as v
    for name in ("predictions_oof_x_sp-multilabel_rf5_s42.json", "predictions_oof_x_sp-multilabel_rf5_s42_smoke.json",
                 "hep_external_summary_sp-multilabel_rf5_s42_smoke.csv", "ablation_results_20261009_1_smoke.json", "smokeless.csv"):
        assert v.is_smoke_file(Path(name)) == ap.is_smoke_file(Path(name)) == ("_smoke" in Path(name).stem)


def test_verify_gate_requires_version2_provenance_on_eeg_files():
    from shared import verify_oof as v
    with_cache = {"metadata": {"eeg_inputs": [{"kind": "cache", "version": 2}]}}
    with_features = {"metadata": {"eeg_inputs": [{"kind": "features", "source_cache_version": 2}]}}
    legacy = {"metadata": {"eeg_inputs": [{"kind": "cache", "version": "legacy"}]}}
    none = {"metadata": {"protocol": "refit"}}
    eeg = "exp9_predictions/predictions_oof_exp9_encoder_eegnet_sp-multilabel_rf5_s42.json"
    assert v.eeg_provenance_problem(eeg, with_cache) is None and v.eeg_provenance_problem(eeg, with_features) is None
    assert "superseded" in v.eeg_provenance_problem(eeg, none)
    assert "no version-2" in v.eeg_provenance_problem(eeg, legacy)
    assert v.eeg_provenance_problem("exp5_predictions/predictions_oof_exp5c_eeg2vec_sp-multilabel_rf5_s42.json", none)
    assert v.eeg_provenance_problem("exp6_predictions/predictions_oof_exp6b_simplecnn_sp-multilabel_rf5_s42.json", none)
    for non_eeg in ("exp4_predictions/predictions_oof_exp4a_mlp_s42.json", "exp5_predictions/predictions_oof_exp5a_chemberta_s42.json",
                    "exp6_predictions/predictions_oof_exp6a_clinicalbert_chemberta_s42.json", "exp1_predictions/predictions_oof_exp1a_x.json",
                    "exp19_predictions/predictions_oof_exp19_A_x.json"):
        assert v.eeg_provenance_problem(non_eeg, none) is None, non_eeg
