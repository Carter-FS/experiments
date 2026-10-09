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
