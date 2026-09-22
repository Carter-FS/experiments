"""Tests for the exp19 clinical-text serialiser (shared/serialise_clinical.py)."""
import itertools

import numpy as np
import pandas as pd
import pytest

from shared.serialise_clinical import (
    ASM_NAMES, BINARY_COLS, asm_code, clean_report, fit_fills, impute, patient_to_text, serialise_frame,
    text_key, validate, with_report,
)

BASE = {"pid": "p1", "sex": 1.0, "age_init": 21.7, "pretrt_sz_5": 1.0, "focal": 1.0, "fam_hx": 0.0,
        "febrile": 0.0, "ci": 0.0, "birth_t": 0.0, "head": 0.0, "drug": 0.0, "alcohol": 0.0, "cvd": 0.0,
        "psy": 1.0, "ld": 0.0, "lesion": 2.0, "eeg_cat": 1.0, "ASM": "CBZ"}

# Anything describing treatment response, dosing, dates or later regimens would leak the label.
FORBIDDEN = ["outcome", "failure", "fail", "success", "seizure-free", "seizure free", "remission", "respon",
             "adverse", "reason", "dose", "mg", "second", "substitution", "combination", "month", "ongoing",
             "date", "stop", "switch", "withdraw", "tolerat"]


def test_matches_the_group_template_wording():
    assert patient_to_text(BASE) == (
        "This patient is female. Her age at the start of antiseizure medication treatment is 21. "
        "Before the commencement of any antiseizure medication treatment, the number of seizures was "
        "greater than five. The seizure type is focal. The patient has no family history of epilepsy, "
        "no history of febrile seizure, no history of cerebral infection, no history of birth trauma, "
        "no history of head injury, no history of drug abuse, no history of alcohol abuse, no history of "
        "cerebrovascular disease, a history of psychiatric comorbidities, and no history of learning "
        "disability. The Computed Tomography or Magnetic Resonance Imaging findings are abnormal but not "
        "epileptogenic, and the Electroencephalogram findings are normal. The first antiseizure medication "
        "regimen used was Carbamazepine."
    )


def test_every_imaging_and_eeg_category_has_its_own_phrase():
    for lesion, phrase in [(1, "findings are normal, and"), (2, "findings are abnormal but not epileptogenic, and"),
                           (3, "findings are epileptogenic, and")]:
        assert phrase in patient_to_text({**BASE, "lesion": lesion})
    for eeg, phrase in [(1, "Electroencephalogram findings are normal."),
                        (2, "Electroencephalogram findings are abnormal but not epileptiform."),
                        (3, "Electroencephalogram findings are epileptiform.")]:
        assert phrase in patient_to_text({**BASE, "eeg_cat": eeg})


@pytest.mark.parametrize("sex,focal,sz,lesion,eeg", list(itertools.product(
    [0, 1, np.nan], [0, 1, np.nan], [0, 1, np.nan], [1, 2, 3, np.nan], [1, 2, 3, np.nan])))
def test_no_label_or_post_treatment_wording_in_any_branch(sex, focal, sz, lesion, eeg):
    for asm, include in [("LEV", True), ("OXC", True), ("LEV", False), (np.nan, True)]:
        row = {**BASE, "sex": sex, "focal": focal, "pretrt_sz_5": sz, "lesion": lesion, "eeg_cat": eeg,
               "psy": np.nan, "ASM": asm}
        text = patient_to_text(row, include_asm=include).lower()
        assert not [w for w in FORBIDDEN if w in text], text


def test_outcome_and_date_columns_never_reach_the_text():
    row = {**BASE, **{c: "SENTINEL" for c in ["outcome", "start_date", "end_date", "ongoing", "regimen",
                                              "eeg_report", "mri_report"]}}
    assert "SENTINEL" not in patient_to_text(row)
    assert "SENTINEL" not in serialise_frame(pd.DataFrame([row]))[0]


def test_drug_sentence_only_when_requested():
    assert "Carbamazepine" in patient_to_text(BASE)
    no_drug = patient_to_text(BASE, include_asm=False)
    assert "Carbamazepine" not in no_drug and "medication regimen" not in no_drug


def test_asm_aliases_and_names():
    assert asm_code("lcm") == "LAC" and asm_code(" Levetiracetam ") == "LEV" and asm_code("CZP") == "CLZ"
    assert asm_code("NOT_A_DRUG") is None and asm_code(np.nan) is None
    assert "Perampanel" in patient_to_text({**BASE, "ASM": "PER"})
    for code, name in ASM_NAMES.items():
        assert f"used was {name}." in patient_to_text({**BASE, "ASM": code})


def test_same_values_same_text_whatever_the_encoding():
    as_int = {k: (int(v) if isinstance(v, float) else v) for k, v in BASE.items()}
    as_str = {k: str(v) for k, v in BASE.items()}
    assert patient_to_text(BASE) == patient_to_text(as_int) == patient_to_text(as_str)


def test_age_is_whole_years():
    assert "is 21." in patient_to_text({**BASE, "age_init": 21.99})
    assert "is 40." in patient_to_text({**BASE, "age_init": 40})


def test_flags_are_exact():
    # Values other than 0/1 are "unknown" in patient_to_text and rejected by validate.
    for bad in (0.5, 2, -1):
        assert "an unknown history of cerebral infection" in patient_to_text({**BASE, "ci": bad})
        with pytest.raises(ValueError, match="ci"):
            validate(pd.DataFrame([{**BASE, "ci": bad}]))


def test_missing_values_have_fixed_wording():
    row = {k: (np.nan if k != "pid" else "p") for k in BASE}
    text = patient_to_text(row)
    assert text.startswith("The sex of this patient is unknown. Their age")
    assert "the number of seizures was unknown" in text and "The seizure type is unknown." in text
    assert "an unknown family history of epilepsy" in text and "an unknown history of learning disability" in text
    assert "findings are unknown, and the Electroencephalogram findings are unknown." in text
    assert text.endswith("The first antiseizure medication regimen is unknown.")


def test_generalised_male_and_categories():
    text = patient_to_text({**BASE, "sex": 0, "focal": 0, "pretrt_sz_5": 0, "lesion": 3, "eeg_cat": 3})
    assert text.startswith("This patient is male. His age")
    assert "five or fewer" in text and "The seizure type is generalised." in text


@pytest.mark.parametrize("change,match", [
    ({"lesion": 4}, "lesion"), ({"eeg_cat": 0}, "eeg_cat"), ({"sex": "Female"}, "sex"),
    ({"age_init": -3}, "age_init"), ({"age_init": 200}, "age_init"), ({"ASM": "ASPIRIN"}, "ASM"),
    ({"febrile": "?"}, "febrile"),
])
def test_validate_rejects_uncoded_input(change, match):
    with pytest.raises(ValueError, match=match):
        serialise_frame(pd.DataFrame([{**BASE, **change}]))


def test_validate_rejects_missing_columns_and_accepts_nan():
    with pytest.raises(ValueError, match="lacks columns"):
        validate(pd.DataFrame([{k: v for k, v in BASE.items() if k != "focal"}]))
    validate(pd.DataFrame([{**BASE, "focal": np.nan, "lesion": np.nan, "age_init": np.nan}]))


def test_reports():
    v2 = with_report(patient_to_text(BASE), "  Normal awake\n EEG.  ")
    assert v2.endswith("\n\nElectroencephalogram report: Normal awake EEG.")
    for bad in (np.nan, None, "", "   \n "):
        with pytest.raises(ValueError):
            clean_report(bad)


def test_fills_mirror_the_tabular_preprocessor_and_remove_unknowns():
    df = pd.DataFrame([{**BASE, "pid": "a", "lesion": 3, "age_init": 30}, {**BASE, "pid": "b", "lesion": 3, "age_init": 40},
                       {**BASE, "pid": "c", "lesion": 1, "age_init": 50, "febrile": np.nan}])
    fills = fit_fills(df)
    assert fills["lesion"] == 3.0 and fills["febrile"] == 0.0 and fills["age_init"] == pytest.approx(40.0)
    missing = pd.DataFrame([{**BASE, "pid": "m", **{c: np.nan for c in BINARY_COLS + ["lesion", "eeg_cat", "age_init"]}}])
    text = serialise_frame(impute(missing, fills))[0]
    assert "unknown" not in text and "is 40." in text and "findings are epileptogenic" in text


def test_serialise_frame_keeps_row_order_and_text_keys_are_stable():
    df = pd.DataFrame([{**BASE, "pid": "b"}, {**BASE, "pid": "a", "ASM": "LEV"}])
    texts = serialise_frame(df)
    assert "Carbamazepine" in texts[0] and "Levetiracetam" in texts[1]
    assert text_key(texts[0]) == text_key(patient_to_text(BASE)) and text_key(texts[0]) != text_key(texts[1])


def test_real_cohorts_validate_and_stay_clean():
    try:
        from shared.hep_cohort import load_alfred, load_hep
        cohorts = [load_alfred(), load_hep()]
    except (FileNotFoundError, OSError):
        pytest.skip("cohort data not available on this host")
    for df in cohorts:
        texts = serialise_frame(df)
        assert len(texts) == len(df)
        assert not [t for t in texts if any(w in t.lower() for w in FORBIDDEN)]
        imputed = serialise_frame(impute(df, fit_fills(cohorts[0])))
        assert not [t for t in imputed if "unknown" in t]
