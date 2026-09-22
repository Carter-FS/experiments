"""Serialise a patient's harmonised clinical features as a fixed English paragraph (exp19).

Follows the research group's second-regimen template (Duong Nhu), cut back to
what is known when the first ASM is chosen: demographics, pre-treatment
seizure burden, seizure type, the ten history items, CT/MRI and EEG findings,
and (optionally) the first ASM by name. The template's later sentences
(first-regimen outcome, reason for change, time to failure, later regimens)
are the label or post-treatment information and are omitted; its dose
sentence is omitted because neither cohort records dose.

The text depends only on the harmonised feature values (shared.hep_cohort
codes both cohorts identically), so the same patient profile yields the same
words in either cohort. Age is written in whole years because HEP1 only
records whole years. Missing values use one fixed wording; ``fit_fills`` /
``impute`` fill them instead, the way the paper's tabular preprocessor does.
``serialise_frame`` validates its input and raises on anything outside the
harmonised coding. See docs/analysis_plan_clean_rerun_exp18.md, Addendum A
and A.1.
"""

from __future__ import annotations

import hashlib
import math
import re

import pandas as pd

TEMPLATE_VERSION = "exp19-v1.1"

ASM_NAMES = {
    "LEV": "Levetiracetam", "VPA": "Valproate", "CBZ": "Carbamazepine",
    "LTG": "Lamotrigine", "PTN": "Phenytoin", "TPM": "Topiramate",
    "OXC": "Oxcarbazepine", "ZNS": "Zonisamide", "LAC": "Lacosamide",
    "BRV": "Brivaracetam", "GBP": "Gabapentin", "PGB": "Pregabalin",
    "CLB": "Clobazam", "CLZ": "Clonazepam", "PER": "Perampanel",
}
# Other spellings of the same drugs (shared/cohort.py codes and full names).
ASM_ALIASES = {"LCM": "LAC", "CZP": "CLZ", "PHT": "PTN", "VALPROIC ACID": "VPA", "SODIUM VALPROATE": "VPA",
               **{name.upper(): code for code, name in ASM_NAMES.items()}}

# (column, phrase when absent, phrase when present, phrase when unknown), in the template's order.
HISTORY = [
    ("fam_hx", "no family history of epilepsy", "a family history of epilepsy",
     "an unknown family history of epilepsy"),
] + [
    (col, f"no history of {item}", f"a history of {item}", f"an unknown history of {item}")
    for col, item in [
        ("febrile", "febrile seizure"), ("ci", "cerebral infection"), ("birth_t", "birth trauma"),
        ("head", "head injury"), ("drug", "drug abuse"), ("alcohol", "alcohol abuse"),
        ("cvd", "cerebrovascular disease"), ("psy", "psychiatric comorbidities"),
        ("ld", "learning disability"),
    ]
]
IMAGING = {1.0: "normal", 2.0: "abnormal but not epileptogenic", 3.0: "epileptogenic"}
EEG = {1.0: "normal", 2.0: "abnormal but not epileptiform", 3.0: "epileptiform"}

BINARY_COLS = ["sex", "pretrt_sz_5", "focal"] + [h[0] for h in HISTORY]
CATEGORICAL_COLS = ["lesion", "eeg_cat"]
REQUIRED_COLS = ["pid", "age_init", *BINARY_COLS, *CATEGORICAL_COLS, "ASM"]


def _code(value) -> float | None:
    """Harmonised numeric code, or None when missing (validated upstream)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _flag(value) -> bool | None:
    """Exactly 0 -> False, 1 -> True, anything else -> None (unknown)."""
    return {0.0: False, 1.0: True}.get(_code(value))


def asm_code(value) -> str | None:
    """Canonical ASM abbreviation (accepts known abbreviations, codes and full names)."""
    s = str(value).strip().upper() if value is not None and not pd.isna(value) else ""
    s = ASM_ALIASES.get(s, s)
    return s if s in ASM_NAMES else None


def patient_to_text(row, include_asm: bool = True) -> str:
    """The template paragraph for one patient (a mapping of harmonised features)."""
    sex = _code(row.get("sex"))
    pronoun = {1.0: "Her", 0.0: "His"}.get(sex, "Their")
    parts = [
        {1.0: "This patient is female.", 0.0: "This patient is male."}.get(sex, "The sex of this patient is unknown.")
    ]
    age = _code(row.get("age_init"))
    parts.append(f"{pronoun} age at the start of antiseizure medication treatment is "
                 + (f"{math.floor(age)}." if age is not None else "unknown."))
    sz = _flag(row.get("pretrt_sz_5"))
    parts.append("Before the commencement of any antiseizure medication treatment, "
                 + {True: "the number of seizures was greater than five.",
                    False: "the number of seizures was five or fewer."}.get(sz, "the number of seizures was unknown."))
    focal = _flag(row.get("focal"))
    parts.append("The seizure type is " + {True: "focal.", False: "generalised."}.get(focal, "unknown."))
    items = []
    for col, absent, present, unknown in HISTORY:
        flag = _flag(row.get(col))
        items.append(unknown if flag is None else (present if flag else absent))
    parts.append("The patient has " + ", ".join(items[:-1]) + ", and " + items[-1] + ".")
    lesion, eeg = _code(row.get("lesion")), _code(row.get("eeg_cat"))
    parts.append("The Computed Tomography or Magnetic Resonance Imaging findings are "
                 f"{IMAGING.get(lesion, 'unknown')}, and the Electroencephalogram findings are "
                 f"{EEG.get(eeg, 'unknown')}.")
    if include_asm:
        code = asm_code(row.get("ASM"))
        parts.append(f"The first antiseizure medication regimen used was {ASM_NAMES[code]}." if code
                     else "The first antiseizure medication regimen is unknown.")
    return " ".join(parts)


def with_report(text: str, report) -> str:
    """V2: the paragraph followed by the free-text EEG report (whitespace normalised)."""
    return f"{text}\n\nElectroencephalogram report: {clean_report(report)}"


def clean_report(report) -> str:
    """Whitespace-normalised report text; raises on a missing or blank report."""
    if report is None or (not isinstance(report, str) and pd.isna(report)):
        raise ValueError("EEG report is missing")
    text = re.sub(r"\s+", " ", str(report)).strip()
    if not text:
        raise ValueError("EEG report is blank")
    return text


def validate(df: pd.DataFrame) -> None:
    """Raise if the frame is not in the harmonised coding (no patient data in the message)."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"serialiser input lacks columns {missing}")
    problems = []
    for col in BINARY_COLS:
        v = pd.to_numeric(df[col], errors="coerce")
        bad = (df[col].notna() & v.isna()) | (v.notna() & ~v.isin([0, 1]))
        if bad.any():
            problems.append(f"{col}: {int(bad.sum())} value(s) not in {{0, 1}}")
    for col in CATEGORICAL_COLS:
        v = pd.to_numeric(df[col], errors="coerce")
        bad = (df[col].notna() & v.isna()) | (v.notna() & ~v.isin([1, 2, 3]))
        if bad.any():
            problems.append(f"{col}: {int(bad.sum())} value(s) not in {{1, 2, 3}}")
    age = pd.to_numeric(df["age_init"], errors="coerce")
    bad_age = (df["age_init"].notna() & age.isna()) | (age.notna() & ~age.between(0, 120))
    if bad_age.any():
        problems.append(f"age_init: {int(bad_age.sum())} value(s) outside 0-120")
    bad_asm = ~df["ASM"].map(lambda a: asm_code(a) is not None)
    if bad_asm.any():
        problems.append(f"ASM: {int(bad_asm.sum())} unrecognised value(s)")
    if problems:
        raise ValueError("serialiser input is not in the harmonised coding: " + "; ".join(problems))


def fit_fills(df: pd.DataFrame) -> dict:
    """Missing-value fills from training rows, as the paper's ClinicalFeaturePreprocessor
    computes them: mode for binary and categorical features, mean for age."""
    fills = {}
    for col in BINARY_COLS + CATEGORICAL_COLS:
        mode = pd.to_numeric(df[col], errors="coerce").mode()
        fills[col] = float(mode.iloc[0]) if len(mode) else (1.0 if col in CATEGORICAL_COLS else 0.0)
    fills["age_init"] = float(pd.to_numeric(df["age_init"], errors="coerce").mean())
    return fills


def impute(df: pd.DataFrame, fills: dict) -> pd.DataFrame:
    """Copy of ``df`` with missing clinical values replaced by ``fills``."""
    out = df.copy()
    for col, value in fills.items():
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(value)
    return out


def serialise_frame(df: pd.DataFrame, include_asm: bool = True) -> list[str]:
    """Validated texts for every row of a harmonised cohort frame, in row order."""
    validate(df)
    return [patient_to_text(r, include_asm=include_asm) for r in df.to_dict("records")]


def text_key(text: str) -> str:
    """Stable key for the embedding store."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
