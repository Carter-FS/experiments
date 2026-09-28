"""Cross-cohort duplicate audit: could any HEP1 patient also be in the Melbourne cohort?

HEP1 recruited at the Royal Melbourne Hospital (29 ``RMH####`` patients) and the
Melbourne cohort includes RMH patients, but the two cohorts use unrelated pids,
so a person in both would silently sit in train and test of the pooled exp18
CV. This scores every (HEP1 RMH, Melbourne) pair on the harmonised clinical
features plus EEG-report text similarity and writes the ranked candidates for
review by the data custodian (Duong). Only aggregate counts are printed; the
per-pair table goes to the gitignored outputs/ directory.

    python -m exp18_mixed_cohort.overlap_audit
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from shared.hep_cohort import EXPERIMENTS_ROOT, load_alfred, load_hep

OUT_DIR = EXPERIMENTS_ROOT / "outputs" / "exp18_mixed_cohort"
HISTORY = ["pretrt_sz_5", "fam_hx", "febrile", "ci", "birth_t", "head", "drug",
           "alcohol", "cvd", "psy", "ld"]
MAX_AGE_GAP = 2  # HEP1 age is whole years and may be taken at enrolment, not initiation
MAX_MISMATCH = 3
STRONG = {"age_gap": 1, "mismatch": 1, "text_sim": 0.8}


def abnormal(series: pd.Series) -> pd.Series:
    """MRI / EEG category collapsed to abnormal (code > 1). A deliberately coarse
    match that tolerates the two sites grading the same report differently; the
    models themselves one-hot all three levels (analysis plan B.3)."""
    return (pd.to_numeric(series, errors="coerce") > 1).astype(float).where(series.notna())


def main() -> None:
    mel, hep = load_alfred(), load_hep()
    hep = hep[hep["pid"].astype(str).str.startswith("RMH")].reset_index(drop=True)
    for df in (mel, hep):
        df["lesion_abn"], df["eeg_abn"] = abnormal(df["lesion"]), abnormal(df["eeg_cat"])
    feats = HISTORY + ["lesion_abn", "eeg_abn"]

    # Character n-grams tolerate the two sites' different report formatting.
    texts = pd.concat([mel["eeg_report"], hep["eeg_report"]]).fillna("").astype(str).str.lower()
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit(texts)
    sim = cosine_similarity(tfidf.transform(hep["eeg_report"].fillna("").astype(str).str.lower()),
                            tfidf.transform(mel["eeg_report"].fillna("").astype(str).str.lower()))

    rows = []
    for i, h in hep.iterrows():
        for j, m in mel.iterrows():
            if h["sex"] != m["sex"] or h["ASM"] != m["ASM"]:
                continue
            age_gap = abs(float(h["age_init"]) - float(m["age_init"]))
            if age_gap > MAX_AGE_GAP:
                continue
            a, b = h[feats].astype(float), m[feats].astype(float)
            both = a.notna() & b.notna()
            mismatch = int((a[both] != b[both]).sum())
            if mismatch > MAX_MISMATCH:
                continue
            rows.append({
                "hep_pid": str(h["pid"]), "mel_pid": str(m["pid"]),
                "mel_focal": m["focal"], "age_gap": age_gap, "mismatch": mismatch,
                "n_compared": int(both.sum()),
                "text_sim": float(sim[i, j]) if h["eeg_report"] == h["eeg_report"] and m["eeg_report"] == m["eeg_report"] else np.nan,
                "same_outcome": int(h["outcome"] == m["outcome"]),
            })
    cand = pd.DataFrame(rows).sort_values(["mismatch", "age_gap", "text_sim"], ascending=[True, True, False]) \
        if rows else pd.DataFrame(columns=["hep_pid", "mel_pid"])
    strong = cand[(cand["age_gap"] <= STRONG["age_gap"]) & (cand["mismatch"] <= STRONG["mismatch"])
                  & (cand["text_sim"] >= STRONG["text_sim"])] if rows else cand

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cand.to_csv(OUT_DIR / "overlap_candidates.csv", index=False)
    best_sim = np.nanmax(sim, axis=1) if sim.size else np.array([])
    print(f"HEP1 RMH patients: {len(hep)}; Melbourne patients: {len(mel)}")
    print(f"candidate pairs (same sex + first ASM, age gap <= {MAX_AGE_GAP}, <= {MAX_MISMATCH} "
          f"feature mismatches): {len(cand)} covering {cand['hep_pid'].nunique() if len(cand) else 0} HEP1 patients")
    print(f"strong candidates (age gap <= {STRONG['age_gap']}, <= {STRONG['mismatch']} mismatch, "
          f"report similarity >= {STRONG['text_sim']}): {len(strong)} pairs, "
          f"{strong['hep_pid'].nunique() if len(strong) else 0} HEP1 patients")
    print("best cross-cohort report similarity per HEP1 RMH patient: "
          f"median {np.nanmedian(best_sim):.2f}, max {np.nanmax(best_sim):.2f}")
    print(f"wrote {OUT_DIR / 'overlap_candidates.csv'} (patient-level; gitignored)")


if __name__ == "__main__":
    main()
