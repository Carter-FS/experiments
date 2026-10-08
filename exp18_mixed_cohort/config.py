"""Configuration for Experiment 18: mixed-cohort (Melbourne + HEP1) training.

Frozen design: docs/analysis_plan_clean_rerun_exp18.md, section 6. The six
configurations are the HEP1 Table 3 rows on the cohort-portable pipeline
(ChemBERTa drug, mean-pooled ClinicalBERT text, 19-channel EEG2Vec), trained
with the same loops and hyperparameters as the external-validation scripts
(shared/portable_models.py).
"""

from shared.hep_cohort import EXPERIMENTS_ROOT

CONFIGS_NON_EEG = ("Exp4a", "Exp5a", "Exp5b")
CONFIGS_EEG = ("Exp5c", "Exp6b", "Exp7a")

# exp19 additions (analysis plan Addendum A.1): serialised-text configurations
# A and D per encoder, and same-class tabular comparators, all
# shared.portable_models.LateFusionMLP. Input names follow exp19's CONFIGS;
# "emb:<variant>:<encoder>" is looked up in the exp19 embedding store. The
# text omits the facts constant in HEP1 (v1xc, plan B.4) and D is the
# segment-balanced mean of paragraph and report (plan B.7).
LF_INPUTS = {}
for _enc in ("pubmedbert", "clinicalbert", "llama31_8b"):
    LF_INPUTS[f"S19A_{_enc}"] = (f"emb:v1xc:{_enc}",)
    LF_INPUTS[f"S19D_{_enc}"] = (f"emb:v1xc|rep:{_enc}",)
LF_INPUTS["LF_T5a-full"] = ("clinical_full", "smiles")
LF_INPUTS["LF_T6a"] = ("clinical", "rep:clinicalbert", "smiles")
CONFIGS_LF = tuple(LF_INPUTS)
CONFIGS = CONFIGS_NON_EEG + CONFIGS_EEG + CONFIGS_LF

# Which cached modalities each configuration consumes besides clinical features.
MODALITIES = {
    "Exp4a": (),
    "Exp5a": ("smiles",),
    "Exp5b": ("text",),
    "Exp5c": ("eeg",),
    "Exp6b": ("smiles", "eeg"),
    "Exp7a": ("text", "smiles", "eeg"),
}
# "text" restricts the pooled cohort to patients with a usable EEG report.
def _needs_report(name: str) -> bool:
    return name.startswith("rep:") or (name.startswith("emb:") and
                                       bool({"v2", "rep"} & set(name.split(":")[1].split("|"))))


for _cfg, _inputs in LF_INPUTS.items():
    MODALITIES[_cfg] = tuple(
        (["text"] if any(_needs_report(n) for n in _inputs) else [])
        + (["smiles"] if "smiles" in _inputs else []))

# Portable model per configuration (shared/portable_models.py). Exp6b uses the
# pre-specified EEG2Vec encoder rather than the published SimpleCNN one.
PORTABLE_MODEL = {cfg: cfg for cfg in CONFIGS}
PORTABLE_MODEL["Exp6b"] = "Exp6b_eeg2vec"
PORTABLE_MODEL.update({cfg: f"LF:{cfg}" for cfg in CONFIGS_LF})

SEEDS = {cfg: (42, 43, 44, 45, 46) for cfg in CONFIGS_NON_EEG}
SEEDS.update({cfg: (42, 43, 44) for cfg in CONFIGS_EEG})
SEEDS.update({cfg: (42, 43, 44, 45, 46) for cfg in CONFIGS_LF})

# Every arm scores every outer test patient in both cohorts.
ARMS = ("mixed", "mel_only", "hep_only")
ARM_COHORTS = {"mixed": ("MEL", "HEP"), "mel_only": ("MEL",), "hep_only": ("HEP",)}

# Size-matched mixed training (Exp4a only): per test cohort, 10 draws.
SIZEMATCH_CONFIGS = ("Exp4a",)
SIZEMATCH_DRAWS = 10

N_SPLITS = 5
INNER_FRAC = 0.2
STRATIFY = ["outcome", "cohort"]

# Sensitivity analysis: drop every HEP1 patient recruited at the Royal
# Melbourne Hospital, whatever the duplicate audit finds.
RMH_PREFIX = "RMH"
SENSITIVITY_CONFIGS = ("Exp4a", "Exp5a", "Exp5b")

OUT_DIR = EXPERIMENTS_ROOT / "outputs" / "exp18_mixed_cohort"
from shared.eeg_cache import CACHE_PATHS as _EEG_CACHE_PATHS  # noqa: E402  version-2 caches

MEL_EEG_CACHE = _EEG_CACHE_PATHS["alfred"]
HEP_EEG_CACHE = _EEG_CACHE_PATHS["hep"]
