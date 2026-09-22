"""Configuration for Experiment 19 (Addendum A and A.1 of the analysis plan)."""

from shared.hep_cohort import EXPERIMENTS_ROOT

# Frozen encoders at pinned revisions. BERT: mask-aware mean pooling, inputs
# over 512 tokens embedded in 510-token windows and token-weighted. Llama-3.1-8B
# (base): bfloat16 on CPU, one text per pass, up to 4096 tokens; mean pooling
# excludes BOS (primary), last-token pooling secondary.
ENCODERS = {
    "pubmedbert": {"model_id": "NeuML/pubmedbert-base-embeddings", "kind": "bert", "max_len": 512,
                   "revision": "b79526d6ef3645e0df4530322e266f24c829f5ef"},
    "clinicalbert": {"model_id": "medicalai/ClinicalBERT", "kind": "bert", "max_len": 512,
                     "revision": "f7c7f65227cb311f33a79c24858d875876d478ac"},
    "llama31_8b": {"model_id": "meta-llama/Llama-3.1-8B", "kind": "decoder", "max_len": 4096,
                   "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"},
}
POOLINGS = {"bert": ("mean",), "decoder": ("mean", "last")}

ZERO_SHOT = {"model_id": "meta-llama/Llama-3.1-8B-Instruct",
             "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
             "system": "You are an experienced epileptologist.",
             "question": "Will this patient be seizure-free for at least 12 months on this first antiseizure "
                         "medication? Answer Yes or No."}

# Inputs per configuration, in order. "emb:<variant>" = text embedding from the
# chosen encoder; "rep:clinicalbert" = the report embedded by ClinicalBERT
# (T6a, whatever the encoder); "smiles" = ChemBERTa; "clinical" = the paper's
# 19-feature preprocessor; "clinical_full" = the information-matched one.
CONFIGS = {
    "A": ("emb:v1",),
    "B": ("emb:v1nodrug", "smiles"),
    "B-imp": ("emb:v1nodrugimp", "smiles"),
    "E": ("emb:v1nodrug",),
    "E-imp": ("emb:v1nodrugimp",),
    "D": ("emb:v2",),
    "D-split": ("emb:v1", "emb:rep"),
    "T4": ("clinical",),
    "T5a": ("clinical", "smiles"),
    "T6a": ("clinical", "rep:clinicalbert", "smiles"),
    "T4-full": ("clinical_full",),
    "T5a-full": ("clinical_full", "smiles"),
}
TEXT_COHORT_CONFIGS = ("D", "D-split", "T6a")   # need an EEG report (117 Melbourne / 207 HEP1)
TABULAR_CONFIGS = ("T4", "T5a", "T6a", "T4-full", "T5a-full")  # encoder-independent

# MLP = LateFusionMLP (primary); PCA32 = each embedding input reduced to 32
# fit-row principal components, then the MLP; LR = L2 logistic regression on
# the concatenated standardised inputs, C chosen on the early-stopping set.
ESTIMATORS = ("mlp", "pca32", "lr")
PCA_DIM = 32
LR_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)

SEEDS = (42, 43, 44, 45, 46)
INNER_FRAC = 0.2
MARGIN = 0.05   # primary contrast: equivalence margin on the AUC difference

EMB_DIR = EXPERIMENTS_ROOT / "outputs" / "exp19_embeddings"
PRED_DIR = EXPERIMENTS_ROOT / "outputs" / "exp19_predictions"
