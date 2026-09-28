"""Configuration for Experiment 19 (Addendum A and A.1 of the analysis plan)."""

from shared.hep_cohort import EXPERIMENTS_ROOT

# Frozen encoders at pinned revisions. BERT: mask-aware mean pooling, inputs
# over 512 tokens embedded in 510-token windows and token-weighted. Llama-3.1-8B
# (base): bfloat16 on CPU, one text per pass, up to 4096 tokens; mean pooling
# excludes BOS (primary), last-token pooling secondary. Qwen3-Embedding-8B
# (Addendum B.7, secondary, outside the primary Holm family): an
# embedding-tuned decoder as in Hegselmann et al. 2025; the model card's
# instruction prefix, final (end-of-text) token's state, L2 normalised.
ENCODERS = {
    "pubmedbert": {"model_id": "NeuML/pubmedbert-base-embeddings", "kind": "bert", "max_len": 512,
                   "revision": "b79526d6ef3645e0df4530322e266f24c829f5ef"},
    "clinicalbert": {"model_id": "medicalai/ClinicalBERT", "kind": "bert", "max_len": 512,
                     "revision": "f7c7f65227cb311f33a79c24858d875876d478ac"},
    "llama31_8b": {"model_id": "meta-llama/Llama-3.1-8B", "kind": "decoder", "max_len": 4096,
                   "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"},
    "qwen3_embed_8b": {"model_id": "Qwen/Qwen3-Embedding-8B", "kind": "embed_lasttok", "max_len": 4096,
                       "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"},
}
POOLINGS = {"bert": ("mean",), "decoder": ("mean", "last"), "embed_lasttok": ("last",)}
QWEN_INSTRUCTION = ("Instruct: Represent this patient summary for predicting seizure freedom on this "
                    "medication\nQuery: ")

ZERO_SHOT = {"model_id": "meta-llama/Llama-3.1-8B-Instruct",
             "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
             "system": "You are an experienced epileptologist.",
             "question": "Will this patient be seizure-free for at least 12 months on this first antiseizure "
                         "medication? Answer Yes or No."}

# Inputs per configuration, in order. "emb:<variant>" = text embedding from the
# chosen encoder; "rep:clinicalbert" = the report embedded by ClinicalBERT
# (T6a, whatever the encoder); "smiles" = ChemBERTa; "clinical" = the paper's
# preprocessor (23 inputs, plan B.3); "clinical_full" = the information-matched
# one. "emb:a|b" is the segment-balanced mean of the two variants' embeddings,
# each embedded on its own (D, plan B.7); D-tok keeps the token mean over the
# joined text.
CONFIGS = {
    "A": ("emb:v1",),
    "B": ("emb:v1nodrug", "smiles"),
    "B-imp": ("emb:v1nodrugimp", "smiles"),
    "E": ("emb:v1nodrug",),
    "E-imp": ("emb:v1nodrugimp",),
    "D": ("emb:v1|rep",),
    "D-tok": ("emb:v2",),
    "D-split": ("emb:v1", "emb:rep"),
    "T4": ("clinical",),
    "T5a": ("clinical", "smiles"),
    "T6a": ("clinical", "rep:clinicalbert", "smiles"),
    "T4-full": ("clinical_full",),
    "T5a-full": ("clinical_full", "smiles"),
}
TEXT_COHORT_CONFIGS = ("D", "D-tok", "D-split", "T6a")   # need an EEG report (117 Melbourne / 207 HEP1)
TABULAR_CONFIGS = ("T4", "T5a", "T6a", "T4-full", "T5a-full")  # encoder-independent

# MLP = LateFusionMLP (primary); PCA32 = each embedding input reduced to 32
# fit-row principal components, then the MLP; LR = L2 logistic regression on
# the concatenated standardised inputs, C chosen on the early-stopping set (or,
# under the refit protocol, by pooled inner-fold AUC, then refitted; plan B.2).
ESTIMATORS = ("mlp", "pca32", "lr")
PCA_DIM = 32
LR_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)

SEEDS = (42, 43, 44, 45, 46)
INNER_FRAC = 0.2
MARGIN = 0.05   # primary contrast: equivalence margin on the AUC difference

EMB_DIR = EXPERIMENTS_ROOT / "outputs" / "exp19_embeddings"
PRED_DIR = EXPERIMENTS_ROOT / "outputs" / "exp19_predictions"
