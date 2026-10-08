# ASM Outcome Prediction: Architecture Documentation

**Date:** 30 January 2026
**Dataset:** 205 patients total (varies by modality availability)
**Diagrams:** Attached as `.drawio` files (open with [app.diagrams.net](https://app.diagrams.net))

---

## Overview

Seven experiment sets predict anti-seizure medication (ASM) treatment response:

| Experiment | Inputs | Fusion Methods | Dataset |
|------------|--------|----------------|---------|
| **Exp1** | Text reports (LLM) + Drug structure (SMILES) | (a) ConcatMLP, (b) FuseMoE | 121 |
| **Exp2** | EEG signals + Drug structure (SMILES) | (a) ConcatMLP, (b) FuseMoE | 151 |
| **Exp3** | Text + EEG + SMILES (triple modality) | (a) ConcatMLP, (b) FuseMoE | 107 |
| **Exp4** | Clinical features only (baseline) | (a) MLP, (b) Attention | 205 |
| **Exp5** | Clinical + single modality | (a) +SMILES, (b) +Text, (c) +EEG | varies |
| **Exp6** | Clinical + SMILES + third modality | (a) +Text, (b) +EEG | 121/147 |
| **Exp7** | Clinical + Text + EEG + SMILES (all four) | (a) MLP, (b) MoE | 107 |
| **Exp10** | Clinical + direct LLM text | Frozen / fine-tuned encoder | varies |

---

## Experiment 1: LLM + SMILES Fusion

### Input Embeddings

| Modality | Models | Dimension |
|----------|--------|-----------|
| Text Reports | ClinicalBERT, PubMedBERT | 768 |
| Drug Structure | ChemBERTa, SMILES Transformer | 768 / 256 |

### Exp1a: ConcatMLP (~953K params)
Concatenate embeddings → 4-layer MLP classifier
**Diagram:** `exp1a_concat_mlp.drawio`

| Layer | Dims | Activation | Regularisation |
|-------|------|------------|----------------|
| FC1 | 1536→512 | ReLU | LayerNorm, Dropout(0.3) |
| FC2 | 512→256 | ReLU | LayerNorm, Dropout(0.3) |
| FC3 | 256→128 | ReLU | LayerNorm, Dropout(0.2) |
| Output | 128→2 | - | - |

### Exp1b: FuseMoE (~2.6M params)
Modality projections → learnable tokens → MoE fusion layers
**Diagram:** `exp1b_fusemoe.drawio`

| Component | Configuration |
|-----------|---------------|
| Projections | 768→256 per modality |
| Self-Attention | 4 heads, dim=256 |
| MoE Layers | 2 layers, 4 experts, top-2 routing |
| Expert FFN | 256→256→256 with GELU |

---

## Experiment 2: EEG + SMILES Fusion

### EEG Preprocessing
**Diagram:** `eeg_preprocessing.drawio`

| Parameter | Value |
|-----------|-------|
| Sampling Rate | 200 Hz |
| Bandpass | 0.1–75 Hz |
| Notch Filter | 50 Hz |
| Duration | Skip 5 min, use next 20 min |
| Windowing | 10s windows (max 120) |
| Channels | 27 (10-20 standard) |

### SimpleCNN Encoder
**Diagram:** `simplecnn_encoder.drawio`

| Layer | Channels | Kernel | Pool |
|-------|----------|--------|------|
| Conv1 | 27→64 | 7 | MaxPool(4) |
| Conv2 | 64→128 | 5 | MaxPool(4) |
| Conv3 | 128→256 | 3 | AdaptiveAvgPool(1) |

### Exp2a: EEG-MLP (~1.2M params)
SimpleCNN → Transformer aggregator (2L, 4H) → concat with SMILES → MLP
**Diagram:** `exp2a_eeg_mlp.drawio`

### Exp2b: EEG-FuseMoE (~2.8M params)
Same EEG path → cross-modal MoE fusion instead of concatenation
**Diagram:** `exp2b_eeg_fusemoe.drawio`

---

## Experiment 3: Triple Modality Fusion (LLM + EEG + SMILES)

### Input Embeddings

| Modality | Models | Dimension |
|----------|--------|-----------|
| Text Reports | ClinicalBERT, PubMedBERT | 768 |
| EEG Signals | SimpleCNN + Transformer | 256 |
| Drug Structure | ChemBERTa, SMILES Transformer | 768 / 256 |

### Exp3a: TripleMLP (~2.5M params)
Project all modalities → 256D → concatenate → MLP classifier
**Diagram:** `exp3a_triple_mlp.drawio`

| Component | Configuration |
|-----------|---------------|
| Text Projection | 768→256 + LayerNorm |
| EEG Encoder | SimpleCNN → Transformer(2L, 4H) → 256 |
| SMILES Projection | dim→256 + LayerNorm |
| Fusion | Concat (768D) → MLP |

### Exp3b: TripleFuseMoE (~4.7M params)
Learnable modality tokens → self-attention → 2× MoE layers → classifier
**Diagram:** `exp3b_triple_fusemoe.drawio`

| Component | Configuration |
|-----------|---------------|
| Modality Projections | All → 256D |
| Learnable Tokens | 3 tokens (text, EEG, SMILES) |
| Self-Attention | 4 heads, dim=256 |
| MoE Layers | 2 layers, 4 experts, top-2 routing |
| Expert FFN | 256→512→256 with GELU |

---

## Experiment 4: Clinical Features Baseline

### Input Features (19D after encoding)

| Category | Features | Encoding |
|----------|----------|----------|
| Binary (13) | sex, pretrt_sz_5, focal, fam_hx, febrile, ci, birth_t, head, drug, alcohol, cvd, psy, ld | 0/1 |
| Age bins (4) | age_init | One-hot: [0,18), [18,29), [29,46), [46,inf) (Hakeem et al. 2022) |
| Categorical (2) | lesion, eeg_cat | Binary presence: normal=0, abnormal=1 |

### Exp4a: Clinical MLP (~3.7K params)
**Diagram:** `exp4a_clinical_mlp.drawio`

| Layer | Dims | Activation | Regularisation |
|-------|------|------------|----------------|
| FC1 | 19→64 | ReLU | LayerNorm, Dropout(0.3) |
| FC2 | 64→32 | ReLU | LayerNorm, Dropout(0.3) |
| Output | 32→2 | - | - |

### Exp4b: Clinical Attention (~104K params)
**Diagram:** `exp4b_clinical_attention.drawio`

| Component | Configuration |
|-----------|---------------|
| Feature Embeddings | 19 features × 64D |
| Positional Embeddings | Learnable, 20 positions |
| CLS Token | Learnable, 64D |
| Transformer Encoder | 2 layers, 4 heads |
| Classifier | 64→32→2 |

---

## Experiment 5: Clinical + Single Modality Fusion

### Architecture Pattern (Late Fusion)

All Exp5 variants use the same late fusion architecture:
```
Clinical (19D) → Encoder → 64D ─┐
                                 ├→ Concat (128D) → Classifier → 2
Modality (xD)  → Encoder → 64D ─┘
```

### Exp5a: Clinical + SMILES
**Diagram:** `exp5a_clinical_smiles.drawio`

| Modality | Input Dim | Encoder | Output |
|----------|-----------|---------|--------|
| Clinical | 19 | Linear+ReLU+LN+Dropout | 64D |
| SMILES | 768/256 | Linear+ReLU+LN+Dropout | 64D |

### Exp5b: Clinical + Text
**Diagram:** `exp5b_clinical_llm.drawio`

| Modality | Input Dim | Encoder | Output |
|----------|-----------|---------|--------|
| Clinical | 19 | Linear+ReLU+LN+Dropout | 64D |
| Text | 768 | Linear+ReLU+LN+Dropout | 64D |

### Exp5c: Clinical + EEG
**Diagram:** `exp5c_clinical_eeg.drawio`

| Modality | Input | Encoder | Output |
|----------|-------|---------|--------|
| Clinical | 19D | Linear+ReLU+LN+Dropout | 64D |
| EEG | (windows, 27, 2000) | SimpleCNN→Transformer→MeanPool | 64D |

---

## Experiment 6: Clinical + SMILES + Third Modality Fusion

### Architecture Pattern (Triple Late Fusion)

All Exp6 variants use three-stream late fusion:
```
Clinical (19D) ──→ Encoder → 64D ─┐
                                   │
SMILES (xD) ────→ Encoder → 64D ─┼→ Concat (192D) → Classifier → 2
                                   │
Third (xD) ─────→ Encoder → 64D ─┘
```

### Exp6a: Clinical + SMILES + Text (~113K params)
**Diagram:** `exp6a_clinical_smiles_text.drawio`

| Modality | Input Dim | Encoder | Output |
|----------|-----------|---------|--------|
| Clinical | 19 | Linear+ReLU+LN+Dropout | 64D |
| SMILES | 768/256 | Linear+ReLU+LN+Dropout | 64D |
| Text | 768 | Linear+ReLU+LN+Dropout | 64D |

### Exp6b: Clinical + SMILES + EEG (~1.96M params)
**Diagram:** `exp6b_clinical_smiles_eeg.drawio`

| Modality | Input | Encoder | Output |
|----------|-------|---------|--------|
| Clinical | 19D | Linear+ReLU+LN+Dropout | 64D |
| SMILES | 768/256 | Linear+ReLU+LN+Dropout | 64D |
| EEG | (windows, 27, 2000) | SimpleCNN→Transformer→MeanPool | 64D |

---

## Experiment 7: All Four Modalities Fusion

### Architecture Pattern (Quad Late Fusion)

Combines all four available modalities:
```
Clinical (19D) ──→ Encoder → 64D ─┐
                                   │
Text (768D) ─────→ Encoder → 64D ─┼→ Concat (256D) → Classifier → 2
                                   │
EEG (windows) ───→ CNN+Trf → 64D ─┤
                                   │
SMILES (768D) ───→ Encoder → 64D ─┘
```

### Exp7a: QuadFusionMLP (1.70M params)

| Modality | Input | Encoder | Output |
|----------|-------|---------|--------|
| Clinical | 19D | Linear(19→64)+ReLU+LN+Dropout | 64D |
| Text | 768D | Linear(768→256→64)+ReLU+LN+Dropout | 64D |
| EEG | (windows, 27, 2000) | SimpleCNN→Transformer→MeanPool | 64D |
| SMILES | 768D | Linear(768→256→64)+ReLU+LN+Dropout | 64D |

| Classifier | Dims | Activation | Regularisation |
|------------|------|------------|----------------|
| FC1 | 256→64 | ReLU | LayerNorm, Dropout(0.3) |
| Output | 64→2 | - | - |

### Exp7b: QuadFusionMoE (6.26M params)

| Component | Configuration |
|-----------|---------------|
| Modality Projections | All → 256D |
| Learnable Tokens | 4 tokens (clinical, text, EEG, SMILES) |
| Self-Attention | 4 heads, dim=256 |
| MoE Layers | 2 layers, 4 experts, top-2 routing |
| Expert FFN | 256→512→256 with GELU |
| Classifier | MeanPool → 256→2 |

---

## Experiment 10: Direct LLM Text Modality

### Architecture Pattern (Late Fusion with Live LLM Inference)

Unlike Exp5b which uses pre-computed text embeddings, Exp10 runs LLM inference at training time:
```
Clinical (19D) -> ModalityEncoder -> 64D ----+
                                              |-> Concat (128D) -> Classifier -> 2
Raw Text -> Tokeniser -> LLM Encoder -> xD -> ModalityEncoder -> 64D --+
```

### LLM Encoders

| Model | HuggingFace ID | Embed Dim | Domain |
|-------|----------------|-----------|--------|
| PubMedBERT | NeuML/pubmedbert-base-embeddings | 768 | Biomedical literature |
| ClinicalBERT | medicalai/ClinicalBERT | 768 | Clinical text |
| Qwen 2.5 0.5B | Qwen/Qwen2.5-0.5B | 896 | General-purpose |

### Modes

- **Frozen**: All LLM parameters frozen, only classification head trains (feature extraction)
- **Fine-tuned**: Last 2 transformer layers unfrozen with differential LR (encoder: 2e-5, head: 1e-3)

**Diagram:** `exp10_clinical_llm.drawio`

---

## Training Configuration

| Parameter | Exp1a | Exp1b | Exp2a | Exp2b | Exp3a | Exp3b | Exp4a | Exp4b | Exp5 | Exp6 | Exp7a | Exp7b |
|-----------|-------|-------|-------|-------|-------|-------|-------|-------|------|------|-------|-------|
| Learning Rate | 1e-4 | 5e-5 | 1e-4 | 1e-4 | 1e-4 | 5e-5 | 1e-3 | 5e-4 | 1e-3 | 1e-3 | 1e-3 | 5e-4 |
| Batch Size | 16 | 16 | 8 | 8 | 8 | 8 | 16 | 8 | 16 | 16/8 | 8 | 8 |
| Max Epochs | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Early Stopping | 15 | 20 | 20 | 20 | 20 | 20 | 15 | 15 | 15 | 20 | 20 | 20 |
| Optimizer | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW |
| CV Folds | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |

---

## Results Summary

| Experiment | Description | Params | Best AUC | Best Bal Acc |
|------------|-------------|--------|----------|--------------|
| **Exp7a** | **Quad MLP (All 4 modalities)** | **2M** | **0.798** | **0.814** |
| Exp11 | Quad MLP (EEG2Vec) | ~2M | 0.791 | 0.776 |
| Exp12 | Triple FuseMoE (tuned HP) | 4.7M | 0.760 | 0.760 |
| Exp7b | Quad MoE (revised FuseMoE) | 4.7M | 0.753 | 0.754 |
| Exp7a | Quad MLP (PubMedBERT) | 2M | 0.752 | 0.766 |
| Exp11 | Triple MLP (EEG2Vec) | ~2.5M | 0.736 | 0.752 |
| Exp9 | EEG+SMILES ablation (128D) | varies | 0.730 | 0.725 |
| Exp7b | Quad MoE (PubMedBERT) | 4.7M | 0.712 | 0.716 |
| Exp6a | Clinical+SMILES+Text | 113K | 0.702 | 0.705 |
| Exp11 | Clinical+SMILES+EEG (EEG2Vec) | ~2M | 0.697 | 0.694 |
| Exp10 | Clinical+Direct LLM (fine-tuned) | varies | 0.691 | 0.723 |
| Exp5a | Clinical+SMILES | 59K | 0.689 | 0.682 |
| Exp10 | Clinical+Direct LLM (frozen) | varies | 0.689 | 0.717 |
| Exp3a | Triple MLP (SimpleCNN) | 2.5M | 0.687 | 0.713 |
| Exp13 | Clinical+Direct LLM (Qwen fine-tuned) | varies | 0.682 | 0.736 |
| Exp3b | Triple FuseMoE (revised) | 4.7M | 0.677 | 0.726 |
| Exp5b | Clinical+Text | 59K | 0.676 | 0.708 |
| Exp5c | Clinical+EEG | 1.9M | 0.675 | 0.698 |
| Exp1b | LLM+SMILES FuseMoE (revised) | 2.6M | 0.674 | 0.720 |
| Exp4a | Clinical MLP | 3.7K | 0.664 | 0.675 |
| Exp6b | Clinical+SMILES+EEG (SimpleCNN) | 1.96M | 0.647 | 0.663 |
| Exp1a | LLM+SMILES MLP | 953K | 0.641 | 0.699 |
| Exp4b | Clinical Attention | 104K | 0.636 | 0.673 |
| Exp2a | EEG+SMILES MLP | 1.2M | 0.634 | 0.699 |
| Exp2b | EEG+SMILES FuseMoE (revised) | 2.8M | 0.611 | 0.621 |

**Key finding:** Quad modality MLP (Exp7a) achieves best AUC (0.798) with ClinicalBERT + ChemBERTa + SimpleCNN. EEG2Vec does not improve quad modality (0.791) but improves triple MLP to 0.736 (Exp11). FuseMoE regression resolved via hyperparameter tuning (Exp12, AUC 0.760).

---

## Attached Diagrams

| Diagram | File |
|---------|------|
| Exp1a ConcatMLP | `exp1a_concat_mlp.drawio` |
| Exp1b FuseMoE | `exp1b_fusemoe.drawio` |
| Exp2a EEG-MLP | `exp2a_eeg_mlp.drawio` |
| Exp2b EEG-FuseMoE | `exp2b_eeg_fusemoe.drawio` |
| Exp3a Triple-MLP | `exp3a_triple_mlp.drawio` |
| Exp3b Triple-FuseMoE | `exp3b_triple_fusemoe.drawio` |
| Exp4a Clinical MLP | `exp4a_clinical_mlp.drawio` |
| Exp4b Clinical Attention | `exp4b_clinical_attention.drawio` |
| Exp5a Clinical+SMILES | `exp5a_clinical_smiles.drawio` |
| Exp5b Clinical+Text | `exp5b_clinical_llm.drawio` |
| Exp5c Clinical+EEG | `exp5c_clinical_eeg.drawio` |
| Exp6a Clinical+SMILES+Text | `exp6a_clinical_smiles_text.drawio` |
| Exp6b Clinical+SMILES+EEG | `exp6b_clinical_smiles_eeg.drawio` |
| EEG Preprocessing | `eeg_preprocessing.drawio` |
| SimpleCNN Encoder | `simplecnn_encoder.drawio` |
| EEGNet Encoder | `exp9_eegnet_encoder.drawio` |
| LaBraM Encoder | `exp9_labram_encoder.drawio` |
| EEG2Vec Encoder | `exp9_eeg2vec_encoder.drawio` |
| Exp9 Ablation Framework | `exp9_ablation_framework.drawio` |
| Exp10 Clinical+LLM | `exp10_clinical_llm.drawio` |
