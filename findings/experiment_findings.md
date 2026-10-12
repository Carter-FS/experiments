# ASM Outcome Prediction: Experimental Findings

**Date:** 18 February 2026 (updated); EEG rerun section added 12 October 2026
**Dataset:** 151 patients with EEG recordings and anti-seizure medication (ASM) outcomes

---

## EEG Rerun on Corrected Inputs (12 October 2026)

Every EEG-dependent result below this section was produced with the superseded
27-channel, unnormalised EEG cache and is kept for the record only. The numbers here
come from the version-2 cache (19 standard channels, microvolts, per-window z-score;
`findings/training_configurations.md` Section 1) under the refit protocol, as the mean
over five repeated cross-validations with the mean of the per-repeat Knapp-Hartung 95%
CIs (`thesisStandalone/analysis/output/metrics_oof.csv`). Non-EEG results are
unchanged.

### Standalone EEG encoder comparison (exp9; EEG plus ChemBERTa drug embedding, n = 148)

| Encoder | Pretrained | AUC (95% CI) |
|---|---|---|
| SimpleCNN | no | 0.548 (0.41-0.68) |
| EEGNet | no | 0.495 (0.35-0.64) |
| EEG2Vec | no | 0.489 (0.35-0.63) |
| LaBraM architecture, from scratch | no | 0.495 (0.35-0.64) |
| LaBraM-base, frozen features | yes | 0.482 (0.34-0.62) |
| REVE-base, frozen features | yes | 0.497 (0.35-0.65) |

### Pre-specified configurations (Table 2 of the paper)

| Configuration | n | AUC (95% CI) | Previous (superseded) |
|---|---|---|---|
| Clinical + EEG (no drug), exp5c | 148 | 0.556 (0.41-0.71) | 0.581 |
| Clinical + EEG, exp6b | 148 | 0.588 (0.42-0.76) | 0.590 |
| Full model, exp7a | 108 | 0.605 (0.44-0.77) | 0.585 |
| EEG, exp2 EEG2Vec MLP | 148 | 0.474 (0.33-0.62) | 0.452 |
| Text + EEG, exp3a EEG2Vec | 108 | 0.503 (0.32-0.68) | 0.463 |
| Full model, mixture of experts, exp7b | 108 | 0.551 (0.36-0.74) | 0.537 |
| Full model with frozen REVE-base features, exp15 | 108 | 0.575 (0.40-0.75) | 0.584 |
| Full model with frozen LaBraM-base features, exp15 | 108 | 0.578 (0.42-0.74) | not run |
| Reduced capacity small / tiny, exp16 | 108 | 0.677 (0.52-0.83) / 0.709 (0.55-0.86) | 0.595 / 0.571 |
| Focal-only full model, exp17 | 83 | 0.685 (0.47-0.90) | 0.629 |

The SimpleCNN rows moved most (for example exp2 SimpleCNN with the SMILES Transformer
0.438 to 0.615), consistent with BatchNorm having been starved by the volt-scale inputs.
HEP1 external AUCs (Melbourne-trained, five-seed mean): Clinical + EEG (no drug) 0.536,
Clinical + EEG 0.568, full model 0.575, reduced-capacity tiny 0.533, every interval
including 0.5. Deferred and not rerun: the exp9 aggregator and embedding-size
ablations, the exp11 grid, the inner-split protocol for the EEG items, the
harmonised-label EEG items and the fine-tuned LaBraM-base arm of plan decision D4
(`rerun_clean.sh` deferred list; Addendum C.5 and C.6).

---

## Executive Summary

We evaluated multimodal fusion approaches for predicting ASM treatment outcomes. Fourteen experiment sets were conducted:

- **Experiment 1:** Text report embeddings (LLM) + drug structure embeddings (SMILES)
- **Experiment 2:** EEG signal embeddings + drug structure embeddings (SMILES)
- **Experiment 3:** LLM + EEG + SMILES embeddings (triple modality)
- **Experiment 4:** Clinical features only (baseline)
- **Experiment 5:** Clinical features + single modality fusion
- **Experiment 6:** Clinical features + SMILES + third modality (text or EEG)
- **Experiment 7:** All four modalities (Clinical + LLM + EEG + SMILES)
- **Experiment 8:** Stratification analysis (multi-label CV)
- **Experiment 9:** EEG encoder ablation (SimpleCNN, EEGNet, LaBraM, EEG2Vec)
- **Experiment 10:** Direct LLM text modality (frozen/fine-tuned encoder)
- **Experiment 11:** EEG2Vec 128D upgrade for triple and clinical+EEG modality experiments
- **Experiment 12:** FuseMoE hyperparameter investigation (exp3b regression)
- **Experiment 13:** Qwen 2.5 fine-tuning with unfrozen transformer layers
- **Experiment 14:** Optuna HP tuning for top 3 models (Exp7a, Exp11, Exp12)

The best performing model achieved **AUC 0.798** with a **balanced accuracy of 0.814** using all four modalities with MLP fusion (Exp7a, ClinicalBERT + ChemBERTa + SimpleCNN). Class weighting and threshold tuning (via Youden's J statistic) were applied to address class imbalance. Exp14 HP tuning did not yield reproducible improvements.

**Key finding:** Optuna HP tuning (Exp14) found trial AUCs of 0.831 (Exp7a) and 0.825 (Exp11), but these did not reproduce on rerun (0.770 and 0.789 respectively). Training variance on small datasets (n=107) causes single-trial AUCs to be optimistic. The original Exp7a baseline (AUC 0.798) remains the best confirmed result. EEG2Vec 128D upgrade improves triple MLP to AUC 0.736 but does not improve quad modality (0.791 vs 0.798 SimpleCNN), suggesting clinical features compensate for weaker EEG encoding.

---

## Experiment 1: LLM + SMILES Fusion

Combined clinical text report embeddings with molecular structure embeddings.

### Models Tested
- **Text encoders:** ClinicalBERT, PubMedBERT
- **SMILES encoders:** ChemBERTa, SMILES Transformer
- **Fusion methods:** Concatenation + MLP (1a), FuseMoE (1b)

### Results (5-fold CV with Threshold Tuning)

| Experiment | Text Model | SMILES Model | AUC | Bal Acc Tuned | F1 Tuned |
|------------|------------|--------------|-----|---------------|----------|
| exp1b | ClinicalBERT | SMILES-Trf | **0.674 +/- 0.139** | **0.720 +/- 0.079** | 0.664 +/- 0.147 |
| exp1a | PubMedBERT | ChemBERTa | 0.641 +/- 0.070 | 0.699 +/- 0.033 | 0.676 +/- 0.082 |
| exp1b | ClinicalBERT | ChemBERTa | 0.636 +/- 0.142 | 0.709 +/- 0.100 | 0.713 +/- 0.116 |
| exp1a | PubMedBERT | SMILES-Trf | 0.632 +/- 0.106 | 0.676 +/- 0.073 | 0.624 +/- 0.198 |
| exp1a | ClinicalBERT | SMILES-Trf | 0.623 +/- 0.112 | 0.677 +/- 0.073 | 0.557 +/- 0.110 |
| exp1b | PubMedBERT | SMILES-Trf | 0.612 +/- 0.049 | 0.689 +/- 0.049 | 0.713 +/- 0.052 |
| exp1a | ClinicalBERT | ChemBERTa | 0.609 +/- 0.099 | 0.669 +/- 0.067 | 0.707 +/- 0.061 |
| exp1b | PubMedBERT | ChemBERTa | 0.601 +/- 0.104 | 0.681 +/- 0.079 | 0.707 +/- 0.070 |

### Key Observations
- Best AUC: exp1b_clinicalbert_smilestrf (0.674) with revised FuseMoE (Laplace gating, MI loss, temperature annealing)
- PubMedBERT + SMILES-Trf FuseMoE has lowest variance (AUC std 0.049) - most stable configuration
- FuseMoE slightly outperforms MLP for best AUC (0.674 vs 0.641)
- High variance across folds (std 0.05-0.14) due to small dataset (n=121)

### Exp1b Results: Exp12 Tuned HP (17 February 2026)

Applied Exp12 best hyperparameters (lr=5e-5, 4 experts, no temp decay) to all exp1b configurations.

| Text Model | SMILES Model | AUC | Bal Acc Tuned | F1 Tuned | vs Default HP |
|------------|--------------|-----|---------------|----------|---------------|
| ClinicalBERT | ChemBERTa | 0.650 +/- 0.111 | 0.706 +/- 0.082 | 0.626 +/- 0.205 | +0.014, std -0.031 |
| PubMedBERT | ChemBERTa | 0.649 +/- 0.089 | 0.701 +/- 0.078 | 0.706 +/- 0.083 | **+0.048**, std -0.015 |
| ClinicalBERT | SMILES-Trf | 0.647 +/- 0.062 | 0.690 +/- 0.018 | 0.704 +/- 0.070 | -0.027, std **-0.077** |
| PubMedBERT | SMILES-Trf | 0.629 +/- 0.077 | 0.671 +/- 0.050 | 0.709 +/- 0.027 | +0.017, std +0.028 |

- PubMedBERT + ChemBERTa shows largest improvement (+0.048 AUC, 0.601 -> 0.649)
- ClinicalBERT + SMILES-Trf AUC regresses (-0.027) but variance drops massively (0.139 -> 0.062)
- Best overall exp1b AUC remains 0.674 (ClinicalBERT + SMILES-Trf, default HP)

---

## Experiment 2: EEG + SMILES Fusion

Combined EEG signal embeddings with molecular structure embeddings.

### Models Tested
- **EEG encoder:** SimpleCNN (27 channels, 10s windows)
- **SMILES encoders:** ChemBERTa, SMILES Transformer
- **Fusion methods:** Concatenation + MLP (2a), FuseMoE (2b)

### Results (5-fold CV with Class Weighting and Threshold Tuning)

| Experiment | SMILES Model | Fusion | AUC | Bal Acc Tuned | F1 Tuned |
|------------|--------------|--------|-----|---------------|----------|
| exp2a | SMILES-Trf | MLP | **0.634 +/- 0.045** | **0.699 +/- 0.047** | **0.720 +/- 0.056** |
| exp2a | ChemBERTa | MLP | 0.611 +/- 0.074 | 0.672 +/- 0.045 | 0.632 +/- 0.075 |
| exp2b | SMILES-Trf | FuseMoE | 0.611 +/- 0.056 | 0.621 +/- 0.049 | 0.556 +/- 0.175 |
| exp2b | ChemBERTa | FuseMoE | 0.572 +/- 0.024 | 0.599 +/- 0.012 | 0.569 +/- 0.133 |

### Key Observations

- MLP fusion still outperforms FuseMoE but gap narrowed (AUC 0.634 vs 0.611 for SMILES-Trf)
- Revised FuseMoE substantially more stable than old implementation (AUC std 0.056 vs 0.095 for SMILES-Trf)
- SMILES Transformer embeddings consistently outperform ChemBERTa across both fusion methods
- FuseMoE F1 variance improved but still high (std 0.13-0.18 vs previous 0.27-0.28)
- Class weighting added in re-run (previously missing)

### Exp2b Results: Exp12 Tuned HP (17 February 2026)

Applied Exp12 best hyperparameters (lr=5e-5, 4 experts, no temp decay) to exp2b FuseMoE.

| SMILES Model | Fusion | AUC | Bal Acc Tuned | F1 Tuned | vs Default HP |
|--------------|--------|-----|---------------|----------|---------------|
| ChemBERTa | FuseMoE (Exp12 HP) | 0.585 +/- 0.077 | 0.603 +/- 0.056 | 0.524 +/- 0.178 | +0.013, std +0.053 |
| SMILES-Trf | FuseMoE (Exp12 HP) | 0.569 +/- 0.087 | 0.602 +/- 0.055 | 0.640 +/- 0.067 | -0.042, std +0.031 |

- Exp12 HP does not improve exp2b - SMILES-Trf regresses (-0.042 AUC) and variance increases for both configs
- EEG + SMILES may need different FuseMoE HP than text-containing experiments; temperature annealing appears beneficial for this modality pairing
- Best exp2b FuseMoE remains SMILES-Trf at 0.611 with default revised HP

---

## Experiment 3: LLM + EEG + SMILES Fusion

Combined all three modalities: text report embeddings, EEG signal embeddings, and molecular structure embeddings.

### Models Tested
- **Text encoders:** ClinicalBERT, PubMedBERT
- **EEG encoder:** SimpleCNN (27 channels, 10s windows)
- **SMILES encoders:** ChemBERTa, SMILES Transformer
- **Fusion methods:** Concatenation + MLP (3a), FuseMoE (3b)
- **Class balancing:** Inverse frequency class weights in loss function
- **Threshold tuning:** Optimal threshold selected via precision-recall curve

### Results (5-fold CV)

| Experiment | Text Model | SMILES Model | Fusion | AUC | F1 | F1_tuned |
|------------|------------|--------------|--------|-----|-----|----------|
| exp3b | ClinicalBERT | SMILES-Trf | FuseMoE | 0.694 | 0.541 | **0.780** |
| exp3b | PubMedBERT | SMILES-Trf | FuseMoE | 0.618 | 0.657 | 0.772 |
| exp3a | PubMedBERT | ChemBERTa | MLP | **0.701** | 0.537 | 0.767 |
| exp3a | PubMedBERT | SMILES-Trf | MLP | 0.672 | 0.655 | 0.756 |
| exp3a | ClinicalBERT | ChemBERTa | MLP | 0.632 | 0.528 | 0.754 |
| exp3a | ClinicalBERT | SMILES-Trf | MLP | 0.683 | 0.518 | 0.750 |
| exp3b | ClinicalBERT | ChemBERTa | FuseMoE | 0.617 | 0.585 | 0.737 |
| exp3b | PubMedBERT | ChemBERTa | FuseMoE | 0.614 | 0.489 | 0.736 |

### Key Observations

- **F1_tuned is the recommended metric** - shows model potential with proper threshold selection
- Optimal thresholds ranged from 0.32-0.40 (vs default 0.5), confirming class imbalance
- FuseMoE now achieves best F1_tuned (0.780) with ClinicalBERT + SMILES-Trf
- F1_tuned variance is low (~0.02-0.05 std) vs high variance at default threshold
- Class weights improved F1 from ~0.22 to ~0.54 at default threshold
- Only 107 patients had all three modalities (vs 151 for dual-modality)

### Updated Results: Balanced Accuracy Threshold Selection

**Methodology change:** Threshold now selected by maximising balanced accuracy using Youden's J statistic (`J = TPR - FPR`) from the ROC curve, rather than maximising F1 from the precision-recall curve. This ensures equal weighting of both classes.

| Experiment | Text Model | SMILES Model | Fusion | AUC | Bal Acc Tuned | F1 Tuned |
|------------|------------|--------------|--------|-----|---------------|----------|
| exp3a | ClinicalBERT | ChemBERTa | MLP | 0.687 | 0.713 | 0.654 |
| exp3b | ClinicalBERT | ChemBERTa | FuseMoE | **0.677 +/- 0.108** | **0.726 +/- 0.092** | **0.761 +/- 0.084** |
| exp3b | PubMedBERT | SMILES-Trf | FuseMoE | 0.657 +/- 0.084 | 0.682 +/- 0.058 | 0.689 +/- 0.060 |
| exp3a | ClinicalBERT | SMILES-Trf | MLP | 0.649 | 0.707 | 0.736 |
| exp3b | ClinicalBERT | SMILES-Trf | FuseMoE | 0.628 +/- 0.124 | 0.684 +/- 0.092 | 0.693 +/- 0.057 |
| exp3a | PubMedBERT | ChemBERTa | MLP | 0.625 | 0.686 | 0.630 |
| exp3a | PubMedBERT | SMILES-Trf | MLP | 0.620 | 0.673 | 0.624 |
| exp3b | PubMedBERT | ChemBERTa | FuseMoE | 0.604 +/- 0.088 | 0.680 +/- 0.061 | 0.658 +/- 0.106 |

> **FuseMoE regression note:** The revised FuseMoE (Laplace gating, MI loss) reduced Exp3b AUC from 0.753 to 0.677 compared to the previous implementation. The old softmax + malformed KL loss may have acted as accidental regularisation. The exp3a MLP results (unchanged at AUC 0.687-0.701) now outperform FuseMoE for triple modality, consistent with the pattern that simpler fusion methods perform better on small datasets (n=107).

### Fold Deviation Statistics (5-fold CV, Jan 28 run)

| Experiment | Text Model | SMILES Model | Fusion | AUC Range | Bal Acc Range | AUC Std | Bal Acc Std |
|------------|------------|--------------|--------|-----------|---------------|---------|-------------|
| exp3a | ClinicalBERT | ChemBERTa | MLP | 0.558-0.868 | 0.658-0.818 | 0.112 | 0.059 |
| exp3a | PubMedBERT | ChemBERTa | MLP | 0.658-0.702 | 0.667-0.742 | 0.016 | 0.028 |
| exp3b | ClinicalBERT | ChemBERTa | FuseMoE | 0.598-0.769 | 0.642-0.773 | 0.061 | 0.052 |
| exp3b | PubMedBERT | ChemBERTa | FuseMoE | 0.583-0.653 | 0.650-0.742 | 0.023 | 0.032 |
| exp3a | ClinicalBERT | SMILES-Trf | MLP | 0.600-0.717 | 0.658-0.767 | 0.038 | 0.040 |
| exp3a | PubMedBERT | SMILES-Trf | MLP | 0.542-0.825 | 0.608-0.783 | 0.098 | 0.063 |
| exp3b | ClinicalBERT | SMILES-Trf | FuseMoE | 0.550-0.744 | 0.608-0.727 | 0.078 | 0.047 |
| exp3b | PubMedBERT | SMILES-Trf | FuseMoE | 0.600-0.674 | 0.667-0.750 | 0.028 | 0.031 |

### Key Observations (Updated)

- **Best model:** ClinicalBERT + ChemBERTa + MLP (AUC 0.687, Balanced Accuracy 0.713) - MLP now outperforms FuseMoE
- **FuseMoE regression:** Revised FuseMoE reduced best AUC from 0.753 to 0.677 - old malformed KL loss may have provided accidental regularisation
- MLP fusion more reliable than MoE for triple modality on small datasets (n=107)
- ChemBERTa outperforms SMILES-Trf for both fusion methods
- Fold variance remains moderate (AUC std 0.088-0.124 for FuseMoE)


---

## Experiment 4: Clinical Features Baseline

Established a clinical-only baseline using 16 demographic/clinical features to benchmark embedding-based approaches.

### Clinical Features Used

- **Binary (13):** sex, pretrt_sz_5, focal, fam_hx, febrile, ci, birth_t, head, drug, alcohol, cvd, psy, ld
- **Numeric (1):** age_init (Z-score normalised)
- **Categorical (2):** lesion, eeg_cat (one-hot encoded, 6 dims total)

### Results (5-fold CV)

| Experiment | Model | AUC | Balanced Acc Tuned | F1 Tuned |
|------------|-------|-----|-------------------|----------|
| **Exp4a** | MLP (~3.7K params) | **0.664 +/- 0.043** | **0.675 +/- 0.032** | 0.627 +/- 0.056 |
| Exp4b | Attention (~104K params) | 0.636 +/- 0.069 | 0.673 +/- 0.061 | 0.629 +/- 0.123 |

### Per-Fold AUC Values

- **Exp4a MLP:** [0.712, 0.614, 0.719, 0.643, 0.630]
- **Exp4b Attention:** [0.690, 0.683, 0.700, 0.538, 0.568]

### Key Observations

- Simple MLP baseline more stable than attention model (lower variance)
- Matches Feng et al. 2025 benchmark (clinical-only AUC 0.67)
- Attention model shows instability on folds 4-5 (may need more data)

---

## Experiment 5: Clinical + Single Modality Fusion

Tested whether fusing clinical features with a single embedding modality improves upon the clinical-only baseline.

### Architecture

Late fusion: each modality encoded to 64D, concatenated (128D), then classified.

### Results (5-fold CV)

| Experiment | Modality | Model | AUC | Balanced Acc Tuned | F1 Tuned |
|------------|----------|-------|-----|-------------------|----------|
| **Exp5a** | Clinical + SMILES | ChemBERTa | **0.689 +/- 0.060** | 0.680 +/- 0.048 | 0.638 +/- 0.103 |
| **Exp5a** | Clinical + SMILES | SMILES-Trf | 0.687 +/- 0.041 | 0.682 +/- 0.042 | 0.674 +/- 0.063 |
| **Exp5b** | Clinical + Text | ClinicalBERT | 0.676 +/- 0.083 | **0.708 +/- 0.073** | 0.716 +/- 0.090 |
| **Exp5b** | Clinical + Text | PubMedBERT | 0.620 +/- 0.038 | 0.690 +/- 0.060 | 0.729 +/- 0.043 |
| **Exp5c** | Clinical + EEG | SimpleCNN | 0.675 +/- 0.061 | 0.698 +/- 0.057 | 0.689 +/- 0.091 |

### Comparison with Exp4a Baseline (AUC 0.664)

| Experiment | AUC | Delta |
|------------|-----|-------|
| exp5a_chemberta | 0.689 | **+0.025** |
| exp5a_smilestrf | 0.687 | **+0.023** |
| exp5b_clinicalbert | 0.676 | +0.012 |
| exp5c_simplecnn | 0.675 | +0.011 |
| exp5b_pubmedbert | 0.620 | -0.044 |

### Key Observations

- SMILES embeddings provide most consistent lift (~+0.02 AUC) with low variance
- ClinicalBERT achieves highest balanced accuracy (0.708) but has high AUC variance
- EEG fusion now stable (std 0.061, down from 0.113) due to multi-label stratification and pipeline improvements
- EEG now provides a small positive lift over clinical baseline (+0.011 AUC)
- Dataset size affects stability: text experiments have smallest dataset (121 patients)

---

## Experiment 6: Clinical + SMILES + Third Modality Fusion

Tested whether combining clinical features with SMILES embeddings AND a third modality improves upon previous experiments.

### Architecture

Late fusion with three modality streams: each modality encoded to 64D, concatenated (192D), then classified.

### Exp6a: Clinical + SMILES + Text (5-fold CV)

| Text Model | SMILES Model | AUC | Bal Acc Tuned | F1 Tuned |
|------------|--------------|-----|---------------|----------|
| **PubMedBERT** | **ChemBERTa** | **0.702 +/- 0.067** | **0.705 +/- 0.058** | **0.738 +/- 0.093** |
| PubMedBERT | SMILES-Trf | 0.651 +/- 0.068 | 0.686 +/- 0.035 | 0.662 +/- 0.103 |
| ClinicalBERT | SMILES-Trf | 0.650 +/- 0.096 | 0.691 +/- 0.067 | 0.698 +/- 0.062 |
| ClinicalBERT | ChemBERTa | 0.627 +/- 0.145 | 0.672 +/- 0.100 | 0.671 +/- 0.115 |

### Exp6b: Clinical + SMILES + EEG (5-fold CV)

| EEG Model | SMILES Model | AUC | Bal Acc Tuned | F1 Tuned |
|-----------|--------------|-----|---------------|----------|
| SimpleCNN | SMILES-Trf | 0.647 +/- 0.061 | 0.663 +/- 0.035 | 0.631 +/- 0.135 |
| SimpleCNN | ChemBERTa | 0.643 +/- 0.061 | 0.692 +/- 0.062 | 0.684 +/- 0.105 |

### Key Observations

- Best Exp6 model: PubMedBERT + ChemBERTa (AUC 0.702, +0.038 vs Exp4a, +0.013 vs Exp5a)
- Text fusion (Exp6a) outperforms EEG fusion (Exp6b)
- Exp6a (AUC 0.702) now outperforms Exp3 triple modality (best MLP AUC 0.687, best FuseMoE AUC 0.677) - clinical features add value
- Clinical features provide meaningful lift when combined with embedding modalities

---

## Experiment 7: All Four Modalities Fusion

Tested whether combining all four modalities improves upon triple modality (Exp3).

### Architecture

- **Exp7a (MLP):** Each modality encoded to 64D, concatenated (256D), classified (~2M params)
- **Exp7b (MoE):** Cross-modal attention + Sparse MoE layers (~4.7M params)

### Results (5-fold CV)

| Experiment | Text Model | Fusion | AUC | Bal Acc Tuned | F1 Tuned |
|------------|------------|--------|-----|---------------|----------|
| **exp7a** | **ClinicalBERT** | **MLP** | **0.798 +/- 0.093** | **0.814 +/- 0.069** | **0.813 +/- 0.071** |
| exp7b | ClinicalBERT | MoE | 0.753 +/- 0.127 | 0.754 +/- 0.079 | 0.716 +/- 0.108 |
| exp7a | PubMedBERT | MLP | 0.752 +/- 0.069 | 0.766 +/- 0.065 | 0.749 +/- 0.119 |
| exp7b | PubMedBERT | MoE | 0.712 +/- 0.072 | 0.716 +/- 0.051 | 0.718 +/- 0.100 |

### Per-Fold AUC (Best Model: exp7a_clinicalbert)

| Fold | AUC | Bal Acc | F1 (tuned) |
|------|-----|---------|------------|
| 1 | 0.689 | 0.731 | 0.786 |
| 2 | 0.818 | 0.864 | 0.842 |
| 3 | 0.700 | 0.742 | 0.700 |
| 4 | 0.933 | 0.908 | 0.917 |
| 5 | 0.850 | 0.825 | 0.818 |

### Exp7b Results: Exp12 Tuned HP (17 February 2026)

Applied Exp12 best hyperparameters (lr=5e-5, 4 experts, no temp decay) to exp7b FuseMoE.

| Text Model | Fusion | AUC | Bal Acc Tuned | F1 Tuned | vs Default HP |
|------------|--------|-----|---------------|----------|---------------|
| ClinicalBERT | MoE (Exp12 HP) | 0.746 +/- 0.098 | 0.779 +/- 0.078 | 0.788 +/- 0.077 | -0.007, std -0.029 |
| **PubMedBERT** | **MoE (Exp12 HP)** | **0.738 +/- 0.084** | **0.737 +/- 0.062** | **0.721 +/- 0.073** | **+0.026**, std +0.012 |

- PubMedBERT FuseMoE improves substantially (+0.026 AUC, 0.712 -> 0.738)
- ClinicalBERT FuseMoE marginally declines in AUC (-0.007) but variance reduces (0.127 -> 0.098)
- Best exp7b ClinicalBERT remains 0.753 (default HP); best exp7b PubMedBERT is now 0.738 (Exp12 HP)

### Key Observations

- **New overall best:** exp7a_clinicalbert_chemberta (MLP) with AUC 0.798 (up from 0.762)
- MLP fusion still outperforms MoE (0.798 vs 0.753) but gap narrowed
- Revised FuseMoE improved MoE substantially: ClinicalBERT +0.033 AUC, PubMedBERT +0.050 AUC
- Exp12 HP further improves PubMedBERT MoE (+0.026 AUC) but not ClinicalBERT (-0.007)
- Fold 4 still shows highest performance (AUC 0.933) - consistent across re-runs
- Multi-label stratification and pipeline improvements contributed to overall gains

### Meta-Analysis (Sidik-Jonkman + Knapp-Hartung) - 30 January 2026 run

Proper confidence intervals accounting for between-fold heterogeneity (from previous run; to be re-computed with updated results):

| Configuration | AUC | 95% CI | I² | τ² |
|---------------|-----|--------|----|----|
| exp7a_clinicalbert | 0.762 | [0.633, 0.891] | 80% | 0.0086 |
| exp7a_pubmedbert | 0.746 | [0.664, 0.829] | 80% | 0.0035 |
| exp7b_clinicalbert | 0.720 | [0.573, 0.867] | 80% | 0.0112 |
| exp7b_pubmedbert | 0.662 | [0.593, 0.731] | 80% | 0.0025 |

**Interpretation:**
- **Wide CIs** (0.14–0.29 range): Limited precision due to small sample (k=5 folds, n=107 patients)
- **High I² (80%)**: Substantial heterogeneity - fold composition strongly affects performance
- **Overlapping CIs**: Cannot definitively rank configurations; differences may be due to chance
- **MoE more variable**: Higher τ² for MoE models indicates less stable training on small data
- **PubMedBERT more consistent**: Lower τ² but lower point estimates than ClinicalBERT

---

## Experiment 8: Stratification Analysis

Investigated whether the high I² (80%) heterogeneity in Exp7 could be reduced through improved cross-validation stratification.

### Motivation

The high fold-to-fold variance observed in previous experiments may be partly due to:
1. **Outcome-only stratification**: Current CV only balances the target variable
2. **Imbalanced clinical features**: Some features have >95% majority class
3. **Unbalanced fold composition**: Key features may cluster in certain folds

### Feature Imbalance Analysis

| Feature | Majority % | Minority n | Status |
|---------|------------|------------|--------|
| `ld` | 98.5% | 3 | SEVERE |
| `birth_t` | 97.5% | 5 | SEVERE |
| `febrile` | 96.0% | 8 | SEVERE |
| `ci` | 95.5% | 9 | SEVERE |
| `fam_hx` | 88.5% | 23 | WARNING |
| `cvd` | 88.4% | 23 | WARNING |
| `focal` | 79.2% | 42 | OK |
| `sex` | 62.7% | 76 | OK |

**Rationale**: Considering dropping severely imbalanced features (`ld`, `birth_t`, `febrile`, `ci`) as they seem to provide minimal impact/discrimination.

### Stratification Comparison

Compared fold balance variance across stratification methods:

| Feature | Outcome-only (fold_std) | Multi-label (fold_std) | Improvement |
|---------|------------------------|------------------------|-------------|
| focal | 10.7% | 1.3% | **8x better** |
| sex | 5.8% | 1.1% | **5x better** |
| outcome | 0.5% | 0.5% | Same |

Multi-label stratification (outcome + focal + sex) dramatically reduces fold-to-fold variance for clinical features while maintaining outcome balance.

### Data Quality Issues Fixed

- `psy` column: Mixed types ('0', '1', '0.0', '1.0', '?') standardised
- `lesion` column: Mixed types (1, '1.0', 'NOT AVAILABLE') standardised
- `outcome` column: String values converted to numeric

### Key Findings

1. **Multi-label stratification reduces fold variance by 5-8x** for key features
2. **Severely imbalanced features** (ld, birth_t, febrile, ci) have <10 minority samples
3. **Data cleaning required**: Several columns had mixed types needing standardisation
4. **Composite stratification** (outcome + focal) also effective but less balanced for sex

### Files

- `exp8_stratification/feature_analysis.py` - Distribution analysis
- `exp8_stratification/stratified_cv.py` - Multi-label stratification implementation
- `exp8_stratification/run_experiments.py` - Full experiment runner

---

## Limitations

- Relatively small sample size (n=151 for dual-modality, n=107 for triple/quad-modality, n=205 for clinical-only)
- High variance across folds (std up to 0.10 for AUC in some configurations)
- Wide 95% CIs due to small k (5 folds) and high heterogeneity (I²=80%)
- Overlapping CIs between best models prevent definitive ranking
- LaBraM EEG encoder underperforms (AUC 0.549) - architecture may be unsuitable for 27-channel clinical EEG with small dataset
- Quad-modality limited by intersection of all data sources (107 patients)
- FuseMoE requires careful hyperparameter tuning per experiment - default configuration caused significant regression in Exp3b (resolved in Exp12)
- Exp7a EEG2Vec results (Exp11) did not complete - most important missing result for determining ceiling performance

---

## Experiment 9: EEG Variance Investigation

Investigated the sources of high fold-to-fold variance in EEG experiments (Exp5c AUC std 0.113) and conducted encoder ablation study.

### Key Findings from Fold Analysis

**Outcome-only stratification (previous):**
- `focal` varies 67.5%-92.7% across folds (25% range)
- Best fold (4, AUC 0.866) has 92.7% focal patients vs 71.8% in worst fold (5, AUC 0.545)
- EEG padding ratio strongly negatively correlated with AUC (r=-0.78)

**Multi-label stratification (implemented):**
- `focal` balanced to 77.5%-80.5% (3% range) - **8x variance reduction**
- `sex` balanced to 60.9%-63.4% (2.5% range) - **6x variance reduction**
- Correlations with AUC drop significantly (focal: 0.74 -> 0.21)

### Implemented Improvements

| Component | Implementation | Location |
|-----------|----------------|----------|
| Multi-label stratification | Integrated into exp2/exp5 training | `exp{2,5}_fusion/training.py` |
| EEG quality metrics | SNR, artifacts, flatlines, correlation | `exp2_fusion/eeg_pipeline.py` |
| EEG normalisation | Global z-score, window z-score, robust | `exp2_fusion/eeg_pipeline.py` |
| Alternative aggregators | Attention, MaxPool, LSTM, MultiScale | `exp2_fusion/models/aggregators.py` |
| EEGNet encoder | Added alongside SimpleCNN (~3x fewer params) | `exp2_fusion/models/eeg_encoders.py` |
| Ablation framework | 12 experiments defined | `exp9_eeg_investigation/run_experiments.py` |

### Results: Encoder Ablation (12 February 2026)

Run on M3 HPC (A100 80GB). Three HPC runs required: Run 1 failed (missing `iterative-stratification`), Run 2 partially succeeded (braindecode CUDA issue), Run 3 all 4 encoders completed successfully.

| Encoder | AUC | Bal Acc Tuned | F1 Tuned | AUC Std |
|---------|-----|---------------|----------|---------|
| **EEG2Vec** | **0.661 +/- 0.061** | **0.689 +/- 0.054** | 0.585 +/- 0.149 | **0.061** |
| EEGNet | 0.648 +/- 0.107 | 0.686 +/- 0.078 | 0.584 +/- 0.239 | 0.107 |
| SimpleCNN (baseline) | 0.607 +/- 0.107 | 0.661 +/- 0.076 | 0.616 +/- 0.059 | 0.107 |
| LaBraM | 0.549 +/- 0.077 | 0.608 +/- 0.036 | 0.512 +/- 0.196 | 0.077 |

### Per-Fold AUC

| Encoder | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---------|--------|--------|--------|--------|--------|
| EEG2Vec | 0.710 | 0.673 | 0.701 | 0.678 | 0.542 |
| EEGNet | 0.741 | 0.711 | 0.705 | 0.640 | 0.444 |
| SimpleCNN | 0.710 | 0.691 | 0.683 | 0.487 | 0.467 |
| LaBraM | 0.629 | 0.653 | 0.509 | 0.471 | 0.481 |

### Key Observations

- **EEG2Vec achieves best AUC (0.661) with lowest variance (std 0.061)** - most stable encoder
- EEGNet outperforms SimpleCNN (+0.041 AUC) but both share high variance (std 0.107)
- LaBraM significantly underperforms (AUC 0.549) - likely due to small dataset or 27-channel clinical EEG mismatch
- Multi-label stratification did not eliminate variance for SimpleCNN/EEGNet - encoder architecture also contributes
- EEG2Vec's CVAE pre-training provides more robust features (std 0.061 vs 0.107)
- Folds 4/5 consistently weakest across all encoders

### Extended Ablation: Aggregator, Depth, and Dimension (13 February 2026)

13 configurations tested on M3 HPC (Job 51370963), extending the encoder ablation with aggregator types, transformer depths, and embedding dimensions. All configs use EEG2Vec encoder unless otherwise noted.

| Config | Encoder | Aggregator | Embed Dim | AUC | Bal Acc | F1 Tuned |
|--------|---------|-----------|-----------|-----|--------|----------|
| **embed_dim_128** | EEG2Vec | Transformer | 128 | **0.730 +/- 0.034** | **0.725 +/- 0.038** | **0.732 +/- 0.060** |
| aggregator_meanmax | EEG2Vec | MeanMax | 256 | 0.722 +/- 0.079 | 0.740 +/- 0.065 | 0.689 +/- 0.119 |
| embed_dim_64 | EEG2Vec | Transformer | 64 | 0.687 +/- 0.129 | 0.715 +/- 0.102 | 0.716 +/- 0.105 |
| aggregator_depth_0 | EEG2Vec | Attention (0 layers) | 256 | 0.669 +/- 0.070 | 0.690 +/- 0.069 | 0.690 +/- 0.112 |
| aggregator_maxpool | EEG2Vec | MaxPool | 256 | 0.668 +/- 0.033 | 0.699 +/- 0.051 | 0.676 +/- 0.142 |
| aggregator_attention | EEG2Vec | Attention | 256 | 0.666 +/- 0.059 | 0.698 +/- 0.053 | 0.674 +/- 0.067 |
| aggregator_depth_1 | EEG2Vec | Transformer (1L) | 256 | 0.666 +/- 0.055 | 0.682 +/- 0.046 | 0.611 +/- 0.070 |
| baseline_simplecnn | SimpleCNN | Transformer | 256 | 0.620 +/- 0.082 | 0.685 +/- 0.038 | 0.606 +/- 0.092 |
| aggregator_depth_4 | EEG2Vec | Transformer (4L) | 256 | 0.605 +/- 0.118 | 0.638 +/- 0.079 | 0.524 +/- 0.169 |
| encoder_eegnet | EEGNet | Transformer | 256 | 0.603 +/- 0.067 | 0.654 +/- 0.051 | 0.636 +/- 0.123 |
| encoder_eeg2vec | EEG2Vec | Transformer | 256 | 0.594 +/- 0.063 | 0.639 +/- 0.060 | 0.471 +/- 0.117 |
| aggregator_lstm | EEG2Vec | LSTM | 256 | 0.588 +/- 0.105 | 0.651 +/- 0.046 | 0.606 +/- 0.125 |
| encoder_labram | LaBraM | Transformer | 128 | 0.575 +/- 0.094 | 0.610 +/- 0.052 | 0.511 +/- 0.201 |
| encoder_frozen | SimpleCNN (frozen) | Transformer | 256 | 0.559 +/- 0.103 | 0.639 +/- 0.069 | 0.627 +/- 0.149 |

#### Extended Ablation Key Findings

- **128D embeddings are optimal** (AUC 0.730, lowest std 0.034) - reducing from 256D improves generalisation
- **MeanMax aggregation** is a strong alternative (highest balanced accuracy 0.740) but has higher AUC variance
- **Transformer depth sweet spot is 2 layers** - 0 layers (attention only) works well (0.669), 1 layer comparable (0.666), 4 layers overfits (0.605)
- **LSTM aggregation underperforms** all other aggregators (0.588) - temporal modelling may not help with 10s windows
- **Freezing encoder hurts** significantly (0.559 vs 0.620 for SimpleCNN) - end-to-end training essential

### Files

- `exp9_eeg_investigation/fold_analysis.py` - Fold composition analysis
- `exp9_eeg_investigation/quality_analysis.py` - EEG quality metrics analysis
- `exp9_eeg_investigation/run_experiments.py` - Ablation experiment framework
- `exp9_eeg_investigation/config.py` - Configuration

---

## Experiment 10: Direct LLM Text Modality

Ran LLM inference at training time (frozen encoder mode) instead of pre-computed embeddings, enabling comparison across different LLM architectures with the same training pipeline.

### Architecture

Late fusion: Clinical features (19D) -> 64D + Raw text -> LLM encoder -> embed_dim -> 64D, concatenated (128D), then classified. Matches Exp5b architecture.

### Results: Frozen Encoder (12 February 2026)

Run on M3 HPC (Job 51362383, node m3n102, A100 80GB, ~20 min runtime).

| Model | AUC | Bal Acc Tuned | F1 Tuned |
|-------|-----|---------------|----------|
| **Qwen 2.5 0.5B** | **0.689 +/- 0.088** | **0.717 +/- 0.073** | 0.666 +/- 0.119 |
| ClinicalBERT | 0.644 +/- 0.121 | 0.695 +/- 0.092 | 0.671 +/- 0.130 |
| PubMedBERT | 0.635 +/- 0.096 | 0.671 +/- 0.075 | 0.674 +/- 0.058 |

### Per-Fold AUC

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|-------|--------|--------|--------|--------|--------|
| Qwen 2.5 0.5B | 0.769 | 0.790 | 0.671 | 0.542 | 0.671 |
| ClinicalBERT | 0.788 | 0.629 | 0.580 | 0.458 | 0.762 |
| PubMedBERT | 0.744 | 0.685 | 0.643 | 0.458 | 0.643 |

### Key Observations

- Qwen 2.5 0.5B (general-purpose) outperforms both biomedical-specific models in frozen mode
- Fold 4 consistently weakest across all 3 models (AUC 0.458-0.542) - data composition issue
- Frozen results comparable to Exp5b pre-computed embeddings (ClinicalBERT 0.644 vs 0.676)

### Results: Fine-tuned Encoder (Phase 2, 13 February 2026)

Run on M3 HPC (Job 51370915, A100 80GB). Last 2 transformer layers unfrozen with differential learning rates (encoder: 2e-5, head: 1e-3).

| Model | AUC | Bal Acc Tuned | F1 Tuned |
|-------|-----|---------------|----------|
| **ClinicalBERT (fine-tuned)** | **0.691 +/- 0.081** | **0.723 +/- 0.057** | **0.698 +/- 0.106** |
| PubMedBERT (fine-tuned) | 0.638 +/- 0.144 | 0.690 +/- 0.084 | 0.674 +/- 0.101 |

### Per-Fold AUC (Fine-tuned)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|-------|--------|--------|--------|--------|--------|
| ClinicalBERT (fine-tuned) | 0.737 | 0.776 | 0.643 | 0.556 | 0.741 |
| PubMedBERT (fine-tuned) | 0.801 | 0.594 | 0.720 | 0.382 | 0.692 |

### Frozen vs Fine-tuned Comparison

| Model | Frozen AUC | Fine-tuned AUC | Delta |
|-------|-----------|---------------|-------|
| ClinicalBERT | 0.644 | 0.691 | **+0.047** |
| PubMedBERT | 0.635 | 0.638 | +0.003 |

### Fine-tuning Key Observations

- ClinicalBERT benefits substantially from fine-tuning (+0.047 AUC, +0.028 Bal Acc)
- PubMedBERT barely improves with fine-tuning (+0.003 AUC) - already well-suited to clinical text in frozen mode
- Fine-tuned ClinicalBERT (0.691) marginally outperforms frozen Qwen 2.5 0.5B (0.689)
- Fold 4 remains weakest for both models (AUC 0.382-0.556) - consistent with frozen results
- PubMedBERT fine-tuning has very high variance (std 0.144) - fold 4 AUC 0.382 is a near-complete failure
- Qwen 2.5 fine-tuning tested in Exp13 - see below. AUC slightly regressed but balanced metrics improved substantially

---

## Experiment 11: EEG2Vec 128D Upgrade (15 February 2026)

Replaced SimpleCNN with EEG2Vec encoder (128D embeddings) across exp3a and exp6b base experiments, testing both transformer and MeanMax aggregators. Validates exp9 ablation findings in multi-modal settings.

### Exp3a Base: Triple MLP with EEG2Vec (n=107)

| Text Model | SMILES Model | Aggregator | AUC | Bal Acc Tuned | F1 Tuned |
|------------|--------------|-----------|-----|---------------|----------|
| **ClinicalBERT** | **SMILES-Trf** | **MeanMax** | **0.736 +/- 0.036** | 0.752 +/- 0.034 | **0.779 +/- 0.020** |
| ClinicalBERT | SMILES-Trf | Transformer | 0.733 +/- 0.087 | **0.777 +/- 0.056** | 0.764 +/- 0.085 |
| ClinicalBERT | ChemBERTa | Transformer | 0.729 +/- 0.078 | 0.756 +/- 0.084 | 0.751 +/- 0.110 |
| ClinicalBERT | ChemBERTa | MeanMax | 0.721 +/- 0.104 | 0.742 +/- 0.095 | 0.723 +/- 0.133 |

Previous exp3a best (SimpleCNN): AUC 0.687. **+0.049 AUC improvement.**

### Exp6b Base: Clinical + SMILES + EEG with EEG2Vec (n=151)

| SMILES Model | Aggregator | AUC | Bal Acc Tuned | F1 Tuned |
|--------------|-----------|-----|---------------|----------|
| **ChemBERTa** | **Transformer** | **0.697 +/- 0.070** | 0.694 +/- 0.050 | 0.672 +/- 0.096 |
| ChemBERTa | MeanMax | 0.693 +/- 0.051 | **0.712 +/- 0.044** | 0.684 +/- 0.037 |

Previous exp6b best (SimpleCNN): AUC 0.647. **+0.050 AUC improvement.**

### Exp7a Base: Quad MLP with EEG2Vec (n=107)

| Text Model | SMILES Model | Aggregator | AUC | Bal Acc Tuned | F1 Tuned |
|------------|--------------|-----------|-----|---------------|----------|
| **ClinicalBERT** | **ChemBERTa** | **Transformer** | **0.791 +/- 0.081** | 0.776 +/- 0.052 | 0.794 +/- 0.061 |
| PubMedBERT | ChemBERTa | MeanMax | 0.781 +/- 0.106 | **0.810 +/- 0.091** | **0.822 +/- 0.085** |
| PubMedBERT | ChemBERTa | Transformer | 0.757 +/- 0.141 | 0.784 +/- 0.111 | 0.796 +/- 0.103 |
| ClinicalBERT | ChemBERTa | MeanMax | 0.749 +/- 0.107 | 0.783 +/- 0.087 | 0.806 +/- 0.079 |

Previous exp7a best (SimpleCNN): AUC 0.798. **EEG2Vec does not improve quad modality** (0.791 vs 0.798).

### Exp11 Key Observations

- EEG2Vec is a clear upgrade over SimpleCNN for triple (+0.049 AUC) and clinical+EEG (+0.050 AUC) experiments
- EEG2Vec does not improve quad modality (0.791 vs 0.798 SimpleCNN) - clinical features may already compensate for weaker EEG encoding
- MeanMax aggregator confirms exp9 findings: competitive AUC with lower variance (std 0.036)
- ClinicalBERT consistently outperforms PubMedBERT with EEG2Vec (all 4 ClinicalBERT configs above 0.721 for exp3a)
- EEG fusion (0.697) now nearly matches text fusion (exp6a, 0.702) with proper EEG encoding
- PubMedBERT + MeanMax achieves best balanced metrics for quad (Bal Acc 0.810, F1 0.822) despite lower AUC

---

## Experiment 12: FuseMoE Hyperparameter Investigation (15 February 2026)

Investigated whether the exp3b FuseMoE regression (AUC 0.753 -> 0.677) was caused by suboptimal default hyperparameters. Tested 12 configurations: 3 learning rates (5e-5, 1e-4, 5e-4) x 2 expert counts (2, 4) x 2 temperature decay settings (0.9995, None). Fixed to ClinicalBERT + ChemBERTa + SimpleCNN.

### Top 5 Configurations (sorted by AUC)

| LR | Experts | Temp Decay | AUC | Bal Acc Tuned | F1 Tuned |
|-----|---------|-----------|-----|---------------|----------|
| **5e-5** | **4** | **None** | **0.760 +/- 0.112** | 0.760 +/- 0.081 | 0.742 +/- 0.132 |
| 1e-4 | 2 | 0.9995 | 0.749 +/- 0.105 | **0.773 +/- 0.064** | 0.745 +/- 0.096 |
| 1e-4 | 4 | 0.9995 | 0.737 +/- 0.080 | 0.763 +/- 0.077 | **0.772 +/- 0.110** |
| 5e-4 | 2 | None | 0.734 +/- 0.111 | 0.770 +/- 0.085 | 0.712 +/- 0.134 |
| 1e-4 | 2 | None | 0.734 +/- 0.107 | 0.747 +/- 0.078 | 0.769 +/- 0.068 |

### FuseMoE Regression Comparison

| Implementation | AUC |
|----------------|-----|
| Old FuseMoE (softmax + malformed KL) | 0.753 |
| Revised FuseMoE (default HP) | 0.677 |
| **Revised FuseMoE (tuned HP)** | **0.760** |

**Regression fully resolved.** Tuned FuseMoE surpasses the old malformed result by +0.007.

### Exp12 Key Observations

- Lower learning rates work much better for FuseMoE (5e-5, 1e-4 >> default 1e-3)
- Temperature annealing is not always beneficial - depends on learning rate and expert count
- 4 experts can outperform 2 experts when learning rate is low enough
- High variance persists (std 0.112) due to small dataset (n=107)

### Cross-Experiment Validation (17 February 2026)

Applied Exp12 best HP (lr=5e-5, 4 experts, no temp decay) to exp1b, exp2b, and exp7b:

| Experiment | Config | Default HP AUC | Exp12 HP AUC | Delta | Std Change |
|------------|--------|---------------|--------------|-------|------------|
| exp1b | ClinicalBERT + ChemBERTa | 0.636 | 0.650 | +0.014 | -0.031 |
| exp1b | ClinicalBERT + SMILES-Trf | 0.674 | 0.647 | -0.027 | **-0.077** |
| exp1b | PubMedBERT + ChemBERTa | 0.601 | 0.649 | **+0.048** | -0.015 |
| exp1b | PubMedBERT + SMILES-Trf | 0.612 | 0.629 | +0.017 | +0.028 |
| exp2b | SimpleCNN + ChemBERTa | 0.572 | 0.585 | +0.013 | +0.053 |
| exp2b | SimpleCNN + SMILES-Trf | 0.611 | 0.569 | -0.042 | +0.031 |
| exp7b | ClinicalBERT + ChemBERTa | 0.753 | 0.746 | -0.007 | -0.029 |
| exp7b | PubMedBERT + ChemBERTa | 0.712 | 0.738 | **+0.026** | +0.012 |

**Not a universal improvement** (5/8 configs improve, 3/8 regress). PubMedBERT benefits more than ClinicalBERT consistently. Variance reduction is the most reliable benefit for ClinicalBERT configs. Per-experiment HP tuning is warranted.

---

## Experiment 13: Qwen 2.5 Fine-tuning (15 February 2026)

Tested Qwen 2.5 0.5B fine-tuning with 1, 2, and 4 unfrozen transformer layers. Differential learning rates (encoder: 1e-5/2e-5, head: 1e-3).

### Results (n=121)

| Config | Layers | AUC | Bal Acc Tuned | F1 Tuned |
|--------|--------|-----|---------------|----------|
| **4 layers** | 4 | **0.682 +/- 0.046** | **0.736 +/- 0.014** | **0.737 +/- 0.043** |
| 1 layer | 1 | 0.653 +/- 0.099 | 0.712 +/- 0.060 | 0.682 +/- 0.083 |
| 2 layers | 2 | 0.640 +/- 0.131 | 0.664 +/- 0.078 | 0.650 +/- 0.159 |

Frozen Qwen baseline: AUC 0.689 +/- 0.088. Fine-tuned (4L): AUC 0.682 (-0.007).

### Exp13 Key Observations

- Fine-tuning Qwen does not improve AUC (-0.007) but dramatically reduces variance (std 0.046 vs 0.088)
- Balanced accuracy and F1 improve substantially (+0.019 and +0.071 respectively)
- 4 layers is optimal for the decoder-only architecture; 2-layer config is unstable (std 0.131)
- Fold 4 no longer catastrophic with 4-layer fine-tuning (AUC 0.632 vs 0.542 frozen)

---

## Experiment 14: Optuna HP Tuning (18 February 2026, updated)

Systematic hyperparameter tuning using Optuna's TPE sampler for the top 3 models: Exp7a QuadFusionMLP (AUC 0.798), Exp11 QuadMLPv2 with EEG2Vec (AUC 0.791), and Exp12 TripleFuseMoE (AUC 0.760). Studies resumed on HPC and are now effectively complete. **Rerun results show Optuna trial AUCs do not reproduce.**

### Search Space Summary

| Model | Parameters Tuned | Key Ranges |
|-------|-----------------|------------|
| Exp7a QuadFusionMLP | lr, wd, dropout, hidden_dim, batch_size (5 params) | lr: [5e-4, 5e-3], hd: {32, 64, 128} |
| Exp11 QuadMLPv2 | lr, wd, dropout, hidden_dim, batch_size, aggregator, eeg_dim (7 params) | lr: [5e-4, 5e-3], agg: {transformer, meanmax} |
| Exp12 TripleFuseMoE | lr, wd, dropout, experts, top_k, aux_loss, temp_decay (7 params) | lr: [1e-5, 5e-4], experts: {2, 4, 6} |

### Trial Statistics

| Study | Completed | Pruned | Total | Target |
|-------|-----------|--------|-------|--------|
| Exp7a QuadFusionMLP | 26 | 71 | 99 | 100 |
| Exp11 QuadMLPv2 | 27 | 72 | 100 | 100 |
| Exp12 TripleFuseMoE | 40 | 5 | 46 | 100 |
| **Total** | **93** | **148** | **245** | **300** |

### Best Trial Results

**Exp7a QuadFusionMLP** - Trial #24, AUC 0.831:
- lr=5.29e-4, wd=2.73e-5, dropout=0.277, hidden_dim=64, batch_size=8
- Key change: halved learning rate (5.29e-4 vs 1e-3 baseline), reduced weight decay by ~4x

**Exp11 QuadMLPv2 (EEG2Vec)** - Trial #81 (new best from resume), AUC 0.825:
- lr=3.73e-3, wd=7.82e-4, dropout=0.483, hidden_dim=32, batch_size=4, aggregator=meanmax, eeg_dim=64
- Key change: switches from transformer to meanmax aggregator. Not yet confirmed with rerun.
- Previous best: Trial #16, AUC 0.822 (transformer aggregator)

**Exp12 TripleFuseMoE** - Trial #10, AUC 0.749:
- lr=1.04e-4, wd=6.22e-5, dropout=0.052, experts=6, top_k=1, aux_loss=0.032, temp_decay=None
- Did not improve on Exp12 grid-search baseline (0.760)

### Confirmed Rerun Results

Best configurations rerun with full metrics (HPC Job 51546896). **None reproduced their Optuna trial AUC.**

| Model | Optuna Best | Rerun AUC | Rerun Bal Acc | Rerun F1 Tuned | vs Baseline |
|-------|------------|-----------|---------------|----------------|-------------|
| Exp7a QuadFusionMLP | 0.831 | 0.770 +/- 0.053 | 0.760 +/- 0.045 | 0.759 +/- 0.049 | -0.028 vs 0.798 |
| Exp11 QuadMLPv2 (EEG2Vec) | 0.822 | 0.789 +/- 0.088 | 0.768 +/- 0.062 | 0.777 +/- 0.047 | -0.002 vs 0.791 |
| Exp12 TripleFuseMoE | 0.749 | 0.697 +/- 0.110 | 0.720 +/- 0.094 | 0.698 +/- 0.131 | -0.063 vs 0.760 |

### Exp14 Key Observations

- **Optuna trial AUCs are optimistic and do not reproduce.** Rerun AUCs are 0.033-0.061 lower than trial values due to training variance (different weight initialisations, data shuffling).
- Exp7a rerun (0.770) is worse than baseline (0.798). Exp11 rerun (0.789) matches baseline (0.791). Exp12 rerun (0.697) is much worse than baseline (0.760).
- **The original Exp7a baseline (AUC 0.798) remains the best confirmed result.**
- fANOVA analysis: learning_rate dominates for Exp7a (68%), aggregator_type for Exp11 (52%), weight_decay for Exp12 (29%).
- Exp11 Trial #81 (meanmax, AUC 0.825) from resumed study still needs rerun confirmation.
- MedianPruner highly effective: 60% of trials pruned overall (148/245).

---

## Comparison: All Experiments

| Experiment | Modality | Best Model | AUC | Bal Acc Tuned | F1 Tuned |
|------------|----------|------------|-----|---------------|----------|
| **Exp7a** | **Clinical + LLM + EEG + SMILES** | **ClinicalBERT + ChemBERTa + MLP** | **0.798** | **0.814** | **0.813** |
| Exp11 | Clinical + LLM + EEG + SMILES | ClinicalBERT + ChemBERTa + Transformer (EEG2Vec) | 0.791 | 0.776 | 0.794 |
| Exp14 (Exp11) | Clinical + LLM + EEG + SMILES | ClinicalBERT + ChemBERTa + Transformer/EEG2Vec (Optuna HP, rerun) | 0.789 | 0.768 | 0.777 |
| Exp14 (Exp7a) | Clinical + LLM + EEG + SMILES | ClinicalBERT + ChemBERTa + MLP (Optuna HP, rerun) | 0.770 | 0.760 | 0.759 |
| Exp12 | LLM + EEG + SMILES | ClinicalBERT + ChemBERTa + FuseMoE (tuned) | 0.760 | 0.760 | 0.742 |
| Exp7b | Clinical + LLM + EEG + SMILES | ClinicalBERT + ChemBERTa + FuseMoE | 0.753 | 0.754 | 0.716 |
| Exp7a | Clinical + LLM + EEG + SMILES | PubMedBERT + ChemBERTa + MLP | 0.752 | 0.766 | 0.749 |
| Exp7b | Clinical + LLM + EEG + SMILES | PubMedBERT + ChemBERTa + FuseMoE (Exp12 HP) | 0.738 | 0.737 | 0.721 |
| Exp11 | LLM + EEG + SMILES | ClinicalBERT + SMILES-Trf + MeanMax (EEG2Vec) | 0.736 | 0.752 | 0.779 |
| Exp9 | EEG + SMILES (ablation) | EEG2Vec 128D + Transformer | 0.730 | 0.725 | 0.732 |
| Exp6a | Clinical + SMILES + Text | PubMedBERT + ChemBERTa | 0.702 | 0.705 | 0.738 |
| Exp14 (Exp12) | LLM + EEG + SMILES | ClinicalBERT + ChemBERTa + FuseMoE (Optuna HP, rerun) | 0.697 | 0.720 | 0.698 |
| Exp11 | Clinical + SMILES + EEG | ChemBERTa + Transformer (EEG2Vec) | 0.697 | 0.694 | 0.672 |
| Exp10 | Clinical + Direct LLM (fine-tuned) | ClinicalBERT | 0.691 | 0.723 | 0.698 |
| Exp5a | Clinical + SMILES | ChemBERTa | 0.689 | 0.680 | 0.638 |
| Exp10 | Clinical + Direct LLM (frozen) | Qwen 2.5 0.5B | 0.689 | 0.717 | 0.666 |
| Exp3a | LLM + EEG + SMILES | ClinicalBERT + ChemBERTa + MLP (SimpleCNN) | 0.687 | 0.713 | 0.654 |
| Exp13 | Clinical + Direct LLM (fine-tuned) | Qwen 2.5 0.5B (4L) | 0.682 | 0.736 | 0.737 |
| Exp3b | LLM + EEG + SMILES | ClinicalBERT + ChemBERTa + FuseMoE | 0.677 | 0.726 | 0.761 |
| Exp5b | Clinical + Text | ClinicalBERT | 0.676 | 0.708 | 0.716 |
| Exp5c | Clinical + EEG | SimpleCNN | 0.675 | 0.698 | 0.689 |
| Exp1b | LLM + SMILES | ClinicalBERT + SMILES-Trf + FuseMoE | 0.674 | 0.720 | 0.664 |
| Exp4a | Clinical only | MLP | 0.664 | 0.675 | 0.627 |
| Exp6b | Clinical + SMILES + EEG | SimpleCNN + SMILES-Trf | 0.647 | 0.663 | 0.631 |
| Exp2a | EEG + SMILES | SimpleCNN + SMILES-Trf + MLP | 0.634 | 0.699 | 0.720 |
| Exp2b | EEG + SMILES | SimpleCNN + SMILES-Trf + FuseMoE | 0.611 | 0.621 | 0.556 |

**Key findings:**
- **Exp7a baseline (AUC 0.798) remains the best confirmed result.** Optuna HP tuning (Exp14) found optimistic trial AUCs (0.831 for Exp7a, 0.825 for Exp11) that did not reproduce on rerun (0.770 and 0.789 respectively). Training variance on small datasets (n=107) means single-trial optimisation cannot overcome inherent noise.
- Exp14 Exp11 rerun (0.789) essentially matches the Exp11 baseline (0.791), while Exp7a rerun (0.770) and Exp12 rerun (0.697) are substantially worse than their baselines.
- Exp14 Exp11 Trial #81 (meanmax aggregator, AUC 0.825 from resumed study) has not yet been rerun - this is the most promising remaining configuration.
- EEG2Vec 128D upgrade (Exp11) improves triple MLP by +0.049 (0.687 -> 0.736) and exp6b by +0.050 (0.647 -> 0.697), but does not improve quad modality (0.791 vs 0.798 SimpleCNN)
- MLP fusion remains more stable than MoE on small datasets, but the gap narrows with proper MoE hyperparameters
- ClinicalBERT is the most consistently strong text model across all experiment configurations
- SMILES embeddings provide complementary signal to all other modalities

---

## Next Steps

1. ~~Look at torch api key padding mask and add to code~~ **DONE** - Already implemented in `exp2_fusion/models/eeg_transformer.py`
2. ~~Double check focal column distribution for stratification~~ **DONE** - Exp8 created with multi-label stratification
3. ~~Investigate high EEG variance~~ **DONE** - Exp9 investigation complete
4. ~~Test LaBraM/EEGNet/EEG2Vec encoders~~ **DONE** - Exp9 encoder ablation complete (EEG2Vec best, LaBraM underperforms)
5. ~~Run Exp9 encoder ablation experiments~~ **DONE** - 4 encoders compared, EEG2Vec selected as best
6. ~~Run Exp10 frozen encoder experiments~~ **DONE** - Qwen 2.5 0.5B outperforms biomedical models in frozen mode
7. ~~Run Exp9 remaining ablations: aggregator, depth, and dimension experiments with EEG2Vec~~ **DONE** - 13 configs tested, 128D embeddings optimal (AUC 0.730)
8. ~~Run Exp10 fine-tuning experiments (Phase 2) - unfreeze last 2 transformer layers~~ **DONE** - ClinicalBERT +0.047 AUC, PubMedBERT +0.003 AUC
9. ~~Replace FuseMoE implementation with revised version~~ **DONE** - Re-run results: exp1b/2b/7b improved, exp3b regressed
10. ~~Re-run Exp5c/Exp7 with pipeline improvements~~ **DONE** - Exp5c AUC 0.675 (was 0.644), Exp7a AUC 0.798 (was 0.762)
11. ~~Execute EEG2Vec swap plan - swap EEG2Vec into exp5c, exp7, exp2, exp3, exp6b~~ **DONE**
12. ~~Re-run exp1b, exp2b, exp3b with revised FuseMoE~~ **DONE** - Mixed results, exp3b regressed
13. ~~Re-run exp3a/exp7a MLP baselines with EEG2Vec encoder~~ **DONE** - Exp3a: AUC 0.736 (was 0.687). Exp7a: AUC 0.791 (vs 0.798 SimpleCNN - no improvement)
14. ~~Re-run exp6b with EEG2Vec encoder~~ **DONE** - AUC 0.697 (was 0.647)
15. ~~Investigate exp3b FuseMoE regression~~ **DONE** - Exp12: tuned FuseMoE achieves AUC 0.760, regression fully resolved. Best config: lr=5e-5, 4 experts, no temperature decay
16. ~~Run Qwen 2.5 fine-tuning~~ **DONE** - Exp13: AUC 0.682 (vs frozen 0.689), variance halved, balanced metrics improved
17. ~~Consider MeanMax aggregator for EEG~~ **DONE** - Tested in Exp11. MeanMax achieves best exp3a result (AUC 0.736, std 0.036)
18. ~~Re-submit exp11 exp7a EEG2Vec configs (quad modality with EEG2Vec)~~ **DONE** - AUC 0.791 (vs 0.798 SimpleCNN). EEG2Vec does not improve quad modality
19. ~~Apply Exp12 best hyperparameters (lr=5e-5, 4 experts, no temp decay) to other FuseMoE experiments (exp1b, exp2b, exp7b)~~ **DONE** - Mixed results: PubMedBERT benefits most (+0.048 exp1b, +0.026 exp7b), ClinicalBERT sees mostly variance reduction. Not a universal improvement.
20. ~~Hyperparameter optimisation for best-performing model (Optuna)~~ **DONE** - Exp14: Exp7a AUC 0.798 -> 0.831 (+0.033), Exp11 AUC 0.791 -> 0.822 (+0.031), Exp12 FuseMoE did not improve (0.749 vs 0.760)
21. ~~Rerun Exp14 best Exp7a and Exp11 configurations with full metrics~~ **DONE** - Reruns complete (Job 51546896). Optuna AUCs not reproduced; Exp7a rerun 0.770 (vs baseline 0.798), Exp11 rerun 0.789 (vs baseline 0.791), Exp12 rerun 0.697 (vs baseline 0.760).
22. ~~Resume Exp14 studies to complete remaining trials~~ **DONE** - Exp11 100/100 complete (Job 51546909). Exp7a 99/100, cancelled at 8h time limit (Job 51546904). New Exp11 best: Trial #81 (meanmax, AUC 0.825).
23. ~~Parameter importance analysis (fANOVA)~~ **DONE** - learning_rate most important for Exp7a (0.676), aggregator_type for Exp11 (0.519), weight_decay for Exp12 (0.293).
24. Exp7a baseline (AUC 0.798) confirmed as best result. Rerun Exp11 Trial #81 (meanmax, lr=3.73e-3, batch_size=4) - new best from resumed study, not yet confirmed.
25. External validation on further data if available
