# Analysis plan: clean-CV rerun and exp18 mixed-cohort experiment

Frozen on 2026-09-21, before any clean-CV or mixed-cohort result was computed.
The commit that adds this file is the reference point. Any later change is
listed under "Deviations" with its date and reason; nothing above that section
is edited after the first clean result exists.

## 1. Why

Every CV loop in exp1-17 and the HEP1 external-validation scripts chose the
early-stopping epoch, the restored weights, the Youden threshold (and, in exp1
and exp2, the `ReduceLROnPlateau` steps) using AUC on the outer held-out fold,
then reported that same fold. Table rows also used the best encoder
combination per row, again chosen on those folds. Both make the published
internal AUCs optimistic. This plan fixes the protocol for a full rerun and
for the new mixed-cohort experiment the supervisor requested.

## 2. Clean CV protocol (every single-cohort experiment)

- **Outer split:** 5 folds, iterative multi-label stratification on
  `outcome`, `focal`, `sex` (`shared.cv_splits.outer_splits(mode="multilabel")`),
  seed 42. This matches what the Methods already state.
- **Inner split:** 20% of each outer training fold, stratified on `outcome`,
  seed `42 + fold` (`shared.cv_splits.inner_val_split`). It is used only for
  early stopping, LR scheduling and the Youden threshold.
- **Early stopping:** criterion and patience unchanged from each experiment's
  config (inner-validation AUC). Best weights are restored, and the outer test
  fold is scored once with those weights.
- **Operating point:** Youden-J threshold chosen on the inner-validation
  predictions, logged with each fold, and applied unchanged to the outer test
  fold.
- **Hyperparameters:** frozen at the values in each `expN/config.py` at this
  commit. No further tuning.
- **Determinism:** `shared.determinism.enable_determinism` with seed 42 (per
  fold where the experiment already does so).
- **Output naming:** clean files carry the suffix `_sp-multilabel_iv20`.

## 3. Headline table rows: pre-specified encoders

Every row uses the headline encoder set: ChemBERTa (drug), ClinicalBERT
(mean-pooled, frozen text), EEG2Vec 256-d with the two-layer transformer
aggregator (EEG), late-fusion MLP.

| Row | Config |
|---|---|
| Clinical | `exp5a_chemberta` |
| Clinical + Text | `exp6a_clinicalbert_chemberta` |
| Clinical + EEG | `exp6b_eeg2vec_chemberta` (new config: EEG2Vec in place of SimpleCNN) |
| Clinical + Text + EEG (full model) | `exp7a` |
| Text | `exp1a_clinicalbert_chemberta` |
| EEG | `exp2_eeg2vec_chemberta_mlp` (new config) |
| Text + EEG | `exp3a_clinicalbert_chemberta` with EEG2Vec (new config) |
| Clinical, drug removed | `exp4a_mlp` |
| Clinical + Text, drug removed | `exp5b_clinicalbert` |
| Clinical + EEG, drug removed | `exp5c_eeg2vec` |

All other encoder, aggregator and fusion variants (SimpleCNN, EEG2Vec-128,
PubMedBERT, SMILES Transformer, FuseMoE, REVE-base) are reported only in the
Supplementary and labelled exploratory.

## 4. Metrics and tests (single-cohort)

- **Primary metric:** patient-level AUC pooled across folds by the existing
  random-effects method (per-fold bootstrap variance, method-of-moments tau2,
  Knapp-Hartung t-interval).
- **Secondary:** sensitivity, specificity, precision and balanced accuracy at
  the inner-validation threshold, pooled the same way.
- **Pairwise:** all-pairs DeLong on out-of-fold predictions, BH-corrected,
  exploratory (as before).
- **Decomposition (exp4a only):** a 2x2 of splitter (legacy outcome-only vs
  multilabel) x early stopping (outer fold vs inner 20%). It attributes the
  old-vs-clean change to the splitter, the leak fix and the 20% training-data
  loss. Descriptive only.

## 5. HEP1 external validation

Same six configurations as the current Table 3, with the same encoder set as
Section 3 (19-channel EEG2Vec for every EEG configuration), trained under the
clean protocol and applied to HEP1 as a five-fold ensemble. External AUC with a
patient-level percentile bootstrap CI (2000 resamples, seed 42). Reverse,
focal-matched and reduced-capacity analyses are rerun under the same protocol.

## 6. exp18 mixed-cohort experiment

**Cohorts:** Melbourne (deduplicated, n = 198) and HEP1 (n = 438), minus any
cross-cohort duplicates confirmed by the overlap audit (kept in Melbourne
only). pids are prefixed `MEL_` / `HEP_`.

**Configurations:** the six Table 3 configurations (Exp4a, Exp5a, Exp5b,
Exp5c, Exp6b, Exp7a) on the portable pipeline: ChemBERTa, mean-pooled
ClinicalBERT, 19-channel EEG2Vec.

**Splits:**
- Outer: 5-fold `StratifiedKFold` on the joint outcome x cohort key.
- Inner: 20% of each arm's training data, stratified on the same key.
- Seeds: 42-46 for the clinical and text configurations (Exp4a, Exp5a,
  Exp5b); 42-44 for the EEG configurations.

**Arms** (every arm scores every outer test patient in both cohorts):
- `mixed`: trained on the full outer training fold.
- `mel_only`: trained on its Melbourne patients only.
- `hep_only`: trained on its HEP1 patients only.
- `mixed_sizematched` (Exp4a only): for each test cohort, the mixed training
  fold subsampled (cohort proportions preserved) to the size of that cohort's
  own-only arm, 10 draws.

Preprocessors and any normalisation are fitted on each arm's own fit set.

**Primary metrics:**
- Overall: the cohort-stratified AUC, i.e. the probability that a seizure-free
  patient outranks a non-seizure-free patient **from the same cohort**. This
  equals the pair-weighted mean of the within-cohort AUCs and cannot be
  inflated by learning cohort base rates.
- Per cohort: out-of-fold AUC on that cohort's test patients, averaged over
  seeds, with a DeLong 95% CI per seed.

**Secondary metrics:**
- The whole-fold AUC requested by the supervisor (per fold, mean and SD),
  always reported next to the **cohort-only floor**: the AUC of a score equal to
  the training-fold seizure-free rate of the patient's cohort.
- Calibration per cohort: Brier score, calibration-in-the-large and
  calibration slope.

**Primary hypothesis tests** (two tests, Holm-adjusted):
1. Exp4a, Melbourne test patients: `mixed` vs `mel_only`.
2. Exp4a, HEP1 test patients: `mixed` vs `hep_only`.

Each uses the Nadeau-Bengio corrected resampled t-test on the per-(seed, fold)
differences in within-cohort AUC: J = 25 differences, variance inflation
`1/J + n_test/n_train`, df = J - 1.

**Exploratory:** every other contrast (other configurations, `mixed` vs the
other-cohort arm, size-matched draws), with paired DeLong per seed on
out-of-fold predictions. Reported without claims of significance.

**Sensitivity:** Exp4a, Exp5a and Exp5b, all arms, excluding all 29 HEP1
patients from the Royal Melbourne Hospital site (`RMH####`), whatever the
overlap audit finds.

**Sanity checks** (not tests):
- `mel_only` on Melbourne should be close to the clean single-cohort Exp4a.
- `mel_only` on HEP1 should be near chance, as in the existing external result.
- The cohort-only floor should be near 0.57 on the clinical tier.

## Deviations

**2026-09-21, seeds (decided before any M3 clean run).** Section 2 used a
single seed (42) for the single-cohort experiments. While building the
retrofit, local development runs (not reported, and not used to choose any
configuration) showed clean-protocol Exp4a fold AUCs from 0.24 to 0.73, and the
same legacy code gave fold-mean AUCs 0.03 apart on two machines. One seed
cannot separate tier differences of 0.02-0.05, so every single-cohort
configuration and variant, the HEP1 external validation and the REVE row now
run with seeds 42-46:

- outer split seed `s`, inner split seed `s + fold`, determinism seed `s`
  (per fold `s + fold` where an experiment already seeded per fold);
- file suffix `_sp-multilabel_iv20_s<seed>`;
- reported estimate: the mean over seeds of each seed's pooled estimate
  (Section 4), with the 95% CI as the mean of the per-seed CI bounds and the
  seed-to-seed SD shown alongside;
- HEP1 external AUC: mean over seeds of the external AUC of each seed's
  five-fold ensemble, with the patient-level bootstrap CI averaged the same way;
- all-pairs DeLong (exploratory): run per seed; report the median p-value and
  the number of seeds with BH-adjusted p < 0.05.

The change was motivated by variance, not by the direction of any dev result.
exp18's seeds (Section 6) are unchanged.

**2026-09-21, exp18 cross-cohort duplicates (decided before any M3 clean
run).** Section 6 excludes duplicates confirmed by the overlap audit before
pooling. The audit found no strong candidate (best report similarity 0.46;
the loose-match rate at the RMH site, 18 of 29, equals the rate at HEP1 sites
that cannot overlap, 244 of 409), and the data custodian's confirmation is
still pending. exp18 is therefore run on the unmodified pooled cohort now. If
any duplicate is confirmed, exp18 is rerun excluding those HEP1 patients
(variant `_dedup`) and that run becomes the primary analysis, with the
unmodified run reported alongside. The RMH-excluded sensitivity analysis
(Section 6) bounds the effect in the meantime.

**2026-09-21, audit fixes (before any M3 clean run).** An independent code
audit led to these implementation clarifications, none of which changes an
estimand:

- A non-finite early-stopping Youden threshold (flat or inverted ROC on a
  small inner set) falls back to 0.5.
- HEP1 external sensitivity and specificity in clean runs are labelled
  `_ownthr`, because their threshold is tuned on the scored cohort. They are
  descriptive only; external AUC stays the reported metric.
- Calibration-in-the-large is computed as the mean predicted minus the
  observed rate.
- The Nadeau-Bengio variance inflation uses n_test / n_train of the outer
  folds.
- Size-matched draws are compared with the own-cohort and mixed arms by
  paired DeLong per draw (exploratory).

## Addendum A (2026-09-22): exp19 serialised clinical text + language-model embeddings

Added before any exp19 result existed. Exploratory throughout. Reported in the
Supplementary unless it changes a conclusion.

**Motivation.** The supervisor suggested converting the tabular (and text)
data into text and embedding it with a language model. The template follows
the research group's second-regimen plan (Duong Nhu).

**Texts** (`shared/serialise_clinical.py`). One paragraph per patient, built
from the harmonised features with identical wording for both cohorts:

- Sex; age as a whole number; more than five pre-treatment seizures; focal or
  generalised onset.
- The ten history items: family history, febrile seizure, cerebral infection,
  birth trauma, head injury, drug abuse, alcohol abuse, cerebrovascular
  disease, psychiatric comorbidities, learning disability.
- CT/MRI findings (normal / abnormal but not epileptogenic / epileptogenic)
  and EEG findings (normal / abnormal but not epileptiform / epileptiform).
- The first ASM by name.
- Nothing about dose, outcome, reason for change, time to failure or later
  regimens, enforced by a unit test. Missing values get one fixed wording.

Three variants:

- **V1:** the paragraph.
- **V1-nodrug:** V1 without the drug sentence.
- **V2:** V1 followed by the free-text EEG report. V2 uses the existing text
  cohorts (117 Melbourne and 207 HEP1 reports).

**Encoders** (frozen; embeddings computed once and cached):

- PubMedBERT (`NeuML/pubmedbert-base-embeddings`) and ClinicalBERT
  (`medicalai/ClinicalBERT`): mask-aware mean pooling, 512 tokens.
- Llama-3.1-8B (`meta-llama/Llama-3.1-8B`, base): bfloat16 on CPU. Primary
  pooling is the mask-aware mean over final hidden states; the last-token
  state is a secondary pooling. Maximum 2048 tokens.

**Model.** One late-fusion MLP class for every exp19 configuration:

- each input is projected by Linear -> 64, ReLU, LayerNorm, dropout 0.3;
- the projections are concatenated;
- the head is Linear -> 64, ReLU, dropout, then Linear -> 2.

Hyperparameters equal the portable non-EEG models: AdamW (lr 1e-3, weight
decay 1e-4), batch 16, at most 80 epochs, patience 15, class-weighted
cross-entropy.

**Configurations.**

- Serialised:
  - A = V1
  - B = V1-nodrug + ChemBERTa SMILES
  - C = V1 + SMILES
  - D = V2

  Each runs with the three encoders; Llama also runs with last-token pooling.
- Tabular comparators, same model class, folds and seeds:
  - T4 = 19-feature clinical
  - T5a = clinical + SMILES
  - T6a = clinical + mean-pooled ClinicalBERT report embedding + SMILES

**Protocol.** Section 2's clean CV on the Melbourne cohort (multilabel outer
folds, inner 20% early stopping and threshold, seeds 42-46). Every fold's
model is also applied to HEP1 (five-fold ensemble per seed, as in Section 5).
exp18 gains configurations A and D for each encoder (arms and metrics as in
Section 6, seeds 42-46).

**Comparisons** (exploratory; paired DeLong per seed on the same patients
and a Nadeau-Bengio corrected t-test over seed x fold, no significance
claims):

- B vs T5a: does text encoding of the clinical features beat the tabular MLP?
- A vs T5a
- A vs C and B vs C: drug by name vs by structure
- D vs T6a: one document vs separate branches
- the Llama pooling variants

Both internal and HEP1 external AUCs are reported.

### A.1 (2026-09-22, pre-result): amendments after two independent design and code audits

No exp19 embedding of the real cohorts has been computed and no outcome
model has been fitted on these representations. These amendments replace
the corresponding parts of Addendum A.

**Revisions pinned.**

| Model | Revision |
|---|---|
| Llama-3.1-8B | d04e592bb4f6aa9cfee91e2e20afa771667e1d4b |
| Llama-3.1-8B-Instruct | 0e9e39f249a16976918f6564b8830bc894c89659 |
| PubMedBERT | b79526d6ef3645e0df4530322e266f24c829f5ef |
| ClinicalBERT | f7c7f65227cb311f33a79c24858d875876d478ac |

**Serialiser.**

- Input is validated: out-of-range codes, unknown drugs and missing columns
  raise errors instead of silently becoming "unknown".
- Binary flags are exact 0/1.
- The dose sentence is omitted because neither cohort records dose.

**Variants.**

- `v1` and `v1nodrug` keep one fixed "unknown" wording for missing values.
- `v1imp` and `v1nodrugimp` fill missing values before serialising, the way
  the paper's tabular preprocessor does (training-fold mode for binary and
  categorical features, training-fold mean age). Fills are computed from
  each fold's fit rows and applied to Melbourne and HEP1 alike. Fold modes
  differ (the CT/MRI mode alternates between normal and epileptogenic), so
  every distinct text any fold produces is embedded.
- `v2` = `v1` plus the EEG report.
- `rep` = the EEG report alone.

**Embeddings.**

- Each unique text is embedded once per encoder, keyed by its hash.
- BERT inputs longer than 512 tokens are chunked: 510-token windows, a
  mask-aware mean per window, then a token-weighted average across windows.
- Llama: at most 4096 tokens, BOS excluded from the mean, pooling in fp32,
  one text per forward pass.

**Inputs.**

- Every frozen embedding input (text, report, SMILES) is z-scored within
  each fold using the fit rows only.
- The paper preprocessor feeds T4, T5a and T6a.
- A new information-matched preprocessor, fitted on the fit rows, feeds
  T4-full and T5a-full:
  - binary features: mode-filled, plus a missing indicator;
  - age: z-scored, mean-filled, plus a missing indicator;
  - CT/MRI and EEG findings: one-hot over normal / non-epileptiform /
    epileptiform / missing.

**Configurations.** Each serialised configuration runs per encoder and
pooling. C is removed.

| Config | Inputs |
|---|---|
| A | [v1] |
| B | [v1nodrug, SMILES] |
| B-imp | [v1nodrugimp, SMILES] |
| E | [v1nodrug] |
| E-imp | [v1nodrugimp] |
| D | [v2] |
| D-split | [v1, rep] |
| T4 | paper clinical |
| T5a | paper clinical, SMILES |
| T6a | paper clinical, ClinicalBERT rep (chunked, computed here), SMILES |
| T4-full | information-matched clinical |
| T5a-full | information-matched clinical, SMILES |

**Estimators.**

- **MLP (primary):** LateFusionMLP with the portable hyperparameters.
- **PCA32:** each embedding input reduced to 32 components fitted on the
  fit rows, then the MLP. Not run for configurations without embedding
  inputs.
- **LR:** L2 logistic regression on the concatenated standardised inputs.
  C is chosen from {0.001, 0.01, 0.1, 1, 10} by AUC on the early-stopping
  set, and the model is fitted on the fit rows.

**Primary contrast** (the only confirmatory one): B vs T5a-full, MLP
estimator, for PubMedBERT-mean, ClinicalBERT-mean and Llama-mean. Holm
adjustment over these three.

- **Internal:** Nadeau-Bengio corrected t over seed x fold AUC
  differences, with a 95% CI, read against a margin of 0.05:
  - equivalent if the CI lies within +/-0.05;
  - text better if the lower bound is above 0;
  - tabular better if the upper bound is below 0;
  - otherwise inconclusive.
- **External (HEP1):** paired patient-level bootstrap (2000 resamples) of
  the AUC difference on seed-averaged ensemble scores. Run on all HEP1
  patients, on the complete-case subgroup (no missing value among the 16
  features) and on the seen-drug subgroup (ASM present in Melbourne
  training).

**Descriptive only:**

- A vs T5a
- B-imp vs T5a
- E vs T4-full
- E-imp vs T4
- D vs D-split
- D-split vs T6a
- Llama mean vs last-token pooling
- the three estimators against one another
- a fold-internal cohort probe per embedding

**Zero-shot baseline** (descriptive). Llama-3.1-8B-Instruct with its chat
template, no training.

- System message: "You are an experienced epileptologist."
- User message: the v1 text, then a blank line, then "Will this patient be
  seizure-free for at least 12 months on this first antiseizure medication?
  Answer Yes or No."
- Score: P("Yes") / (P("Yes") + P("No")) from the next-token distribution.
- Reported: AUC with a DeLong CI on all Melbourne patients and on HEP1.

**exp18.** Configurations A and D per encoder (MLP) are added, with the
same-class comparators T5a-full and T6a. Standardisation, PCA and the
preprocessors are fitted on each arm's fit rows. Arms and metrics follow
Section 6, seeds 42-46.

**Placement.** Main text vs Supplementary is decided with the supervisor
after the results. No promotion rule is pre-specified.

## Addendum B (2026-09-28): post-audit corrections and the refit protocol

Written after the first clean M3 rerun (Sections 2-6 plus Addendum A) had
produced results, and after three independent read-only audits of that run.
Every item below is therefore a **post-hoc deviation**, decided with those
results in view. They are listed with the reason for each so a reader can
judge them. Everything is committed before any run that uses them.

### B.1 Melbourne outcome polarity

The Melbourne CSV codes `outcome` 1/2. Every script mapped raw 1 to
not-seizure-free and raw 2 to seizure-free (`shared/cohort.py`). The only
source for that convention was a code comment. The data dictionary that came
with the data (`List_Missing_clinical_factors_07Nov2025.xlsx`, sheet
`ASM_regimen`, field `outcome_12m`) says "1 = success, 2 = failure", where
success is seizure-free for the first 12 months while still taking the
regimen, and failure is not seizure-free or a switch to or addition of
another ASM within 12 months. For the 38 Melbourne patients who also appear
in that workbook the raw CSV value agrees with the dictionary field in 19 of
21 (raw 1) and 15 of 15 (raw 2) cases. The audits also found that every
univariate association with established predictors ran the wrong way in
Melbourne and the right way in HEP1, and that every Melbourne-to-HEP1 AUC sat
below 0.5.

From here on raw 1 = seizure-free (label 1) and raw 2 = not seizure-free
(label 0). Every prediction made before this addendum used the inverted
Melbourne label. Internal AUCs are unaffected in expectation (AUC is symmetric
under a consistent flip and retraining), but every cross-cohort estimate,
threshold, calibration, recommendation and descriptive table is. All are
rerun; nothing is re-mapped after the fact. Confirmation from the data
custodian is still requested.

Check after the flip (`python -m shared.polarity_check`, aggregate rates of
label 1 with vs without each predictor of drug resistance):

| Predictor | Melbourne | HEP1 |
|---|---|---|
| >5 pre-treatment seizures | 0.37 vs 0.55 | 0.32 vs 0.35 |
| psychiatric history | 0.37 vs 0.55 | 0.26 vs 0.36 |
| epileptiform EEG | 0.47 vs 0.52 | 0.25 vs 0.39 |
| abnormal imaging | 0.50 vs 0.53 | 0.32 vs 0.35 |
| head trauma | 0.53 vs 0.51 | 0.23 vs 0.34 |
| learning disability (Melbourne n=3) | 0.67 vs 0.51 | 0.36 vs 0.33 |

With the corrected label the main predictors lower seizure freedom in both
cohorts, as expected.

### B.2 Refit protocol (primary from here on)

The audit of the first clean run found that the Section 2 inner split is too
small to choose an epoch: 13 to 32 patients, and 18% of early stops chose
epoch 1 or earlier, so those folds scored an almost untrained model. The
protocol for every experiment becomes, per outer fold (`train_idx` = the whole
outer training fold):

1. Five stratified inner folds of `train_idx`
   (`StratifiedKFold(5, shuffle=True, random_state=s + fold)`) on the same
   stratification labels as the outer split. All preprocessing is refitted on
   each inner-train set.
2. Each inner model trains with the frozen loop and hyperparameters,
   patience included, and records its inner-validation probabilities at every
   epoch. A model that stops early carries its last recorded probabilities
   forward to the longest inner run.
3. Criterion per epoch: AUC of the pooled inner out-of-fold probabilities over
   all of `train_idx`; for exp18 mixed arms the cohort-stratified pooled AUC
   (B.6). The curve is smoothed with a centred 3-epoch moving average and the
   selected epoch E* is its argmax (earliest on ties).
4. Threshold: Youden J on the pooled inner out-of-fold probabilities at E*
   (0.5 if non-finite).
5. Refit: preprocessing and class weights on all of `train_idx`, a fresh model
   trained for exactly E* epochs with the same seeds, E* not rescaled.
   exp1 and exp2 replay the per-epoch median learning rate of the inner runs
   in place of `ReduceLROnPlateau`. exp3b replays the inner temperature
   schedule by epoch.
6. The outer test fold is scored once.

Logistic regression estimators (exp19) choose C the same way (pooled inner
out-of-fold AUC per C), then refit. File suffix `_sp-multilabel_rf5_s<seed>`.
Seeds 42-46 as before.

The refit protocol is the primary analysis. The Section 2 inner-split results
(`_sp-multilabel_iv20_s<seed>`) are kept unchanged and reported in the
Supplementary as the pre-registered protocol, next to the refit results, so
the effect of this post-hoc change is visible. The exp4 decomposition grows to
splitter x {no inner selection, inner split, refit}.

### B.3 Clinical encoding

`lesion` and `eeg_cat` (codes 1/2/3) were collapsed to one normal/abnormal
flag each, which discards the epileptiform vs non-epileptiform distinction
that carries most of the HEP1 signal. They become 3-level one-hot (mode
imputation before encoding, fitted on training rows). Age stays as four bins
after mean imputation; binary features keep mode imputation. The clinical
input grows from 19 to 23 dimensions.

### B.4 Cross-cohort feature set

`drug`, `alcohol` and `focal` are constant in HEP1 (every patient "No", "No",
"focal"), so in any model that sees both cohorts they carry no within-HEP1
information and act as cohort indicators. exp18 and every HEP1 transfer
script (forward, reverse, focal, reduced, EEG) use the remaining 13 features.
Melbourne-only experiments keep all 16.

### B.5 HEP1 harmonised outcome (sensitivity analysis)

HEP1's provided `outcome` (1 = still on the first regimen and seizure-free)
stays the primary label. Among outcome-0 patients, 52 have `end_date` at least
12 months after `start_date`; if `end_date` is the failure date, those patients
completed 12 months on the regimen, which is a success under the Melbourne
definition. Sensitivity label `harmonised12`: outcome 0 with
`end_date - start_date >= 365.25` days is recoded to 1, and rows with a
non-positive duration are dropped. Run for exp18, the HEP1 forward and reverse
scripts and exp19 external (file tag `_h12`). Seizure-free patients have no
follow-up date, so censoring cannot be applied. The meaning of `end_date` is
to be confirmed by the data custodian.

### B.6 exp18 selection and test ratio

The mixed arm selects its epoch on the cohort-stratified pooled inner AUC
(pair-weighted mean of within-cohort AUCs, Section 6), so selection cannot
reward separating the cohorts. The Nadeau-Bengio ratio uses n_test / n_train
of the arm being tested rather than of the pooled outer fold.

### B.7 exp19

- Zero-shot scores are computed in float32 from the final hidden state and the
  two `lm_head` rows. The bf16 output layer gave about 25 distinct scores for
  198 patients.
- Configuration D uses segment-balanced pooling: the mean of the paragraph
  embedding and the report embedding, each embedded on its own. The
  token-mean D is kept as `D-tok` (descriptive).
- Qwen3-Embedding-8B (last-token pooling, L2 normalised, the model card's
  instruction prefix) is added as a secondary encoder for parity with
  Hegselmann et al. 2025. It is outside the Holm family of A.1.
- Equivalence is read from a 90% Nadeau-Bengio CI against +/-0.05 (two
  one-sided tests), Holm-adjusted over the three primary encoders.
  Superiority still uses the 95% CI.

### B.8 Seed averaging and CIs

The point estimate stays the mean over seeds of each seed's pooled estimate.
The CI changes from the mean of per-seed CI bounds to a patient-level
bootstrap shared across seeds: each replicate resamples patients once, applies
that resample to every seed's file (each seed keeps its own fold assignment),
recomputes each seed's pooled estimate and averages over seeds. 1000
replicates, 2.5 and 97.5 percentiles. Seed-to-seed SD is still reported.

### B.9 (2026-09-28, before any Addendum B run): amendments after an independent audit

An independent read-only audit compared B.1-B.8 with the code. These
clarifications and corrections are made before any run under Addendum B:

- **B.2 wording.** Inner folds are stratified on the outcome alone (the
  outer split is multilabel on outcome, focal and sex); exp18 uses its joint
  outcome x cohort key for both. "With the same seeds" means the per-fold
  determinism seed is set once before the inner runs; the refit model is
  initialised from the same seeded stream after them (reproducible, not
  reseeded). For logistic regression, ties in pooled inner AUC go to the
  smallest C.
- **B.4 scope.** exp18's serialised-text configurations use a paragraph that
  omits the seizure-type sentence and the drug and alcohol items (variant
  `v1xc`), for the same reason as the tabular drop. exp19's own external
  validation (Melbourne to HEP1, Supplementary) keeps all 16 features in text
  and table, as in Addendum A.
- **B.5 scope.** The harmonised label is run for the HEP1 forward, forward
  EEG and reverse scripts and for every exp18 configuration, under the refit
  protocol only; exp19 external does not run it. Durations are whole days, so
  the rule is a regimen lasting 366 days or more.
- **B.6 scope and ratio.** The cohort-stratified selection applies to the
  refit protocol (the mixed arm and the size-matched draws); the inner-split
  rerun keeps Section 6's pooled-AUC early stopping. The Nadeau-Bengio ratio
  for each test cohort is n_test / n_train of that cohort's own share of each
  outer fold (the own-cohort arm's training set), and the same ratio is used
  for the refit (primary), inner-split and harmonised-label tests.
- **B.7 details.** D-tok contrasts carry p-values but are descriptive. The
  Qwen3-Embedding instruction follows the model card's
  "Instruct: ... Query: " format with our own task sentence. The primary
  verdict applies Holm over the three encoders to both families: equivalence
  when the Holm-adjusted two one-sided tests give p < 0.05, otherwise
  superiority when the Holm-adjusted two-sided Nadeau-Bengio test gives
  p < 0.05, otherwise inconclusive. Zero-shot scores are written to
  `zeroshot_fp32_<cohort>.csv`. The embeddings and the zero-shot baseline run
  on the laptop; exp19's models and exp18's text configurations run in the M3
  array from the copied embedding stores (a partial laptop run was stopped for
  load and its files set aside unused).
- **B.8 details.** Within each bootstrap replicate a seed's pooled estimate
  is the random-effects re-pool of the resampled fold estimates with that
  seed's fold variances held fixed. exp18 keeps the Section 6 per-seed DeLong
  intervals.
- **External operating point.** In clean HEP1 runs, external sensitivity and
  specificity are reported at a transported threshold, the mean over the five
  folds of each fold's inner Youden threshold, applied to the five-fold
  ensemble score; the scored cohort's own Youden point is kept as `_ownthr`
  (descriptive).

### B.10 (2026-10-02): outcome coding confirmed

The data custodian (D. Nhu) checked the code and confirmed the Melbourne
outcome coding: raw 1 = seizure-free, raw 2 = not seizure-free. This is the
mapping B.1 adopted from the data dictionary and every Addendum B run used,
so no result changes.

### B.11 (2026-10-05, after the results): confidence-interval method and audit notes

Three independent audits of the complete rerun reproduced every reported
estimate. One found that the B.8 interval, a patient bootstrap shared across
seeds with each seed's models and fold variances held fixed, is conditional
on the fitted models: it leaves out retraining variability and the
Knapp-Hartung small-k inflation, is about half the width of the per-seed
intervals (median ratio 0.49), and for some near-chance rows excludes 0.5
where no per-seed interval does (exp7a's seed-to-seed range, 0.52-0.67, also
falls outside its B.8 interval). Decided with the results in view, so post
hoc:

- **Reported 95% CI:** the mean over seeds of each seed's random-effects
  Knapp-Hartung interval (the 2026-09-21 deviation's method). The B.8
  bootstrap is kept in the outputs as a secondary, conditional interval
  (`ci_low_boot`, `ci_high_boot`); the seed-to-seed SD is reported alongside.
- **HEP1 external CIs:** the mean over seeds of each seed's patient-level
  percentile bootstrap interval (2000 resamples), as the 2026-09-21
  deviation specified; the shared-resample interval is kept as secondary.
- **Paired comparisons in Figure 2:** a paired t-test on the five seed-level
  mean fold AUCs (the 25 seed x fold pairs are not independent).
- **Notes for reporting:** exp18's EEG cohorts (148 and 108 Melbourne
  patients, 19-channel EEG cache, 13 clinical variables) are not the main
  table's cohorts and are not compared with it. exp9's `aggregator_depth_0`
  is the same model as `aggregator_attention` by construction and is reported
  once. GPU results differ between M3 node types at the level of individual
  predictions, so run-to-run differences of about 0.02 AUC are noise.

### B.12 (2026-10-09, after the results): inner-fold stratification labels

A configuration audit on 2026-10-09, after every result had been produced,
re-checked the inner folds against the plan. B.2 step 1 specifies inner folds
"on the same stratification labels as the outer split" (outcome, focal and
sex); B.9 amended this before any Addendum B run to the outcome alone, with
exp18 using its outcome x cohort key. The code matches B.9, not the B.2 text:
`shared/epoch_selection.py:56-65` builds
`StratifiedKFold(5, shuffle=True, random_state=seed + fold)` (seed = the
active repeat seed) on the labels each runner passes, which are the outcome
vector (for example `exp4_baseline/training.py:339`,
`exp7_all_modalities/training.py:507`, `exp1_fusion/training.py:393`), and
`exp18_mixed_cohort/run_experiments.py:141` passes the outcome x cohort key
built at `exp18_mixed_cohort/data_pipeline.py:128`; the outer split stays
multilabel (`shared/cv_splits.py:94-114`; exp18 joint key at
`run_experiments.py:170-171`). Recorded here so that the Methods statement has
a single source. No rerun is planned for this item: the inner folds choose only
the epoch count and the threshold, and the outer test folds are unaffected.

## Addendum C (2026-10-09, before any EEG rerun): EEG inputs, pretrained encoders and the EEG rerun

### C.1 Defects in the EEG inputs of every run to date

Found 2026-10-09 while compiling the training configurations
(`findings/training_configurations.md`):

- The EEG cache read by every EEG experiment (`outputs/eeg_cache/processed_eeg.pkl`)
  held 27 channels: the 19 standard 10-20 EEG channels plus EMG+/EMG-, PG1/PG2,
  ECG+/ECG- and A1/A2, which the EDF headers type as EEG. Channel selection was by
  header type, not by name.
- The windows were in volts and no amplitude normalisation was applied, so the
  BatchNorm layers of the trained encoders saw values of the order of 1e-5 against an
  epsilon of 1e-5 (EEG2Vec, SimpleCNN) or 1e-3 (EEGNet).
- The "LaBraM" row of the encoder comparison was braindecode 1.2's `Labram`
  architecture with 2 layers, 4 heads and a 128-dimensional embedding, trained from
  random initialisation; the published pretrained weights were never loaded.
- The REVE features (`outputs/reve_features_*.npz`, version 1) were extracted from a
  separate 19-channel cache with a per-window z-score and a 5-SD clip.

Every result that depends on EEG (exp2, exp3, exp5c, exp6b, exp7a/7b, exp9, exp11,
exp15, exp16, exp17, the HEP1 EEG and reduced models, exp18 Exp5c/Exp6b/Exp7a) is
superseded; the paper's EEG statements are marked pending until the rerun.

### C.2 Version-2 EEG cache (`shared/eeg_cache.py`)

- Channels: exactly the 19 standard 10-20 channels selected by name
  (FP1 FP2 F7 F3 FZ F4 F8 T7 C3 CZ C4 T8 P7 P3 PZ P4 P8 O1 O2; legacy names
  T3/T4/T5/T6 mapped to T7/T8/P7/P8); a recording missing any of them is skipped.
- Units: the EDF physical dimension must be a voltage unit; data are stored in
  microvolts.
- A leading flat segment (consecutive 100-s chunks whose standard deviation, pooled
  over all channels and samples, is at most 0.01 microvolts) is skipped before
  resampling, as in the supervisor's pipeline.
- 200 Hz; 0.1-75 Hz zero-phase FIR; notch 50 Hz (Melbourne) or 60 Hz (HEP1);
  the first 300 s skipped, the next 1200 s used, recordings shorter than 600 s
  skipped; 10-s windows, at most 120, zero-padded with a mask.
- Amplitude conventions applied at load time (`load_cache(path, convention)`):
  `zscore_window` (per window and channel, mean 0 and SD 1, SD floor 1e-6
  microvolts, no clipping) for every trained-from-scratch encoder and for REVE;
  `labram` (microvolts divided by 100) for the pretrained LaBraM.
- Each cache carries a sidecar with aggregate statistics only (no patient ids).

Caches built on M3 on 2026-10-09 (`shared/eeg_cache.py` at commit 46cc3f5,
`--expect-files 157` and `98`):

| Cohort | EDF files | CSV patients with an EDF | Kept | Skipped | Leading flat segment | Median per-window SD (microvolts) | Full length (120 windows) | Fewest windows |
|---|---|---|---|---|---|---|---|---|
| Melbourne (Alfred) | 157 | 148 (9 files match no CSV patient) | 148 | 0 | 50 | 8.59 | 114 | 37 |
| HEP1 | 98 | 96 (one patient with three files, the first kept) | 95 | 1 (shorter than 600 s) | 0 | 2.89 | 86 | 90 |

Frozen cohorts for the rerun: Melbourne EEG cohort 148 patients (72 with outcome 1
and 76 with outcome 0 under the B.1 coding), of whom 108 also have a usable EEG report
(the text + EEG and four-modality cohort) and 83 of those have focal epilepsy; HEP1
EEG cohort 95 patients (32 with outcome 1 and 63 with outcome 0 under HEP1's own
coding). `shared/verify_oof.py` requires these counts.

### C.3 Pretrained EEG encoders as frozen per-window features

- LaBraM-base: the Hugging Face snapshot `braindecode/labram-pretrained` at revision
  0563b6c626e7b40d9a36653b763715db94d945d7, whose 221 tensors are bit-identical, by
  name, to the official `labram-base.pth` (sha256 7c505838...bb57c37c; checked by
  `shared/labram_pretrained.py verify-official`). The 19-channel model keeps the
  pretrained channel embeddings selected by name, with the four temporal channels
  under the legacy names T3/T4/T5/T6 used by the official clinical fine-tuning runs,
  and the temporal embedding cut to ten 1-s patches plus [CLS]. Input microvolts/100.
  Per-window feature: the mean over the 190 patch tokens through a parameter-free
  LayerNorm (`use_mean_pooling=True`, the official fine-tuning default); the [CLS]
  feature is stored alongside. Files `outputs/labram_features_v2_<cohort>.npz`
  (200-dimensional).
- REVE-base (`brain-bzh/reve-base`): attention-pooled 512-dimensional feature per
  window from the `zscore_window` cache. Files `outputs/reve_features_v2_<cohort>.npz`.
- Both feature sets are checked against their cache (same patients, same padding
  masks) before a run (`python -m shared.eeg_features check`).
- The architecture of braindecode 1.6.0.dev1024's `Labram` is carried in
  `shared/vendor/labram.py` (BSD-3-Clause) for the training environment, which cannot
  install that release; a reference fixture recorded from the upstream model is
  reproduced by the vendored copy to 1e-5 of the output scale.

### C.4 Configurations

- exp9 encoder comparison on the version-2 cache: `baseline_simplecnn_transformer`,
  `encoder_eegnet`, `encoder_eeg2vec`, `encoder_labram_scratch` (the 2-layer
  architecture trained from scratch, formerly `encoder_labram`),
  `encoder_labram_pretrained_frozen` (LaBraM-base features, 200 wide) and
  `encoder_reve_frozen` (REVE-base features, 512 wide); the two frozen arms go through
  the same window aggregator and head as the others (encoder type `precomputed`) and
  replace the standalone REVE row. The aggregator's token width equals each arm's
  embedding; its output is 256 throughout. A fine-tuned LaBraM-base arm with the
  official recipe (learning rate 5e-4, weight decay 0.05, layer decay 0.65, drop path
  0.1, warm-up) is scheduled after the frozen-feature results.
- exp15 runs on each feature set (`--feature-set reve_v2` and `labram_v2`), both
  ASM-balance modes; outputs carry the feature-set tag.
- exp5 and exp6 rerun their EEG rows only (5c, 6b); every other configuration keeps
  its clean-rerun results.
- Training settings are otherwise those of Addendum B (refit protocol, five seeds
  42-46, exp18 EEG configurations seeds 42-44).

### C.5 Rerun

- Items: exp2, exp3, exp5 (5c), exp6 (6b), exp7a, exp7b, exp7a_stratbatch, exp9 (the
  six encoder arms of C.4: SimpleCNN, EEGNet, EEG2Vec, LaBraM from scratch, LaBraM-base
  frozen, REVE-base frozen), exp15 (both feature sets), exp16, exp17, hep_eeg,
  hep_reduced, exp18 Exp5c/Exp6b/Exp7a. As in the earlier rerun, the deferred items
  are not submitted: exp9's frozen-SimpleCNN, aggregator and embedding-size ablations,
  the exp11 grid, and the harmonised-HEP1-label EEG items (hep_eeg_h12, exp18_h12
  Exp5c/Exp6b/Exp7a); `RUN_DEFERRED=1` runs them. The task list holds 373 work items
  (190 of them on the CPU list); the pending, non-deferred ones are submitted
  (`rerun_clean.sh pending-array`, `pending-array-cpu`).
- Before submission `rerun_clean.sh archive-eeg` moves every output that depended on
  the previous cache, the previous caches, the version-1 REVE features, the derived
  thesis tables and the done markers of the EEG tasks to
  `outputs/_archive_eeg_defect_20261009/`; it halts if any such file is newer than the
  version-2 cache.
- Every submitted item's entry point was exercised end to end on this laptop's subset
  of the data with the smoke mode (`--smoke`: one outer fold, two inner folds, two
  epochs, outputs tagged `_smoke` and never read as results) before submission.
- `shared/verify_oof.py` and the expected-file manifest name the new arms and files
  and the counts of C.2.
