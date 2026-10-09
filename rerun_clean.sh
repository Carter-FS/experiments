#!/bin/bash
# Clean-CV rerun (docs/analysis_plan_clean_rerun_exp18.md): one work item per call.
#
#   bash rerun_clean.sh preflight         # check env, data, caches and repos before submitting
#   bash rerun_clean.sh list              # work items "task:seed", in array-index order
#   bash rerun_clean.sh list-cpu          # the items that need no GPU (rerun_clean_cpu.slurm)
#   bash rerun_clean.sh <task>:<seed>     # run one item (skipped if already done)
#   bash rerun_clean.sh smoke <task>      # any task: 1 outer fold, 2 inner folds, 2 epochs, outputs to /tmp or suffixed _smoke
#   bash rerun_clean.sh verify            # gate: verify_oof + expected files + exp18 + HEP
#   bash rerun_clean.sh archive-iv20      # move pre-2026-09-28 inner-split outputs aside (once)
#   sbatch rerun_clean.slurm              # every item as a slurm array (see that file)
#
# PROTOCOL selects the selection protocol (docs/analysis_plan_clean_rerun_exp18.md):
#   PROTOCOL=refit (default)  inner 5-fold epoch selection + refit (Addendum B.2),
#                             files suffixed _sp-multilabel_rf5_s<seed>; also the
#                             harmonised-HEP1-label sensitivity items (B.5, _h12)
#                             and the splitter x protocol decomposition on exp4a
#   PROTOCOL=innersplit       the pre-registered 20% inner split (Section 2),
#                             _sp-multilabel_iv20_s<seed>, rerun under the
#                             corrected outcome label and encoding (B.1, B.3);
#                             run `archive-iv20` first so no stale file remains
# Both run under each repeated-CV seed 42-46 (--cv-seed). exp18 keeps its own
# seed sets (EEG configurations 42-44, so those items are not listed for
# seeds 45/46). If
# outputs/exp18_mixed_cohort/confirmed_duplicates.txt exists (HEP1 pids the
# data custodian confirmed as Melbourne patients; patient-level, so it lives
# in the gitignored outputs), exp18 excludes them and writes the _dedup
# variant, which the analysis then treats as primary; adding the file after a
# first run makes those exp18 items not-done again, so resubmitting the array
# (or `verify`) picks them up without FORCE. A finished item writes
# outputs/_clean_rerun_v2/<protocol>/<task>_s<seed>.done (the first run's
# markers in outputs/_clean_rerun/ belong to the superseded labels and are
# ignored); rerunning a done item is a no-op
# unless FORCE=1 (which also makes exp18 overwrite its outputs). Set
# ASM_EXPERIMENTS_DIR if thesisStandalone is not cloned inside this repo.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
PY="$REPO_DIR/.venv-others/bin/python"
OUT=outputs
PROTOCOL="${PROTOCOL:-refit}"
case "$PROTOCOL" in
    refit) SEL=rf5; CV_SEL=(--refit-folds 5); EXP18_SEL=(--refit-folds 5); EXP18_TAG=_rf5 ;;
    innersplit) SEL=iv20; CV_SEL=(--inner-val 0.2); EXP18_SEL=(); EXP18_TAG="" ;;
    *) echo "PROTOCOL must be refit or innersplit, not $PROTOCOL" >&2; exit 2 ;;
esac
DONE="$OUT/_clean_rerun_v2/$PROTOCOL"
THESIS="$REPO_DIR/thesisStandalone"
SEEDS=(42 43 44 45 46)
EXP18_EEG_SEEDS=" 42 43 44 "
EXP18_DUPLICATES="$OUT/exp18_mixed_cohort/confirmed_duplicates.txt"

# The variant an exp18 item must have written for the current duplicates file.
exp18_ready () {
    local task="$1" seed="$2" cfgs variant=""
    [[ -f "$EXP18_DUPLICATES" ]] || return 0
    case "$task" in
        exp18_noRMH) cfgs="Exp4a Exp5a Exp5b"; variant="${EXP18_TAG}_noRMH" ;;
        exp18_h12_*) cfgs="${task#exp18_h12_}"; variant="${EXP18_TAG}_h12" ;;
        exp18_*) cfgs="${task#exp18_}"; variant="$EXP18_TAG" ;;
        *) return 0 ;;
    esac
    for cfg in $cfgs; do
        [[ -f "$OUT/exp18_mixed_cohort/predictions_${cfg}${variant}_dedup_seed${seed}.csv" ]] || return 1
    done
}

exp18 () {
    local extra=()
    [[ "${FORCE:-0}" == 1 ]] && extra+=(--force)
    [[ -f "$EXP18_DUPLICATES" ]] && extra+=(--exclude-hep-pids "$EXP18_DUPLICATES")
    # ${extra[@]+...}: expanding an empty array trips `set -u` on bash < 4.4.
    "$PY" -m exp18_mixed_cohort.run_experiments "$@" ${EXP18_SEL[@]+"${EXP18_SEL[@]}"} ${extra[@]+"${extra[@]}"}
}
exp19 () {
    local extra=()
    [[ "${FORCE:-0}" == 1 ]] && extra+=(--force)
    [[ "$PROTOCOL" == refit ]] && extra+=(--refit-folds 5)
    "$PY" -m exp19_serialised_clinical.run_experiments "$@" ${extra[@]+"${extra[@]}"}
}
export ASM_EXPERIMENTS_DIR="${ASM_EXPERIMENTS_DIR:-$REPO_DIR}"

# exp9 and exp11 run one item per ablation / base x aggregator: the refit
# protocol costs about six times the inner split and the whole experiments
# would exceed the 16 h limit.
EXP19_TEXT_CONFIGS=(A B B-imp E E-imp D D-tok D-split)
EXP19_TABULAR_CONFIGS=(T4 T5a T6a T4-full T5a-full)
EXP18_TEXT_CONFIGS=(S19A_pubmedbert S19D_pubmedbert S19A_clinicalbert S19D_clinicalbert
                    S19A_llama31_8b S19D_llama31_8b LF_T5a-full LF_T6a)
EXP9_ABLATIONS=(baseline_simplecnn_transformer encoder_eegnet encoder_labram_scratch encoder_eeg2vec
                encoder_labram_pretrained_frozen encoder_reve_frozen encoder_frozen
                aggregator_attention aggregator_maxpool aggregator_meanmax aggregator_lstm
                aggregator_depth_0 aggregator_depth_1 aggregator_depth_4 embed_dim_64 embed_dim_128)
TASKS=(
    exp4 exp5 exp6 exp1 exp2 exp3
    exp7a exp7b exp7a_stratbatch exp15 exp16 exp17
    exp11_exp3a_transformer exp11_exp3a_meanmax exp11_exp6b_transformer exp11_exp6b_meanmax
    exp11_exp7a_transformer exp11_exp7a_meanmax
    "${EXP9_ABLATIONS[@]/#/exp9_}"
    hep_forward hep_eeg hep_reverse hep_focal hep_reduced reve
    exp18_Exp4a exp18_Exp5a exp18_Exp5b exp18_Exp5c exp18_Exp6b exp18_Exp7a exp18_noRMH
)
if [[ "$PROTOCOL" == refit ]]; then
    TASKS+=(exp4_decomp hep_forward_h12 hep_eeg_h12 hep_reverse_h12
            exp18_h12_Exp4a exp18_h12_Exp5a exp18_h12_Exp5b exp18_h12_Exp5c exp18_h12_Exp6b exp18_h12_Exp7a)
fi
# Added 2026-09-28 after the first arrays were queued: appended so the array
# indices of every earlier item stay the same. exp19 (Addendum A/B.7):
# tabular comparators, then the serialised-text configurations per encoder
# (needs outputs/exp19_embeddings, built on the laptop and copied up); then
# exp18's text and late-fusion configurations.
TASKS+=(exp19_tabular exp19_pubmedbert exp19_clinicalbert exp19_llama31_8b exp19_qwen3_embed_8b
        "${EXP18_TEXT_CONFIGS[@]/#/exp18_}")
[[ "$PROTOCOL" == refit ]] && TASKS+=("${EXP18_TEXT_CONFIGS[@]/#/exp18_h12_}")

# Both balance modes, as in the legacy rerun. CV is set per item (seed).
balanced () {
    local module="$1"; shift
    for mode in none weighted; do
        "$PY" -m "$module.run_experiments" "$@" --asm-balance "$mode" \
            --log-predictions --deterministic "${CV[@]}" || return 1
    done
}

run_task () {
    local task="$1" seed="$2"
    CV=(--splitter multilabel "${CV_SEL[@]}" --cv-seed "$seed")
    local H12=(--hep-outcome harmonised12)
    case "$task" in
        exp1) balanced exp1_fusion ;;
        exp2) balanced exp2_fusion \
              && balanced exp2_fusion --eeg-encoder eeg2vec --smiles-model chemberta --fusion mlp ;;
        exp3) balanced exp3_fusion ;;
        exp4) balanced exp4_baseline ;;
        exp5) balanced exp5_clinical_fusion ;;
        exp6) balanced exp6_clinical_triple ;;
        exp11_*) local spec="${task#exp11_}"
                 balanced exp11_eeg_upgrade --base "${spec%%_*}" --aggregator "${spec#*_}" ;;
        exp9_*) "$PY" -m exp9_eeg_investigation.run_experiments --experiment "${task#exp9_}" \
                    --log-predictions --deterministic "${CV[@]}" ;;
        exp7a)
            for mode in none weighted; do
                "$PY" -m exp7_all_modalities.run_experiments --mode predictions --asm-balance "$mode" \
                    --deterministic --output_dir "$OUT/exp7_predictions" "${CV[@]}" || return 1
            done ;;
        exp7b)
            for mode in none weighted; do
                "$PY" -m exp7_all_modalities.run_experiments --mode predictions --exp 7b --asm-balance "$mode" \
                    --deterministic --output_dir "$OUT/exp7_predictions" "${CV[@]}" || return 1
            done ;;
        exp7a_stratbatch)
            "$PY" -m exp7_all_modalities.run_experiments --mode predictions --asm-balance stratified_batch \
                --deterministic --output_dir "$OUT/exp7_predictions" "${CV[@]}" ;;
        exp15)
            for fs in reve_v2 labram_v2; do
                for mode in none weighted; do
                    "$PY" -m exp15_reve_quad_mlp.run_experiments --mode predictions --asm-balance "$mode" \
                        --feature-set "$fs" --seed "$seed" --output-dir "$OUT/exp15_predictions" "${CV[@]}" || return 1
                done
            done ;;
        exp16) "$PY" -m exp16_reduced_capacity.run_experiments --mode predictions --seed "$seed" \
                   --output-dir "$OUT/exp16_predictions" "${CV[@]}" ;;
        exp17) "$PY" -m exp17_focal_only.run_experiments --mode predictions --seed "$seed" \
                   --output-dir "$OUT/exp17_predictions" "${CV[@]}" ;;
        exp4_decomp)
            # splitter x {no inner selection, inner split, refit} on exp4a, all on
            # this host, kept out of exp4_predictions so the legacy cell cannot
            # replace the archived file (analysis plan B.2).
            for cell in "legacy --inner-val 0" "legacy --inner-val 0.2" "legacy --refit-folds 5" \
                        "multilabel --inner-val 0" "multilabel --inner-val 0.2" "multilabel --refit-folds 5"; do
                set -- $cell
                "$PY" -m exp4_baseline.run_experiments --model mlp --log-predictions --deterministic \
                    --splitter "$1" "$2" "$3" --cv-seed "$seed" \
                    --predictions-dir "$OUT/exp4_decomposition" \
                    --output "$OUT/exp4_decomposition/results_$1_${2#--}$3_s$seed.json" || return 1
            done ;;
        hep_forward) (cd "$THESIS" && "$PY" analysis/hep_external_validation.py "${CV[@]}") ;;
        hep_eeg) (cd "$THESIS" && "$PY" -m analysis.hep_external_validation_eeg "${CV[@]}") ;;
        hep_reverse) (cd "$THESIS" && "$PY" analysis/hep_reverse_validation.py "${CV[@]}") ;;
        hep_focal) (cd "$THESIS" && "$PY" analysis/hep_focal_external_validation.py "${CV[@]}") ;;
        hep_reduced) (cd "$THESIS" && "$PY" -m analysis.hep_reduced_external_validation "${CV[@]}") ;;
        hep_forward_h12) (cd "$THESIS" && "$PY" analysis/hep_external_validation.py "${CV[@]}" "${H12[@]}") ;;
        hep_eeg_h12) (cd "$THESIS" && "$PY" -m analysis.hep_external_validation_eeg "${CV[@]}" "${H12[@]}") ;;
        hep_reverse_h12) (cd "$THESIS" && "$PY" analysis/hep_reverse_validation.py "${CV[@]}" "${H12[@]}") ;;
        reve) (cd "$THESIS" && "$PY" analysis/reve_standalone.py "${CV[@]}" \
                   --log-predictions "$REPO_DIR/$OUT/exp9_predictions") ;;
        exp19_tabular) exp19 --configs "${EXP19_TABULAR_CONFIGS[@]}" --seeds "$seed" ;;
        exp19_*) exp19 --configs "${EXP19_TEXT_CONFIGS[@]}" --encoders "${task#exp19_}" --seeds "$seed" ;;
        exp18_noRMH) exp18 --config Exp4a Exp5a Exp5b --exclude-rmh --seeds "$seed" ;;
        exp18_Exp5c|exp18_Exp6b|exp18_Exp7a|exp18_h12_Exp5c|exp18_h12_Exp6b|exp18_h12_Exp7a)
            if [[ "$EXP18_EEG_SEEDS" != *" $seed "* ]]; then
                echo "   (exp18 EEG configurations use seeds${EXP18_EEG_SEEDS}only; nothing to do)"; return 0
            fi
            if [[ "$task" == exp18_h12_* ]]; then exp18 --config "${task#exp18_h12_}" --seeds "$seed" "${H12[@]}"
            else exp18 --config "${task#exp18_}" --seeds "$seed"; fi ;;
        exp18_h12_*) exp18 --config "${task#exp18_h12_}" --seeds "$seed" "${H12[@]}" ;;
        exp18_*) exp18 --config "${task#exp18_}" --seeds "$seed" ;;
        *) echo "unknown task: $task (see: bash rerun_clean.sh list)" >&2; return 2 ;;
    esac
}

verify () {
    local rc=0
    echo "== verify_oof (all prediction files + expected $PROTOCOL files) =="
    local expect
    expect="$(mktemp)"
    expected_files > "$expect"
    "$PY" -m shared.verify_oof "$OUT" --expect "$expect" || rc=1
    rm -f "$expect"
    echo ""
    echo "== task completion =="
    local n_done=0 n_deferred=0
    for item in $(items); do
        if [[ -f "$DONE/${item%%:*}_s${item##*:}.done" ]] && exp18_ready "${item%%:*}" "${item##*:}"; then
            n_done=$((n_done + 1))
        elif deferred "$item"; then n_deferred=$((n_deferred + 1))
        else echo "MISSING $item"; rc=1; fi
    done
    echo "$n_done/$(items | wc -l) work items done, $n_deferred deferred (RUN_DEFERRED=1 to include)"
    local locks
    locks=$(ls -d "$DONE"/*.lock 2>/dev/null)
    if [[ -n "$locks" ]]; then
        echo "items locked (running now, or left by a killed job; remove the .lock directory if nothing runs it):"
        echo "$locks" | sed 's/^/  /'
    fi
    echo ""
    echo "== clean HEP outputs (file per seed, one summary row per configuration) =="
    local f want csv rows
    local specs=(hep_external_summary:3: hep_reverse_summary:3: hep_focal_external_summary:2:)
    if [[ "$PROTOCOL" == refit || "${RUN_DEFERRED:-0}" == 1 ]]; then
        specs+=(hep_external_summary_eeg:3: hep_reduced_external_summary:1:)
    fi
    [[ "$PROTOCOL" == refit ]] && specs+=(hep_external_summary:3:_h12 hep_reverse_summary:3:_h12)
    [[ "$PROTOCOL" == refit && "${RUN_DEFERRED:-0}" == 1 ]] && specs+=(hep_external_summary_eeg:3:_h12)
    local tag
    for spec in "${specs[@]}"; do
        IFS=: read -r f want tag <<< "$spec"
        for seed in "${SEEDS[@]}"; do
            csv="$THESIS/analysis/output/${f}_sp-multilabel_${SEL}_s${seed}${tag}.csv"
            if [[ ! -f "$csv" ]]; then echo "MISSING ${f}${tag} s${seed}"; rc=1; continue; fi
            rows=$(( $(wc -l < "$csv") - 1 ))
            (( rows == want )) || { echo "INCOMPLETE ${f}${tag} s${seed}: $rows of $want rows"; rc=1; }
        done
    done
    echo ""
    echo "== exp18 analysis =="
    "$PY" -m exp18_mixed_cohort.analyse || rc=1
    echo ""
    if (( rc == 0 )); then echo "GATE: PASS"; else echo "GATE: FAIL"; fi
    return $rc
}

preflight () {
    local rc=0
    check () { if eval "$2" >/dev/null 2>&1; then echo "ok      $1"; else echo "MISSING $1"; rc=1; fi; }
    check "python env (.venv-others)" "[[ -x '$PY' ]]"
    check "packages (torch, iterstrat, sklearn, scipy, pandas)" \
        "'$PY' -c 'import torch, iterstrat, sklearn, scipy, pandas'"
    check "braindecode (exp9 EEGNet/LaBraM encoders)" "'$PY' -c 'import braindecode'"
    check "CUDA visible (expected on a GPU node only)" "'$PY' -c 'import torch; assert torch.cuda.is_available()'"
    check "asm_data (clinical CSVs)" "'$PY' -c 'from shared.hep_cohort import ALFRED_CSV, HEP_CSV; assert ALFRED_CSV.exists() and HEP_CSV.exists()'"
    check "EEG cache v2, Melbourne (exp2-7, 9, 11, 16, 17, 18, HEP; python -m shared.eeg_cache build --cohort alfred)" \
        "'$PY' -m shared.eeg_cache stats $OUT/eeg_cache/eeg19_v2_alfred.pkl > /dev/null"
    check "EEG cache v2, HEP1 (HEP EEG, exp18; python -m shared.eeg_cache build --cohort hep)" \
        "'$PY' -m shared.eeg_cache stats $OUT/eeg_cache/eeg19_v2_hep.pkl > /dev/null"
    check "text + SMILES embeddings" "[[ -f $OUT/bert_alfred_1stregimen_eeg_embeddings.npy && -f $OUT/hep_clinicalbert_eeg_embeddings.npy && -f $OUT/chemberta_asm_embeddings.npy ]]"
    check "REVE features v2 (exp9 encoder_reve_frozen, exp15; thesisStandalone/analysis/reve_extract_features.py)" \
        "'$PY' -m shared.eeg_features check --feature-set reve_v2 --cohort alfred > /dev/null"
    check "LaBraM features v2 (exp9 encoder_labram_pretrained_frozen, exp15; python -m shared.labram_pretrained extract)" \
        "'$PY' -m shared.eeg_features check --feature-set labram_v2 --cohort alfred > /dev/null"
    check "legacy exp9 EEG2Vec OOF file (reve's 147-patient cohort)" \
        "[[ -f $OUT/exp9_predictions/predictions_oof_exp9_encoder_eeg2vec.json ]]"
    check "logs/ directory (slurm opens its log files before the job starts)" "mkdir -p logs"
    if [[ -f "$EXP18_DUPLICATES" ]]; then
        echo "note    exp18 will exclude $(grep -c . "$EXP18_DUPLICATES") confirmed duplicate HEP1 pid(s)"
    else
        echo "note    no confirmed-duplicates file: exp18 runs on the unmodified pooled cohort"
    fi
    check "thesisStandalone clone" "[[ -f '$THESIS/analysis/hep_external_validation.py' ]]"
    check "expected-files manifest" "[[ -f clean_rerun_expected.txt ]]"
    check "exp19 embedding stores (copied from the laptop)" \
        "'$PY' -c 'from exp19_serialised_clinical.texts import all_texts, load_frames; from exp19_serialised_clinical.run_experiments import lookup; [lookup(e, p, all_texts(load_frames())) for e, p in [(\"pubmedbert\",\"mean\"),(\"clinicalbert\",\"mean\"),(\"llama31_8b\",\"mean\"),(\"llama31_8b\",\"last\"),(\"qwen3_embed_8b\",\"last\")]]'"
    echo "versions: bash ${BASH_VERSION}; $("$PY" -c 'import sklearn, pandas, torch; print(f"sklearn {sklearn.__version__}, pandas {pandas.__version__}, torch {torch.__version__}")' 2>/dev/null)"
    echo "experiments $(git rev-parse --short HEAD)  thesisStandalone $(git -C "$THESIS" rev-parse --short HEAD 2>/dev/null)"
    echo "(compare both commits with the laptop before submitting)"
    return $rc
}

# Items that run comfortably without a GPU (tabular, embedding-only or
# precomputed-feature models). rerun_clean_cpu.slurm runs them on the CPU
# partition, outside the per-user GPU cap; the GPU arrays skip them once done.
# A static filter of `list`, so array indices never move.
CPU_TASK_RE='^(exp4|exp15|exp4_decomp|hep_forward|hep_forward_h12|hep_reverse|hep_reverse_h12|hep_focal|reve|exp18_(h12_)?(Exp4a|Exp5a|Exp5b|noRMH|S19[AD]_[a-z0-9_]+|LF_T5a-full|LF_T6a)|exp19_[a-z0-9_]+):'

# Deferred (2026-09-30): items nothing in the paper reports, skipped to fit the
# per-user GPU cap. A deferred item exits at once without a done marker and
# `verify` counts it as deferred, not missing; RUN_DEFERRED=1 runs it.
#   both protocols: exp9 ablations other than the four encoders compared in
#                   the paper, and exp11 (EEG2Vec-128 variants, an appendix aside)
#   refit:          the harmonised-HEP1-label items for EEG configurations
#   innersplit:     everything except the headline configurations (exp1-6,
#                   exp7a) and the cheap CPU items (HEP forward/reverse/focal,
#                   REVE, exp15, exp18 clinical/text, exp19)
DEFER_BOTH='exp9_(encoder_frozen|aggregator_[a-z0-9_]+|embed_dim_[0-9]+)|exp11_[a-z0-9_]+'
case "$PROTOCOL" in
    refit) DEFER_RE="^(${DEFER_BOTH}|hep_eeg_h12|exp18_h12_(Exp5c|Exp6b|Exp7a)):" ;;
    innersplit) DEFER_RE="^(${DEFER_BOTH}|exp9_[a-z0-9_]+|exp7b|exp7a_stratbatch|exp16|exp17|hep_eeg|hep_reduced|exp18_(Exp5c|Exp6b|Exp7a)):" ;;
esac
deferred () { [[ "${RUN_DEFERRED:-0}" != 1 ]] && grep -qE "$DEFER_RE" <<< "$1"; }

# The clean_rerun_expected.txt globs for the active protocol: the file is
# written for the inner split; the refit protocol swaps the suffix and adds
# its decomposition cells.
expected_files () {
    expected_files_all | filter_deferred_files
}

# Drop the expected files of deferred items (unless RUN_DEFERRED=1).
filter_deferred_files () {
    if [[ "${RUN_DEFERRED:-0}" == 1 ]]; then cat; return; fi
    local drop='^exp11_predictions/|^exp9_predictions/predictions_oof_exp9_(encoder_frozen|aggregator_|embed_dim_)'
    [[ "$PROTOCOL" == innersplit ]] && drop="$drop"'|^exp9_predictions/predictions_oof_exp9_(baseline|encoder_(eegnet|labram|eeg2vec|reve))|^exp7_predictions/predictions_oof_(7b|asmstratbatch)|^exp16_predictions/|^exp17_predictions/'
    grep -vE "$drop"
}

expected_files_all () {
    local s a b
    # Every exp9 ablation and every exp11 base (one work item each).
    for s in "${SEEDS[@]}"; do
        for a in "${EXP9_ABLATIONS[@]}"; do
            echo "exp9_predictions/predictions_oof_exp9_${a}_sp-multilabel_${SEL}_s${s}.json"
        done
        for b in 3a 6b 7a; do
            echo "exp11_predictions/predictions_oof_exp11_exp11_${b}_*_sp-multilabel_${SEL}_s${s}.json"
        done
    done
    if [[ "$PROTOCOL" == refit ]]; then
        grep -v '^exp4_decomposition/' clean_rerun_expected.txt | sed 's/_sp-multilabel_iv20_s/_sp-multilabel_rf5_s/'
        for s in "${SEEDS[@]}"; do
            for cell in "_s$s" "_sp-legacy_iv20_s$s" "_sp-multilabel_iv0_s$s" "_sp-multilabel_iv20_s$s" \
                        "_sp-legacy_rf5_s$s" "_sp-multilabel_rf5_s$s"; do
                echo "exp4_decomposition/predictions_oof_exp4a_mlp${cell}.json"
            done
        done
    else
        grep -v '^exp4_decomposition/' clean_rerun_expected.txt
    fi
}

# One-off, before the PROTOCOL=innersplit rerun: move every inner-split output
# written before the 2026-09-28 outcome-label correction (and its done markers)
# into outputs/_archive_prepolarity_20260928/, keeping the directory layout.
# Only files last modified before that date move, so nothing a post-correction
# run wrote (the refit decomposition writes inner-split-named cells too) can.
archive_iv20 () {
    local dest="$OUT/_archive_prepolarity_20260928" n=0 f
    mkdir -p "$dest"
    while IFS= read -r -d '' f; do
        mkdir -p "$dest/$(dirname "${f#$OUT/}")"
        mv -n "$f" "$dest/${f#$OUT/}" && n=$((n + 1))
    # Inner-split and decomposition files by suffix; exp18 and exp19's HEP1
    # ensembles by the absence of a refit (_rf<k>) tag.
    done < <(find -L "$OUT"/exp*_predictions "$OUT"/exp4_decomposition "$OUT"/exp18_mixed_cohort \
                 -maxdepth 1 -type f -not -newermt 2026-09-28 \
                 \( -name '*_sp-multilabel_iv20_*' -o -name '*_sp-legacy_iv20_*' -o -name '*_sp-multilabel_iv0_*' \
                    -o -name 'predictions_oof_exp4a_mlp_s4?.json' -o -name 'results_*_s4?.json' \
                    -o \( -regex '.*/\(predictions\|folds\|run\)_[A-Za-z0-9_-]*_seed4[2-6]\.\(csv\|json\)' \
                          -not -name '*_rf[0-9]*' \) \
                    -o \( -name 'hep_external_exp19_*_s4?.csv' -not -name '*_rf[0-9]*' \) \) -print0)
    while IFS= read -r -d '' f; do
        mkdir -p "$dest/thesis_output"
        mv -n "$f" "$dest/thesis_output/" && n=$((n + 1))
    done < <(find "$THESIS/analysis/output" -maxdepth 1 -type f -not -newermt 2026-09-28 \
                 -name 'hep_*_sp-multilabel_iv20_s*' -print0)
    [[ -d "$OUT/_clean_rerun" ]] && mv -n "$OUT/_clean_rerun" "$dest/_clean_rerun" && n=$((n + 1))
    echo "moved $n file(s)/dir(s) to $dest"
}

items () {
    local t s
    for t in "${TASKS[@]}"; do
        for s in "${SEEDS[@]}"; do
            # exp18 EEG configurations run seeds 42-44 only; do not queue no-op GPU jobs.
            case "$t" in exp18_Exp5c|exp18_Exp6b|exp18_Exp7a|exp18_h12_Exp5c|exp18_h12_Exp6b|exp18_h12_Exp7a)
                [[ "$EXP18_EEG_SEEDS" == *" $s "* ]] || continue ;;
            esac
            echo "$t:$s"
        done
    done
}

case "${1:-}" in
    list) items ;;
    list-cpu) items | grep -E "$CPU_TASK_RE" ;;
    preflight) preflight ;;
    verify) verify ;;
    archive-iv20) archive_iv20 ;;
    smoke)
        # An end-to-end dry run of one task on this host's data: first outer fold,
        # two inner folds, two epochs (shared.cv_splits --smoke; exp18's own --smoke),
        # seed 42, no done marker. Tasks that take an output directory write under
        # $SMOKE_OUT; the others write beside their real outputs with every file
        # suffixed _smoke, so no dry run can be taken for a result.
        task="${2:-}"; [[ -n "$task" ]] || { echo "usage: smoke <task>" >&2; exit 2; }
        SMOKE_OUT="${SMOKE_OUT:-/tmp/asm_smoke_$$}"; mkdir -p "$SMOKE_OUT"
        OUT="$SMOKE_OUT"
        CV_SEL+=(--smoke)
        EXP18_SEL+=(--smoke --out-dir "$SMOKE_OUT/exp18_mixed_cohort")
        echo "== smoke $task (outputs under $SMOKE_OUT, files suffixed _smoke) =="
        run_task "$task" 42 ;;
    "") echo "usage: [PROTOCOL=refit|innersplit] bash rerun_clean.sh {preflight|list|verify|archive-iv20|smoke <task>|<task>:<seed>}" >&2; exit 2 ;;
    *:*)
        task="${1%%:*}"; seed="${1##*:}"
        marker="$DONE/${task}_s${seed}.done"
        mkdir -p "$DONE"
        if [[ -f "$marker" && "${FORCE:-0}" != 1 ]] && exp18_ready "$task" "$seed"; then
            echo "== $task seed $seed already done ($marker); FORCE=1 to rerun =="; exit 0
        fi
        if deferred "$task:$seed"; then
            echo "== $task seed $seed deferred for $PROTOCOL (RUN_DEFERRED=1 to run it) =="; exit 0
        fi
        # One runner per item: the CPU and GPU arrays can both reach it. mkdir is
        # atomic on Lustre; the lock goes when this shell exits or is cancelled.
        lock="$DONE/${task}_s${seed}.lock"
        if ! mkdir "$lock" 2>/dev/null; then
            # A cancelled job can be killed before its trap removes the lock;
            # take over a lock whose slurm job no longer exists.
            owner_job=$(awk '{print $3}' "$lock/owner" 2>/dev/null)
            if [[ -n "$owner_job" && "$owner_job" != none ]] && command -v squeue >/dev/null \
                    && [[ -z "$(squeue -h -j "$owner_job" 2>/dev/null)" ]]; then
                echo "== stale lock from finished job $owner_job; taking it over =="
                rm -rf "$lock"
            fi
            if ! mkdir "$lock" 2>/dev/null; then
                echo "== $task seed $seed is running elsewhere ($(cat "$lock/owner" 2>/dev/null)); skipping =="
                exit 0
            fi
        fi
        echo "$(hostname) job ${SLURM_JOB_ID:-none} $(date -Is)" > "$lock/owner"
        trap 'rm -rf "$lock"' EXIT
        trap 'exit 143' TERM INT
        if [[ -f "$marker" && "${FORCE:-0}" != 1 ]] && exp18_ready "$task" "$seed"; then
            echo "== $task seed $seed finished while waiting ($marker) =="; exit 0
        fi
        echo "== $task seed $seed, protocol $PROTOCOL  (host $(hostname), $(date -Is)) =="
        start=$(date +%s)
        if run_task "$task" "$seed"; then
            echo "$(date -Is) $(( $(date +%s) - start ))s $(git rev-parse --short HEAD)" > "$marker"
            echo "== $task seed $seed done in $(( $(date +%s) - start ))s =="
        else
            echo "== $task seed $seed FAILED ==" >&2; exit 1
        fi ;;
    *) echo "expected <task>:<seed>, e.g. exp4:42 (see: bash rerun_clean.sh list)" >&2; exit 2 ;;
esac
