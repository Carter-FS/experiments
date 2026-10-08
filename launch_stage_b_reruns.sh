#!/bin/bash
#==============================================================================
# Stage B: ASM-balanced training reruns.
#
# Submits exp7 in two variants (--asm-balance weighted and
# --asm-balance stratified_batch) via exp7's predictions mode, which is
# the only mode that produces per-ASM counterfactual probabilities for
# the best-ASM simulation comparison.
#
# Exp3a, Exp5a, Exp6a wiring is pending (see docs/STAGE_B_README.md). When
# wired, add them to the EXPERIMENTS array below.
#
# Usage:
#   bash launch_stage_b_reruns.sh           # submit all to SLURM
#   bash launch_stage_b_reruns.sh --dry-run # show commands, don't submit
#   bash launch_stage_b_reruns.sh --local   # run sequentially on this machine
#==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=false
LOCAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --local)   LOCAL=true;   shift ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Each row: <job_name>|<entry args>|<wall_time>
# Stage B: 4 headline configs (Exp3a, Exp5a, Exp6a, Exp7a) x 2 methods
# (weighted, stratified_batch) = 8 jobs total.
# Output predictions land in outputs/exp{N}_predictions/predictions_oof_*_asm{method}.json.
EXPERIMENTS=(
    "stageB_exp3a_weighted|-e exp3 -m run_experiments -a \"--fusion mlp --log-predictions --deterministic --asm-balance weighted\"|3:00:00"
    "stageB_exp3a_stratbatch|-e exp3 -m run_experiments -a \"--fusion mlp --log-predictions --deterministic --asm-balance stratified_batch\"|3:00:00"
    "stageB_exp5a_weighted|-e exp5 -m run_experiments -a \"--exp 5a --log-predictions --deterministic --asm-balance weighted\"|2:00:00"
    "stageB_exp5a_stratbatch|-e exp5 -m run_experiments -a \"--exp 5a --log-predictions --deterministic --asm-balance stratified_batch\"|2:00:00"
    "stageB_exp6a_weighted|-e exp6 -m run_experiments -a \"--exp 6a --log-predictions --deterministic --asm-balance weighted\"|3:00:00"
    "stageB_exp6a_stratbatch|-e exp6 -m run_experiments -a \"--exp 6a --log-predictions --deterministic --asm-balance stratified_batch\"|3:00:00"
    "stageB_exp7a_weighted|-e exp7 -m run_experiments -a \"--mode predictions --asm-balance weighted --deterministic\"|3:00:00"
    "stageB_exp7a_stratbatch|-e exp7 -m run_experiments -a \"--mode predictions --asm-balance stratified_batch --deterministic\"|3:00:00"
)

echo "============================================================"
echo "Stage B rerun launcher"
echo "Mode: $([ "$LOCAL" = "true" ] && echo "local" || echo "SLURM")"
echo "Dry run: $DRY_RUN"
echo "Experiments to launch: ${#EXPERIMENTS[@]}"
echo "============================================================"

JOB_IDS=()
for row in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r JOB_NAME ENTRY_ARGS WALL_TIME <<< "$row"

    if [ "$LOCAL" = "true" ]; then
        CMD="bash submit_job.sh $ENTRY_ARGS"
        echo ""
        echo ">> $JOB_NAME (local): $CMD"
        if [ "$DRY_RUN" = "false" ]; then
            eval "bash submit_job.sh $ENTRY_ARGS" 2>&1 | tee "logs/${JOB_NAME}_local.out"
        fi
    else
        SBATCH_FLAGS="--time=$WALL_TIME --job-name=$JOB_NAME"
        CMD="sbatch $SBATCH_FLAGS submit_job.sh $ENTRY_ARGS"
        echo ""
        echo ">> $JOB_NAME: $CMD"
        if [ "$DRY_RUN" = "false" ]; then
            if JOB_OUTPUT=$(eval "sbatch $SBATCH_FLAGS submit_job.sh $ENTRY_ARGS" 2>&1); then
                echo "  $JOB_OUTPUT"
                JOB_ID=$(echo "$JOB_OUTPUT" | grep -oE '[0-9]+')
                JOB_IDS+=("$JOB_ID:$JOB_NAME")
            else
                echo "  FAILED: $JOB_OUTPUT"
            fi
        fi
    fi
done

if [ "$LOCAL" = "false" ] && [ "$DRY_RUN" = "false" ] && [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo ""
    echo "============================================================"
    echo "Submitted ${#JOB_IDS[@]} jobs:"
    for entry in "${JOB_IDS[@]}"; do
        echo "  Job ${entry%%:*}  (${entry##*:})"
    done
    echo ""
    echo "Output predictions land in outputs/exp7_predictions/"
    echo "  predictions_oof_asmweighted.json     (--asm-balance weighted)"
    echo "  predictions_oof_asmstratbatch.json   (--asm-balance stratified_batch)"
    echo "Existing Stage A predictions_oof.json (baseline) is NOT overwritten."
    echo "============================================================"
fi
