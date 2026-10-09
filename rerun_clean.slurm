#!/bin/bash
# M3 array wrapper for rerun_clean.sh: array index i runs the i-th work item of
# `PROTOCOL=<p> bash rerun_clean.sh list`. Tasks are independent and skip
# themselves once done, so a failed or timed-out index can simply be resubmitted.
# PROTOCOL (refit, the default, or innersplit) is passed through the environment.
#
# Submit from the experiments repo root on M3 (after pulling both repos and
# running `bash rerun_clean.sh preflight`, which also creates logs/):
#     PROTOCOL=refit bash rerun_clean.sh list | cat -n        # index <-> task (373 items)
#     sbatch --export=ALL,PROTOCOL=refit rerun_clean.slurm    # every refit item, 24 at a time
#     sbatch --export=ALL,PROTOCOL=innersplit --array=0-288%24 rerun_clean.slurm
#     sbatch --export=ALL,PROTOCOL=refit --array=0,5,14 rerun_clean.slurm   # a subset
#     sbatch --export=ALL,PROTOCOL=refit --array=$(PROTOCOL=refit bash rerun_clean.sh pending-array)%24 rerun_clean.slurm   # the items not yet done
# then, once an array has finished:
#     PROTOCOL=refit bash rerun_clean.sh verify
#SBATCH --job-name=clean_rerun
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-372%24
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/clean_rerun_%A_%a.out
#SBATCH --error=logs/clean_rerun_%A_%a.err

set -uo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_DIR"
mkdir -p logs

export PROTOCOL="${PROTOCOL:-refit}"
TASK="$(bash rerun_clean.sh list | sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p")"
if [[ -z "$TASK" ]]; then
    echo "no task for array index ${SLURM_ARRAY_TASK_ID:-0}" >&2
    exit 2
fi
echo "== array index ${SLURM_ARRAY_TASK_ID:-0} -> $TASK ($PROTOCOL) =="
# Match BLAS/OpenMP threads to the allocation (CPU-side work otherwise oversubscribes).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash rerun_clean.sh "$TASK"
