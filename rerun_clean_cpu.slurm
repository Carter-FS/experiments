#!/bin/bash
# CPU-partition array for the rerun items that need no GPU (see
# `bash rerun_clean.sh list-cpu`): tabular, embedding-only and
# precomputed-feature models, exp19 and exp18's clinical and text
# configurations. Runs outside the per-user GPU cap alongside the GPU arrays;
# every item takes a lock, so whichever array reaches an item first runs it
# and the other skips it (a finished item is skipped by its done marker).
#
#     PROTOCOL=refit bash rerun_clean.sh list-cpu | cat -n     # index <-> task (190 items)
#     sbatch --export=ALL,PROTOCOL=refit rerun_clean_cpu.slurm
#     sbatch --export=ALL,PROTOCOL=innersplit --array=0-119%24 rerun_clean_cpu.slurm
#     sbatch --export=ALL,PROTOCOL=refit --array=$(PROTOCOL=refit bash rerun_clean.sh pending-array-cpu)%24 rerun_clean_cpu.slurm   # the items not yet done
#SBATCH --job-name=clean_rerun_cpu
#SBATCH --partition=comp
#SBATCH --array=0-189%24
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/clean_rerun_cpu_%A_%a.out
#SBATCH --error=logs/clean_rerun_cpu_%A_%a.err

set -uo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_DIR"
mkdir -p logs

export PROTOCOL="${PROTOCOL:-refit}"
TASK="$(bash rerun_clean.sh list-cpu | sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p")"
if [[ -z "$TASK" ]]; then
    echo "no CPU task for array index ${SLURM_ARRAY_TASK_ID:-0}" >&2
    exit 2
fi
echo "== CPU array index ${SLURM_ARRAY_TASK_ID:-0} -> $TASK ($PROTOCOL) on $(hostname) =="
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_VISIBLE_DEVICES=""
bash rerun_clean.sh "$TASK"
