#!/bin/bash
#==============================================================================
# M3 HPC Job Submission Script for ASM Experiments
#==============================================================================
# Usage:
#   sbatch submit_job.sh -e exp2                    # Run experiment 2
#   sbatch submit_job.sh -e exp1 -a "--dry-run"    # Exp 1 dry run
#   bash submit_job.sh -e exp2 -d                   # Local dry run test
#
# Options:
#   -e, --experiment EXP   Experiment to run (e.g. exp1, exp2, exp9)
#   -m, --module MOD       Override entry point module (default: run_experiments)
#   -a, --args ARGS        Extra arguments for the experiment script
#   -d, --dry-run          Test mode - show config without executing
#   -h, --help             Show help message
#
# Override SBATCH settings via command line:
#   sbatch --time=8:00:00 submit_job.sh -e exp2
#   sbatch --gres=gpu:A40:1 submit_job.sh -e exp2
#==============================================================================

#==============================================================================
# SBATCH DIRECTIVES - Edit these as needed - Keep first hash
#==============================================================================
#SBATCH --job-name=asm_exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/asm_%j.out
#SBATCH --error=logs/asm_%j.err

# Uncomment for specific GPU type:
#SBATCH --gres=gpu:A100:1
##SBATCH --gres=gpu:A40:1

# Uncomment for email notifications:
##SBATCH --mail-user=your.email@monash.edu
##SBATCH --mail-type=BEGIN,END,FAIL

#==============================================================================
# CLI ARGUMENT PARSING
#==============================================================================

# Default values (can also be set via environment variables)
EXPERIMENT="${EXPERIMENT:-exp1}"
MODULE="${MODULE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-false}"

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --experiment EXP   Experiment to run (e.g. exp1, exp2, exp9)"
    echo "  -m, --module MOD       Override entry point module (default: run_experiments)"
    echo "  -a, --args ARGS        Extra arguments for the experiment script"
    echo "  -d, --dry-run          Test mode - show config without executing"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Available experiments:"
    # List experiment directories dynamically
    SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
    for dir in "${SCRIPT_DIR}"/exp*_*/; do
        dirname=$(basename "$dir")
        prefix="${dirname%%_*}"
        echo "  ${prefix}  -> ${dirname}/"
    done
    echo ""
    echo "Examples:"
    echo "  sbatch $0 -e exp2"
    echo "  sbatch $0 -e exp2 -a '--eeg-encoder simplecnn'"
    echo "  sbatch $0 -e exp9 -a '--experiment encoder_labram_scratch'"
    echo "  sbatch --mem=64G $0 -e exp9 -a '--experiment encoder_labram_scratch'"
    echo "  sbatch $0 -e exp9 -a '--quick'"
    echo "  bash $0 -e exp1 -d"
    echo "  EXPERIMENT=exp2 sbatch $0"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        -m|--module)
            MODULE="$2"
            shift 2
            ;;
        -a|--args)
            EXTRA_ARGS="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

#==============================================================================
# EXPERIMENT DISCOVERY
#==============================================================================

# Get the directory where this script is located
# Use SLURM_SUBMIT_DIR when running via sbatch (SLURM copies scripts to a spool directory)
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Find the experiment directory matching the prefix
EXP_DIR=""
EXP_DIR_COUNT=0
for dir in "${SCRIPT_DIR}"/${EXPERIMENT}_*/; do
    if [ -d "$dir" ]; then
        EXP_DIR="$dir"
        EXP_DIR_COUNT=$((EXP_DIR_COUNT + 1))
    fi
done

if [ $EXP_DIR_COUNT -eq 0 ]; then
    echo "ERROR: No directory found matching '${EXPERIMENT}_*'."
    echo "Available experiments:"
    for dir in "${SCRIPT_DIR}"/exp*_*/; do
        dirname=$(basename "$dir")
        prefix="${dirname%%_*}"
        echo "  ${prefix}  -> ${dirname}/"
    done
    exit 1
fi

if [ $EXP_DIR_COUNT -gt 1 ]; then
    # Disambiguate: prefer the directory containing run_experiments.py
    CANDIDATE=""
    CANDIDATE_COUNT=0
    for dir in "${SCRIPT_DIR}"/${EXPERIMENT}_*/; do
        if [ -f "${dir}/run_experiments.py" ]; then
            CANDIDATE="$dir"
            CANDIDATE_COUNT=$((CANDIDATE_COUNT + 1))
        fi
    done

    if [ $CANDIDATE_COUNT -eq 1 ]; then
        EXP_DIR="$CANDIDATE"
    else
        echo "ERROR: Multiple directories match '${EXPERIMENT}_*':"
        for dir in "${SCRIPT_DIR}"/${EXPERIMENT}_*/; do
            echo "  $(basename "$dir")"
        done
        echo "Use the full directory name or -m to specify the module."
        exit 1
    fi
fi

EXP_DIRNAME=$(basename "$EXP_DIR")

# Determine entry point module
if [ -n "$MODULE" ]; then
    # User-specified module override
    ENTRY_MODULE="${MODULE}"
    if [ ! -f "${EXP_DIR}/${ENTRY_MODULE}.py" ]; then
        echo "ERROR: Module '${ENTRY_MODULE}' not found in ${EXP_DIRNAME}/"
        echo "Available modules:"
        for f in "${EXP_DIR}"/*.py; do
            fname=$(basename "$f" .py)
            if [ "$fname" != "__init__" ] && [ "$fname" != "__pycache__" ]; then
                echo "  ${fname}"
            fi
        done
        exit 1
    fi
else
    # Default: look for run_experiments.py, fall back to __main__.py
    if [ -f "${EXP_DIR}/run_experiments.py" ]; then
        ENTRY_MODULE="run_experiments"
    elif [ -f "${EXP_DIR}/__main__.py" ]; then
        ENTRY_MODULE="__main__"
    else
        echo "ERROR: No run_experiments.py or __main__.py found in ${EXP_DIRNAME}/."
        echo "Use -m to specify the entry point module."
        echo "Available modules:"
        for f in "${EXP_DIR}"/*.py; do
            fname=$(basename "$f" .py)
            if [ "$fname" != "__init__" ] && [ "$fname" != "__pycache__" ]; then
                echo "  ${fname}"
            fi
        done
        exit 1
    fi
fi

# Use python -m package for __main__.py, python -m package.module otherwise
if [ "$ENTRY_MODULE" = "__main__" ]; then
    CMD="python -m ${EXP_DIRNAME} ${EXTRA_ARGS}"
else
    CMD="python -m ${EXP_DIRNAME}.${ENTRY_MODULE} ${EXTRA_ARGS}"
fi

#==============================================================================
# JOB EXECUTION
#==============================================================================

echo "============================================================"
echo "ASM Experiment Job"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        $(hostname)"
echo "Start time:  $(date)"
echo "Directory:   ${SCRIPT_DIR}"
echo "Experiment:  ${EXPERIMENT} (${EXP_DIRNAME})"
echo "Module:      ${ENTRY_MODULE}"
echo "Extra args:  ${EXTRA_ARGS:-none}"
echo "============================================================"

# Create logs directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/logs"

# Change to project directory
cd "${SCRIPT_DIR}"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv-others/bin/activate

# Verify GPU access
echo ""
echo "GPU Status:"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"

# Check nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "nvidia-smi:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
fi

# Run environment validation
echo ""
echo "============================================================"
echo "Environment Validation"
echo "============================================================"
if [ "$EXPERIMENT" = "exp1" ]; then
    python check_environment.py --exp1
    ENV_CHECK=$?
elif [ "$EXPERIMENT" = "exp2" ]; then
    python check_environment.py --exp2
    ENV_CHECK=$?
else
    python check_environment.py --all
    ENV_CHECK=$?
fi

if [ $ENV_CHECK -ne 0 ]; then
    echo ""
    echo "WARNING: Environment check reported issues (exit code $ENV_CHECK)"
    echo "The experiment may fail. Check the output above for details."
    echo ""
fi

# Run the experiment (or show what would run in dry-run mode)
echo ""
echo "============================================================"
if [ "$DRY_RUN" = "true" ]; then
    echo "DRY RUN MODE - Not executing, just showing configuration"
    echo "============================================================"
    echo ""
    echo "Would execute: ${CMD}"
    echo ""
    echo "Environment check complete. To run for real:"
    echo "  sbatch submit_job.sh -e ${EXPERIMENT}"
    EXIT_CODE=0
else
    echo "Starting ${EXPERIMENT}..."
    echo "============================================================"
    echo ""
    echo "Command: ${CMD}"
    echo ""
    ${CMD}
    EXIT_CODE=$?
fi

echo ""
echo "============================================================"
echo "Job completed"
echo "Exit code:   ${EXIT_CODE}"
echo "End time:    $(date)"
echo "============================================================"

exit ${EXIT_CODE}
