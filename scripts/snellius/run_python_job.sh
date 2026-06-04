#!/bin/bash
# Generic Snellius runner for one Python pipeline script.
# Submit with:
#   sbatch --export=PROJECT_DIR=/home/user/thesis,DATA_DIR=/scratch-shared/user/datasets,CONDA_ENV=mysimenv,SCRIPT_PATH=scripts/snellius/train_formula_b1.py,SCRIPT_ARGS="--total-timesteps 10000" scripts/snellius/run_python_job.sh

#SBATCH --partition=genoa
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=formula_job
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

if [[ -z "${PROJECT_DIR:-}" ]]; then
  echo "PROJECT_DIR is required" >&2
  exit 2
fi
if [[ -z "${DATA_DIR:-}" ]]; then
  echo "DATA_DIR is required" >&2
  exit 2
fi
if [[ -z "${CONDA_ENV:-}" ]]; then
  echo "CONDA_ENV is required" >&2
  exit 2
fi
if [[ -z "${SCRIPT_PATH:-}" ]]; then
  echo "SCRIPT_PATH is required" >&2
  exit 2
fi

mkdir -p "$PROJECT_DIR/logs"

module load 2023
module load Miniconda3/23.5.2-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"

echo "[$(date)] Host      : $(hostname)"
echo "[$(date)] Job       : ${SLURM_JOB_ID:-local}"
echo "[$(date)] Array task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "[$(date)] Script    : $SCRIPT_PATH ${SCRIPT_ARGS:-}"
echo "[$(date)] DATA_DIR  : $DATA_DIR"

python "$SCRIPT_PATH" ${SCRIPT_ARGS:-}

echo "[$(date)] Finished $SCRIPT_PATH"

