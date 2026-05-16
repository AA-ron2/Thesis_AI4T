#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=as_baseline

set -euo pipefail

module load 2023
module load Miniconda3/23.5.2-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"

echo "[$(date)] Starting day index $SLURM_ARRAY_TASK_ID"
python run_one_day.py --manifest "$MANIFEST_PATH" --day-index "$SLURM_ARRAY_TASK_ID"
echo "[$(date)] Finished day index $SLURM_ARRAY_TASK_ID"
