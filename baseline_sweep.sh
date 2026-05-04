#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=as_baseline
#SBATCH --array=0-28%6
#SBATCH --output=/home/hmalash/thesis/logs/baseline_%A_%a.out
#SBATCH --error=/home/hmalash/thesis/logs/baseline_%A_%a.err

module load 2023
module load Miniconda3/23.5.2-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mysimenv

export DATA_DIR=/scratch-shared/hmalash/datasets/

cd /home/hmalash/thesis

echo "[$(date)] Starting day index $SLURM_ARRAY_TASK_ID"

python run_one_day.py --day-index $SLURM_ARRAY_TASK_ID

echo "[$(date)] Finished day index $SLURM_ARRAY_TASK_ID"
