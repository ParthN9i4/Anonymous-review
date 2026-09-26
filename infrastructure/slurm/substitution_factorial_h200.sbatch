#!/bin/bash
# Clean, validation-selected eight-arm factorial: 000, A-G.
# Submit one seed per task so a wall-time failure costs one seed only.
# Do not git pull while any task from this array is queued or running.
#
# Examples:
#   sbatch infrastructure/slurm/substitution_factorial_h200.sbatch
#   DATASET=cifar100 sbatch infrastructure/slurm/substitution_factorial_h200.sbatch
#   CONFIG_SET=000 RUN_TAG=000kd sbatch infrastructure/slurm/substitution_factorial_h200.sbatch
#
# Outputs are deliberately tagged so this run cannot overwrite the historical
# files from the incomplete A-I campaign.
#SBATCH --job-name=pvit8
#SBATCH --output=slurm_logs/pvit8_%A_%a.out
#SBATCH --error=slurm_logs/pvit8_%A_%a.err
#SBATCH --gres=gpu:nvidia_h200_nvl_2g.35gb:1
# The model's measured peak is ~1.5 GB GPU and it needs only two loader
# workers. Request the smallest practical host footprint as well.
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=09:00:00
#SBATCH --partition=debug
#SBATCH --account=hpc
#SBATCH --qos=normal
#SBATCH --nice=100
#SBATCH --array=42-46%1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs results/ablations checkpoints_substitution
module load anaconda-2025.12-2
# Repository layout (2026-09-23): puts core/ and the repo root on PYTHONPATH and cd's to the root.
source "${SLURM_SUBMIT_DIR:-$PWD}/infrastructure/env.sh"
export PYTHONUNBUFFERED=1

DS="${DATASET:-cifar10}"
TAG="${RUN_TAG:-clean8}"
SET="${CONFIG_SET:-primary}"
SEED="${SLURM_ARRAY_TASK_ID}"

echo "=== Private-ViT-FHE clean factorial ==="
echo "host=$(hostname) date=$(date -u +%FT%TZ) dataset=${DS} seed=${SEED} config_set=${SET} tag=${TAG}"
echo "git=$(git rev-parse HEAD)"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

ARGS=(--dataset "${DS}" --seeds "${SEED}" --config-set "${SET}"
      --run-tag "${TAG}" --skip-no-kd --save-checkpoints
      --num-workers 2 --gpu-memory-fraction 0.20)
python -u core/substitution_ablation.py "${ARGS[@]}"
