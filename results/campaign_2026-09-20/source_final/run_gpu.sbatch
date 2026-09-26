#!/bin/bash
#SBATCH --job-name=pvit_assess
#SBATCH --partition=debug
#SBATCH --account=hpc
#SBATCH --qos=normal
#SBATCH --nice=100
#SBATCH --gres=gpu:nvidia_h200_nvl_2g.35gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail
module load anaconda-2025.12-2
PVIT_ROOT="$1"
PVIT_KIND="$2"
cd "$PVIT_ROOT/code"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
if [[ "$PVIT_KIND" == smoke ]]; then
 python -m unittest test_core test_torch -v
 python smoke.py "$PVIT_ROOT"
else
 python -c 'from io_utils import verify; import sys; verify(sys.argv[1])' "$PVIT_ROOT"
 python worker.py "$PVIT_ROOT" "$PVIT_KIND" "$SLURM_ARRAY_TASK_ID"
fi
