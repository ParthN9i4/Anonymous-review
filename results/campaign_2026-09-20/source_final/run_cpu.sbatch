#!/bin/bash
#SBATCH --job-name=pvit_ckks
#SBATCH --partition=debug
#SBATCH --account=hpc
#SBATCH --qos=normal
#SBATCH --nice=100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
PVIT_ROOT="$1"
PVIT_KIND="$2"
cd "$PVIT_ROOT/code"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
PVIT_PYTHON="$(cat "$PVIT_ROOT/fhe_python.txt")"
"$PVIT_PYTHON" -c 'from io_utils import verify; import sys; verify(sys.argv[1])' "$PVIT_ROOT"
if [[ "$PVIT_KIND" == smoke ]]; then
 "$PVIT_PYTHON" test_fhe.py
else
 "$PVIT_PYTHON" fhe.py "$PVIT_ROOT" "$SLURM_ARRAY_TASK_ID"
fi
