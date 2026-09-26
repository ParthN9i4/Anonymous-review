#!/bin/bash
#SBATCH --job-name=pvit_v2_ckks
#SBATCH --partition=debug
#SBATCH --account=hpc
#SBATCH --qos=normal
#SBATCH --nice=100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# Usage: sbatch [--array=...] run_cpu_v2.sbatch <run_root> <kind>
#   kind: smoke | f0 | f1_smoke | f1
# Runs in the TenSEAL environment recorded in <run_root>/fhe_python.txt (no PyTorch needed).
set -euo pipefail
PVIT_ROOT="$1"; PVIT_KIND="$2"
cd "$PVIT_ROOT/code"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
PY="$(cat "$PVIT_ROOT/fhe_python.txt")"
"$PY" -c 'from io_utils import verify; import sys; verify(sys.argv[1])' "$PVIT_ROOT"
case "$PVIT_KIND" in
  smoke)    "$PY" test_fhe.py && "$PY" -m unittest test_v2.FHETests -v ;;
  f0)       "$PY" f0_depth_ledger.py "$PVIT_ROOT" "$SLURM_ARRAY_TASK_ID" ;;
  f1_smoke) "$PY" f1_mlp_block.py smoke "$PVIT_ROOT" --threads 4 ;;
  f1)       "$PY" f1_mlp_block.py encrypt "$PVIT_ROOT" "$SLURM_ARRAY_TASK_ID" --threads 4 ;;
  *) echo "unknown kind $PVIT_KIND"; exit 2 ;;
esac
