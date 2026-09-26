#!/bin/bash
#SBATCH --job-name=pvit_v2_gpu
#SBATCH --partition=debug
#SBATCH --account=hpc
#SBATCH --qos=normal
#SBATCH --nice=100
#SBATCH --gres=gpu:nvidia_h200_nvl_2g.35gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# Usage: sbatch [--array=...] run_gpu_v2.sbatch <run_root> <kind>
#   kind: smoke | pilot | e1 | e2 | e3 | confirm | med_e1 | med_e2 | f1_capture | f1_inject
set -euo pipefail
module load anaconda-2025.12-2
PVIT_ROOT="$1"; PVIT_KIND="$2"
cd "$PVIT_ROOT/code"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c 'from io_utils import verify; import sys; verify(sys.argv[1])' "$PVIT_ROOT"
case "$PVIT_KIND" in
  smoke)
    python -m unittest test_core test_torch test_v2 -v
    python worker_v2.py --synthetic-selftest ;;
  pilot|e1|e2|e3|confirm|med_e1|med_e2)
    python worker_v2.py "$PVIT_ROOT" "$PVIT_KIND" "$SLURM_ARRAY_TASK_ID" ;;
  f1_capture) python f1_mlp_block.py capture "$PVIT_ROOT" ;;
  f1_inject)  python f1_mlp_block.py inject "$PVIT_ROOT" ;;
  *) echo "unknown kind $PVIT_KIND"; exit 2 ;;
esac
