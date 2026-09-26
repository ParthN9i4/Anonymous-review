#!/bin/bash
#SBATCH --job-name=strong_full
#SBATCH --output=slurm_logs/strong_full_%j.out
#SBATCH --error=slurm_logs/strong_full_%j.err
#SBATCH --gres=gpu:nvidia_h200_nvl_2g.35gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=debug
#SBATCH --account=hpc
#SBATCH --qos=normal

# FULL: the two results the paper actually needs.
#
#   ARM 1 -- a genuine 5-seed plaintext baseline, so "Fix 2 recovers the
#            teacher" stops being anchored to an undertrained 77% reference.
#
#   ARM 2 -- the same strong recipe applied to Config E (all three
#            substitutions). This answers the question the paper cannot
#            currently answer: does interaction collapse survive a properly
#            regularised training regime, or is it an artefact of an
#            under-regularised one? BOTH outcomes are publishable:
#              * collapse persists -> finding is far stronger, and immune to
#                the "your baseline was weak" critique;
#              * collapse disappears -> an important, honest scope result
#                about when the failure mode does and does not occur.
#
# Run the pilot first; only submit this once the pilot shows the strong
# recipe actually clears the old 77%.

set -euo pipefail
mkdir -p slurm_logs
module load anaconda-2025.12-2
# Repository layout (2026-09-23): puts core/ and the repo root on PYTHONPATH and cd's to the root.
source "${SLURM_SUBMIT_DIR:-$PWD}/infrastructure/env.sh"
# Unbuffered stdout. Required, not cosmetic: tools/recover_results_from_log.py
# reconstructs a killed job's results by parsing this log, so anything
# still sitting in a buffer when SLURM kills the job is lost outright.
# (SLURM_Job_Scheduling_H200.docx S4.)
export PYTHONUNBUFFERED=1


echo "=== pre-flight ==="
python experiments/active/smoke_test_strong_baseline.py

echo
echo "=== ARM 1: strong recipe, TEACHER, 5 seeds, 300 epochs ==="
python core/strong_baseline.py --seeds 42 43 44 45 46 --epochs 300 \
    --recipe strong --tag teacher

echo
echo "=== ARM 2: strong recipe, CONFIG E (all three substitutions), 5 seeds ==="
python core/strong_baseline.py --seeds 42 43 44 45 46 --epochs 300 \
    --recipe strong --poly-gelu --poly-softmax --poly-norm --tag configE
