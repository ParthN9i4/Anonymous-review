#!/bin/bash
#SBATCH --job-name=strong_pilot
#SBATCH --output=slurm_logs/strong_pilot_%j.out
#SBATCH --error=slurm_logs/strong_pilot_%j.err
#SBATCH --gres=gpu:nvidia_h200_nvl_2g.35gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=debug
#SBATCH --account=hpc
#SBATCH --qos=normal

# PILOT: answers one question fast -- can a proper training recipe lift the
# CIFAR-10 teacher off 77%? One seed, teacher only, plus the weak-recipe
# control at the same seed/epoch budget so the comparison is apples-to-apples
# rather than against the historical 100-epoch number.
#
# Diagnosis this tests (from archive/logs/log_subs.txt): the old teacher's train loss
# falls to ~0.01 while val gains ~1pt over the last 40 epochs -- it is
# overfitting, not undertrained. If that reading is right, the strong arm
# should beat the weak arm by a wide margin at EQUAL epochs.

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
echo "=== ARM 1: strong recipe, teacher, seed 42, 300 epochs ==="
python core/strong_baseline.py --seeds 42 --epochs 300 --recipe strong --tag teacher

echo
echo "=== ARM 2: weak recipe control, same seed, same 300 epochs ==="
echo "    (isolates recipe from epoch budget -- if weak also reaches ~90%,"
echo "     the problem was epochs after all and the diagnosis was wrong)"
python core/strong_baseline.py --seeds 42 --epochs 300 --recipe weak --tag teacher
