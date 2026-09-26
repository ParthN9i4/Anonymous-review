#!/bin/bash
#SBATCH --job-name=strong_blood
#SBATCH --output=slurm_logs/strong_blood_%j.out
#SBATCH --error=slurm_logs/strong_blood_%j.err
#SBATCH --gres=gpu:nvidia_h200_nvl_2g.35gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --partition=debug
#SBATCH --account=hpc
#SBATCH --qos=normal

# REGIME 3 on the dataset that anchors the paper (see docs/design/BASELINE_AND_DATASETS.md).
#
# BloodMNIST's teacher is already healthy (94.55 +/- 0.27, against the ~95%
# ResNet-18 reference MedMNIST reports), so this is NOT about rescuing a weak
# baseline. It asks the decisive question:
#
#   Does interaction collapse survive proper regularisation?
#
# BloodMNIST is the right place to ask, because its collapse signature is the
# strongest we have: Config E per-seed 73.31/82.81/18.24/90.18/92.81, with one
# seed pinned near the 12.5% chance floor, and a 146x variance ratio against
# Fix 2 (CIFAR-10's is 34x). If strong regularisation removes THAT, the
# metastability claim (N3) needs rescoping. If it does not, N3 gets much
# harder to attack.

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
echo "=== ARM 1: strong recipe, TEACHER, BloodMNIST, 5 seeds ==="
python core/strong_baseline.py --dataset bloodmnist --seeds 42 43 44 45 46 \
    --epochs 200 --recipe strong --tag teacher

echo
echo "=== ARM 2: strong recipe, CONFIG E (all three), BloodMNIST, 5 seeds ==="
echo "    Regime-1 comparator: 71.47 +/- 30.70, per-seed 73.31/82.81/18.24/90.18/92.81"
python core/strong_baseline.py --dataset bloodmnist --seeds 42 43 44 45 46 \
    --epochs 200 --recipe strong --poly-gelu --poly-softmax --poly-norm --tag configE
