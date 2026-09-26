#!/bin/bash
#SBATCH --job-name=lut_acc
#SBATCH --output=slurm_logs/lut_acc_%j.out
#SBATCH --error=slurm_logs/lut_acc_%j.err
#SBATCH --gres=gpu:nvidia_h200_nvl_2g.35gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=debug
#SBATCH --account=hpc
#SBATCH --qos=normal

# Does the model TOLERATE finite-LUT precision on ReLU and 1/rowsum?
# The numeric half is already measured (results/ablations/lut_precision_numeric.json):
# an 8-bit uniform table gives 2.9% max relative error on 1/x; log-spacing
# improves that 5.3x. This job answers whether that error costs accuracy.
#
# Gates whether CKKS functional bootstrapping (Alexandru/Kim/Polyakov,
# CRYPTO 2025) can absorb 26-27 of Fix 2's 36-37 levels per block.

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


# --range-mode measured is REQUIRED, not optional. Job 473 ran without it and
# inherited recip_range=(2.95, 46.01), a BloodMNIST-measured domain: 74% of this
# model's row-sums fell outside, accuracy sat at chance from 4 bits to 16, and
# the bit axis showed nothing because more bits subdivides a LUT's domain but
# never widens it. Output is tagged by range mode so the old artifact survives
# as the negative control.
python experiments/exploratory/lut_precision_sweep.py --mode accuracy --dataset cifar10 \
    --epochs 100 --seed 42 --bits 4 6 8 10 12 14 16 \
    --range-mode measured
