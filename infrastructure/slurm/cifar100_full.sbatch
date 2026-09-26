#!/bin/bash
#SBATCH --job-name=subst_cifar100
#SBATCH --output=slurm_logs/subst_cifar100_%j.out
#SBATCH --error=slurm_logs/subst_cifar100_%j.err
#SBATCH --gres=gpu:nvidia_h200_nvl_2g.35gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
# H200NODE is a single-node cluster: one H200 split via NVIDIA MIG into
# one 3g.71gb slice and two 2g.35gb slices (confirmed via `scontrol show
# node H200NODE` on 2026-09-10). This tiny model (2.69M params) does not
# need the larger slice, and it was already in use by another job, so all
# three of our jobs target the smaller, free 2g.35gb slices; whichever
# can't start immediately queues (PD) until one frees up.
#SBATCH --partition=debug
#SBATCH --account=hpc
#SBATCH --qos=normal

set -euo pipefail
mkdir -p slurm_logs
# conda activate failed silently in the interactive session tonight
# (needs `conda init` first) -- what actually worked was loading the
# module directly; torch/torchvision/medmnist/timm are pip-installed into
# this user's ~/.local site-packages on top of it. Confirmed working via
# CUDA check + both smoke tests, 2026-09-10.
module load anaconda-2025.12-2
# Repository layout (2026-09-23): puts core/ and the repo root on PYTHONPATH and cd's to the root.
source "${SLURM_SUBMIT_DIR:-$PWD}/infrastructure/env.sh"
# Unbuffered stdout. Required, not cosmetic: tools/recover_results_from_log.py
# reconstructs a killed job's results by parsing this log, so anything
# still sitting in a buffer when SLURM kills the job is lost outright.
# (SLURM_Job_Scheduling_H200.docx S4.)
export PYTHONUNBUFFERED=1


echo "=== CIFAR-100, full 7-arm factorial (A-G) ==="
echo "    Directly comparable to Zimerman et al. (ICML 2024), whose own"
echo "    vision benchmark is CIFAR-100/Tiny-ImageNet, not CIFAR-10."
python core/substitution_ablation.py --dataset cifar100 --seeds 42 43 44 45 46 --skip-no-kd
