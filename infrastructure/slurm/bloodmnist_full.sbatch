#!/bin/bash
#SBATCH --job-name=subst_bloodmnist
#SBATCH --output=slurm_logs/subst_bloodmnist_%j.out
#SBATCH --error=slurm_logs/subst_bloodmnist_%j.err
#SBATCH --gres=gpu:nvidia_h200_nvl_2g.35gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
# H200NODE is a single-node cluster: one H200 split via NVIDIA MIG into
# one 3g.71gb slice and two 2g.35gb slices (confirmed via `scontrol show
# node H200NODE` on 2026-09-10). This tiny model (2.69M params) does not
# need the larger slice, and it was already in use by another job, so all
# three of our jobs target the smaller, free 2g.35gb slices; whichever
# can't start immediately queues (PD) until one frees up.
#SBATCH --partition=debug
#SBATCH --account=hpc
#SBATCH --qos=normal
#SBATCH --nice=100

# ── Sharing H200NODE with the lab ──────────────────────────────────
# The node has THREE GPUs and 224 CPUs, so the GPU is the only scarce
# resource here; cutting --mem or --cpus-per-task frees nothing anyone
# is waiting on.
#
#   gres/gpu:nvidia_h200_nvl_2g.35gb = 2      <- everyone competes for these
#   gres/gpu:nvidia_h200_nvl_3g.71gb = 1      <- nothing has ever requested it
#
# Measured GPU memory for this workload: ~1.5 GB (nvidia-smi, job 470).
# That is ~4% of a 2g.35gb slice, but 2g.35gb is the SMALLEST profile the
# node advertises, so a smaller request is not available.
#
# What actually helps, and is set below:
#   %1 on the array  one task at a time -- never hold more than one slice
#   --nice=100       lab jobs schedule ahead of ours at equal priority
#   --time           right-sized from the measured rate: an inflated wall
#                    blocks SLURM's backfill from slotting others into gaps
#
# The counterintuitive option: run on the 3g.71gb slice instead
# (infrastructure/slurm/retarget_gres.sh). Taking the big slice nobody requests leaves
# BOTH 2g.35gb slices to the lab, instead of one. Strictly better for
# them, and it schedules sooner for you.
#
# --time is set GENEROUSLY on purpose (SLURM_Job_Scheduling_H200.docx S4): a job
# that finishes early releases its slice immediately, but a job that hits its
# wall is KILLED and only an admin can extend it. That is what destroyed jobs
# 468/469. An earlier revision of this file cut the wall to squeeze SLURM's
# backfill -- wrong, because the documented workflow SHARES a MIG slice between
# jobs, so wall-clock stretches under contention and a tight margin is what
# gets you killed.
# Measured rate: 40 runs x ~9.4 min = ~6.3h measured

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


echo "=== smoke test (synthetic data, catches code bugs before real GPU time) ==="
python experiments/active/smoke_test_substitution_ablation_bloodmnist.py

echo "=== BloodMNIST, full 7-arm factorial (A-G), 150 epochs each, 35 runs ==="
python core/substitution_ablation_bloodmnist.py --seeds 42 43 44 45 46
