#!/bin/bash
#SBATCH --job-name=reverify
#SBATCH --output=slurm_logs/reverify_%A_%a.out
#SBATCH --error=slurm_logs/reverify_%A_%a.err
#SBATCH --gres=gpu:nvidia_h200_nvl_2g.35gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=09:00:00
#SBATCH --partition=debug
#SBATCH --account=hpc
#SBATCH --qos=normal
#SBATCH --nice=100

# ── Sharing H200NODE with the lab ──────────────────────────────────
# The node has THREE GPUs and 224 CPUs. cpus-per-task/mem were cut here
# (8->4, 32G->16G) as a diagnostic on 2026-09-11 after jobs sat
# PENDING(Priority)/PENDING(Resources) against an apparently-idle GI-4 --
# NOT because CPU/mem were suspected as the real cause. They almost
# certainly are not: 224 cores and ~257 GB RAM against an 8-core/32G ask
# leaves no plausible CPU/mem contention, and the GPU gres line below is
# unchanged. If this does NOT unstick the queue, revert it -- a smaller
# footprint is still marginally more polite, but don't mistake this for
# the fix; re-check `scontrol show job` for the real Reason field.
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
# Measured rate: 5 runs x ~34 min = ~2.8h measured
#SBATCH --array=42-46%1

# RE-RUN the paper's headline CIFAR-10 table under a leak-free protocol.
#
# results/ablations/verification_summary.json is TAINTED: experiments/active/verify_fixes.py
# selected its reported checkpoint by evaluating the TEST set every epoch
# (`best_acc = max(best_acc, acc)`), and its get_cifar10 returned no validation
# split at all. Every figure in the paper's CIFAR-10 table -- Teacher 76.54,
# Config_E 40.08, Fix1 69.33, Fix2 80.47, Fix3 71.35 -- and every p-value
# derives from it.
#
# Fixed 2026-09-10: get_cifar10 now delegates to substitution_ablation.get_dataset
# (fixed 45k/5k split, split seed independent of model seed), and train_one
# selects on val, touching test exactly twice after training. Delegating rather
# than reimplementing keeps this experiment and the substitution factorial on an
# identical split, so the two tables stay comparable.
#
# One seed per array task: the monolithic form is what killed jobs 468/469.

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


echo "=== verify_fixes, leak-free, seed ${SLURM_ARRAY_TASK_ID} ==="
python experiments/active/verify_fixes.py --seed-list "${SLURM_ARRAY_TASK_ID}"
