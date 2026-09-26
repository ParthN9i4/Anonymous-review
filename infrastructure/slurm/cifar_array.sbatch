#!/bin/bash
#SBATCH --job-name=subst_arr
#SBATCH --output=slurm_logs/subst_%x_%A_%a.out
#SBATCH --error=slurm_logs/subst_%x_%A_%a.err
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
# Measured rate: 10 runs x ~34 min = ~5.7h measured
#SBATCH --array=42-46%1

# ONE SEED PER ARRAY TASK.
#
# Jobs 468/469 were designed as a single 40-run job (5 seeds x 8 models).
# Measured rate on this MIG slice is ~34 min/run, so 40 runs need ~22.7h --
# far past any wall we can request, and scontrol cannot extend a limit here.
# Both were killed with most of the grid unrun.
#
# One seed = 10 runs (teacher + A-I) = ~5.7h at the measured 34 min/run,
# which fits a 9h wall with margin. The array
# also parallelises across whatever slices are free instead of serialising,
# and a single seed dying costs one seed, not the whole grid.
#
# Dataset is chosen by DATASET (default cifar10):
#   sbatch infrastructure/slurm/cifar_array.sbatch
#   DATASET=cifar100 sbatch --export=ALL,DATASET=cifar100 infrastructure/slurm/cifar_array.sbatch

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


DS="${DATASET:-cifar10}"
echo "=== ${DS}, seed ${SLURM_ARRAY_TASK_ID}, all 9 configs (A-I) + teacher = 10 runs ==="
echo "    (no smoke test here -- it costs ~16 runs of log noise and the code"
echo "     is already validated; run experiments/active/smoke_test_substitution_ablation.py once"
echo "     interactively instead)"
python core/substitution_ablation.py --dataset "${DS}" \
    --seeds "${SLURM_ARRAY_TASK_ID}" --skip-no-kd
