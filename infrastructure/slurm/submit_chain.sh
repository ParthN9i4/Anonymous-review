#!/usr/bin/env bash
# submit_chain.sh -- reschedule the re-runs as a CHAIN, not a pile.
#
# WHY (SLURM_Job_Scheduling_H200.docx S2, "Job Dependencies")
#   The lab shares three MIG slices on H200NODE. Submitting the re-runs as
#   independent jobs puts several of ours in the queue at once, each competing
#   for a slice; that is what made scancelling four jobs necessary.
#
#   --dependency=afterany chains them so AT MOST ONE of ours is ever eligible to
#   run. The rest are ineligible, not merely low-priority, so they cannot win a
#   slice ahead of another user no matter how the scheduler weights them. Total
#   GPU-hours are unchanged; the peak footprint drops to one slice.
#
#   afterany, not afterok: a task that hits its wall still leaves a recoverable
#   log (tools/recover_results_from_log.py), and one dead seed must not strand the
#   rest of the chain.
#
# ORDER matters and is not arbitrary -- see below.
#
# USAGE
#   bash infrastructure/slurm/submit_chain.sh            # show the plan, submit nothing
#   bash infrastructure/slurm/submit_chain.sh --go       # submit

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p slurm_logs

# Ordered by what the paper is blocked on, most blocking first.
#   1. reverify   -- the leak-free replacement for the headline CIFAR-10 table.
#                    Every suspended number in main.tex waits on this, and it is
#                    the cheapest at ~2.8h for all five seeds.
#   2. lut        -- ~21 min, and it gates the functional-bootstrapping argument.
#                    Cheap enough to slot in early. Needs --range-mode measured:
#                    the previous run inherited a BloodMNIST LUT domain and
#                    measured clamping, not precision.
#   3. cifar10    -- the A-I factorial, carrying H and I, the only arms citable
#                    against Powerformer.
#   4. cifar100   -- control. Last: droppable without touching a claim.
PLAN=(
  "reverify|sbatch --parsable infrastructure/slurm/rerun_verify_fixes.sbatch"
  "lut|sbatch --parsable infrastructure/slurm/lut_accuracy.sbatch"
  "cifar10|sbatch --parsable infrastructure/slurm/cifar_array.sbatch"
  "cifar100|sbatch --parsable --export=ALL,DATASET=cifar100 infrastructure/slurm/cifar_array.sbatch"
)

if [ "${1:-}" != "--go" ]; then
  echo "PLAN (nothing submitted -- pass --go):"
  i=0
  for entry in "${PLAN[@]}"; do
    i=$((i+1))
    name="${entry%%|*}"; cmd="${entry#*|}"
    dep=$([ $i -eq 1 ] && echo "starts when a slice frees" || echo "waits for step $((i-1))")
    printf "  %d. %-9s %s\n     %s\n" "$i" "$name" "$dep" "$cmd"
  done
  echo
  echo "At most ONE of these is ever eligible to run."
  exit 0
fi

PREV=""
for entry in "${PLAN[@]}"; do
  name="${entry%%|*}"; cmd="${entry#*|}"
  if [ -n "$PREV" ]; then
    cmd="${cmd/sbatch --parsable/sbatch --parsable --dependency=afterany:$PREV}"
  fi
  echo "+ $cmd"
  JID="$(eval "$cmd")"
  echo "  $name -> job $JID"
  PREV="$JID"
done

echo
echo "Chained. squeue will show one RUNNING (or PENDING for a slice) and the"
echo "rest PENDING with reason (Dependency) -- those cannot take a slice."
echo "Cancel the whole chain with: scancel -u \$USER --name=subst_arr,reverify,lut_acc"
