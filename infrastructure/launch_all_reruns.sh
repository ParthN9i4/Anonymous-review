#!/bin/bash
# infrastructure/launch_all_reruns.sh -- submit every rectified re-run to the H200.
#
# Run from the repo root ON H200NODE (this cannot be run from an assistant
# session: YOUR_H200_HOST is an RFC1918 address with no route from outside the
# institute network -- verified, TCP connect times out).
#
# See infrastructure/RERUN_PLAN.md for what was wrong and tools/audit_results.py for the live
# verdict on every artifact.

set -euo pipefail
cd "$(dirname "$0")"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found. Are you on H200NODE?" >&2
    exit 1
fi

echo "=== pre-flight: results audit ==="
python3 tools/audit_results.py || true      # exits non-zero while re-runs are pending

echo
echo "=== submitting ==="
J1=$(sbatch --parsable infrastructure/slurm/rerun_verify_fixes.sbatch)
echo "  ${J1}  verify_fixes, leak-free    (headline CIFAR-10 table + p-values)"

J2=$(sbatch --parsable infrastructure/slurm/cifar_array.sbatch)
echo "  ${J2}  factorial A-I, CIFAR-10    (incl. F and the global-denom arms H/I)"

J3=$(sbatch --parsable --export=ALL,DATASET=cifar100 infrastructure/slurm/cifar_array.sbatch)
echo "  ${J3}  factorial A-I, CIFAR-100   (Zimerman-comparable)"

echo
echo "Each is a 5-task array over seeds 42-46; two tasks run at a time on the"
echo "two free 2g.35gb MIG slices, the rest queue. Per-seed JSONs land in"
echo "results/ablations/ and are classified SAFE automatically by prefix."
echo
squeue -u "$USER"
echo
echo "When they finish:  python3 tools/audit_results.py   (exit 0 == nothing left to re-run)"
