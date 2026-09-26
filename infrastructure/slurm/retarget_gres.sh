#!/usr/bin/env bash
# retarget_gres.sh -- point an sbatch at a MIG profile other than 2g.35gb.
#
# WHY
#   Every sbatch in this repo hardcodes --gres=gpu:nvidia_h200_nvl_2g.35gb:1.
#   H200NODE exposes three MIG instances and only two of them are 2g.35gb, so
#   the largest slice is never requested by anything and sits idle while jobs
#   queue. The node is not busy; one profile is oversubscribed.
#
#   The exact gres STRING is cluster config, not something to guess, so this
#   reads it off the node rather than hardcoding a second name.
#
# USAGE
#   bash infrastructure/slurm/retarget_gres.sh                          # list what exists
#   bash infrastructure/slurm/retarget_gres.sh infrastructure/slurm/cifar_array.sbatch <gres-name>
#
# It writes a new sbatch beside the original and PRINTS the submit command.
# It does not submit: which job to move is a judgement call about what you are
# willing to requeue from zero.

set -euo pipefail
NODE="${NODE:-H200NODE}"

echo "=== gres configured on ${NODE} ==="
if ! scontrol show node "$NODE" 2>/dev/null | tr ' ' '\n' | grep -i '^gres' ; then
  echo "  (scontrol returned nothing -- try: sinfo -o '%20N %10T %40G')"
fi
echo
echo "=== profiles parsed out ==="
scontrol show node "$NODE" 2>/dev/null \
  | tr ' ,' '\n\n' | grep -oE 'gpu:[A-Za-z0-9_.]+:[0-9]+' | sort -u | sed 's/^/  /' || true
echo
echo "=== what is already allocated ==="
squeue -w "$NODE" -o "%.10i %.12j %.8T %.20b" 2>/dev/null || true

[ $# -lt 2 ] && { echo; echo "To retarget:  bash $0 <sbatch-file> <gres-name>"; echo \
  "e.g.        bash $0 infrastructure/slurm/cifar_array.sbatch nvidia_h200_nvl_3g.71gb"; exit 0; }

SRC="$1"; NEW_GRES="$2"
[ -f "$SRC" ] || { echo "no such file: $SRC" >&2; exit 1; }

# Refuse a name the node does not actually advertise. A typo here produces a
# job that pends forever with a reason that looks like contention.
if ! scontrol show node "$NODE" 2>/dev/null | grep -q -- "$NEW_GRES"; then
  echo "REFUSED: '${NEW_GRES}' does not appear in ${NODE}'s gres." >&2
  echo "Pick one of the profiles listed above, exactly as printed." >&2
  exit 1
fi

OUT="${SRC%.sbatch}_$(echo "$NEW_GRES" | sed 's/.*_//').sbatch"
sed -E "s|(--gres=gpu:)[A-Za-z0-9_.]+(:[0-9]+)|\1${NEW_GRES}\2|" "$SRC" > "$OUT"
chmod +x "$OUT"

echo
echo "wrote ${OUT}"
diff <(grep -- --gres "$SRC") <(grep -- --gres "$OUT") || true
echo
echo "Submit with:"
echo "  sbatch ${OUT}                                   # cifar10"
echo "  sbatch --export=ALL,DATASET=cifar100 ${OUT}     # cifar100"
echo
echo "NOTE: a bigger slice buys CONCURRENCY, not speed. At 2.69M params and"
echo "batch 128 the bottleneck is likely the 8-CPU dataloader, not the GPU."
