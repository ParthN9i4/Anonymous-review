#!/bin/bash
set -euo pipefail
PVIT_ROOT="$(cat "$HOME/pvit_results/latest_final_assessment.txt")"
PVIT_JOBS="$(paste -sd, "$PVIT_ROOT"/*job_id.txt | paste -sd, -)"
sacct -j "$PVIT_JOBS" -P --format=JobID%24,State,Elapsed,ExitCode,MaxRSS
printf '\nRUN: %s\n' "$PVIT_ROOT"
if [[ -f "$PVIT_ROOT/summary/completion.json" ]]; then
 python -m json.tool "$PVIT_ROOT/summary/completion.json"
 cat "$PVIT_ROOT/summary/READOUT.md"
 cat "$PVIT_ROOT/return_archive.txt"
 cat "$PVIT_ROOT/return_sha256.txt"
else
 echo 'Report is not present yet. Inspect smoke/full logs for failed or blocked jobs.'
fi
