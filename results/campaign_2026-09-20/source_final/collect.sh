#!/bin/bash
set -euo pipefail
module load anaconda-2025.12-2
PVIT_ROOT="$(cat "$HOME/pvit_results/latest_final_assessment.txt")"
cd "$PVIT_ROOT/code"
python report.py "$PVIT_ROOT"
PVIT_ARCHIVE="$(cat "$PVIT_ROOT/return_archive.txt")"
sha256sum "$PVIT_ARCHIVE"
du -h "$PVIT_ARCHIVE"
