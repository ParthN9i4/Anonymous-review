#!/bin/bash
# Pack reports, tables and logs (not checkpoints) of the latest run into one archive.
set -euo pipefail
R="$(cat "${PVIT_RESULTS:-$HOME/pvit_results}/latest_closeout_v2.txt")"
OUT="$R.return.tgz"
tar -czf "$OUT" -C "$(dirname "$R")" "$(basename "$R")"
sha256sum "$OUT" | tee "$OUT.sha256"
