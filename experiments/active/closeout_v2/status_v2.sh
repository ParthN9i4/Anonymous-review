#!/bin/bash
# Progress of the latest closeout_v2 run: recorded / failed reports per group, queue state.
R="$(cat "${PVIT_RESULTS:-$HOME/pvit_results}/latest_closeout_v2.txt")"
echo "RUN $R"
for g in e1 e2 e3 confirm f0 f1; do
  printf '%-8s reports=%-4s errors=%s\n' "$g" "$(ls "$R/$g"/*/report.json 2>/dev/null | wc -l)" "$(ls "$R/$g"/*/error.json 2>/dev/null | wc -l)"
done
squeue -u "$USER" -o '%.10i %.14j %.8T %.10M %R' 2>/dev/null | head -40
