#!/bin/bash
# One submission creates a frozen copy. Existing project files remain untouched.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
module load anaconda-2025.12-2
if [[ -f submission_receipt.txt ]]; then
 echo 'This extracted package has already submitted a campaign:'
 cat submission_receipt.txt
 exit 1
fi
python -m unittest test_core -v
PVIT_ENV="$HOME/pvit_envs/final_assessment_ckks"
if [[ ! -x "$PVIT_ENV/bin/python" ]]; then
 mkdir -p "$HOME/pvit_envs"
 python -m venv "$PVIT_ENV"
fi
"$PVIT_ENV/bin/python" -m pip install --disable-pip-version-check numpy==2.5.3 tenseal==0.3.18
"$PVIT_ENV/bin/python" -c 'import numpy,tenseal; print("FHE environment",numpy.__version__,tenseal.__version__)'
PVIT_ROOT="$(python prepare.py)"
python audit_previous.py "$HOME/pvit_results" "$PVIT_ROOT/previous_ckks_audit"
printf '%s\n' "$PVIT_ENV/bin/python" > "$PVIT_ROOT/fhe_python.txt"
printf '%s\n' "$PVIT_ROOT" > submission_receipt.txt
submit() {
 local result
 result="$(sbatch --parsable "$@")"
 printf '%s' "${result%%;*}"
}
PVIT_SMOKE="$(submit --output="$PVIT_ROOT/logs/smoke_%j.out" --error="$PVIT_ROOT/logs/smoke_%j.err" "$PVIT_ROOT/code/run_gpu.sbatch" "$PVIT_ROOT" smoke)"
printf '%s\n' "$PVIT_SMOKE" > "$PVIT_ROOT/smoke_job_id.txt"
PVIT_CRYPTO_SMOKE="$(submit --output="$PVIT_ROOT/logs/crypto_smoke_%j.out" --error="$PVIT_ROOT/logs/crypto_smoke_%j.err" "$PVIT_ROOT/code/run_cpu.sbatch" "$PVIT_ROOT" smoke)"
printf '%s\n' "$PVIT_CRYPTO_SMOKE" > "$PVIT_ROOT/crypto_smoke_job_id.txt"
PVIT_CONVERT="$(submit --array=0-25%1 --dependency="afterok:$PVIT_SMOKE" --output="$PVIT_ROOT/logs/convert_%A_%a.out" --error="$PVIT_ROOT/logs/convert_%A_%a.err" "$PVIT_ROOT/code/run_gpu.sbatch" "$PVIT_ROOT" conversion)"
printf '%s\n' "$PVIT_CONVERT" > "$PVIT_ROOT/conversion_job_id.txt"
PVIT_ENDPOINT="$(submit --array=0-14%1 --dependency="afterok:$PVIT_SMOKE,afterany:$PVIT_CONVERT" --output="$PVIT_ROOT/logs/endpoint_%A_%a.out" --error="$PVIT_ROOT/logs/endpoint_%A_%a.err" "$PVIT_ROOT/code/run_gpu.sbatch" "$PVIT_ROOT" endpoint)"
printf '%s\n' "$PVIT_ENDPOINT" > "$PVIT_ROOT/endpoint_job_id.txt"
PVIT_FHE="$(submit --array=0-11%1 --dependency="afterok:$PVIT_CRYPTO_SMOKE,afterany:$PVIT_CONVERT" --output="$PVIT_ROOT/logs/fhe_%A_%a.out" --error="$PVIT_ROOT/logs/fhe_%A_%a.err" "$PVIT_ROOT/code/run_cpu.sbatch" "$PVIT_ROOT" fhe)"
printf '%s\n' "$PVIT_FHE" > "$PVIT_ROOT/fhe_job_id.txt"
PVIT_REPORT="$(submit --dependency="afterany:$PVIT_CONVERT:$PVIT_ENDPOINT:$PVIT_FHE" --output="$PVIT_ROOT/logs/report_%j.out" --error="$PVIT_ROOT/logs/report_%j.err" "$PVIT_ROOT/code/run_report.sbatch" "$PVIT_ROOT")"
printf '%s\n' "$PVIT_REPORT" > "$PVIT_ROOT/report_job_id.txt"
printf 'RUN: %s\nGPU smoke: %s; CKKS smoke: %s\nConversions: %s (26); endpoint context: %s (15); FHE: %s (12); report: %s\n' "$PVIT_ROOT" "$PVIT_SMOKE" "$PVIT_CRYPTO_SMOKE" "$PVIT_CONVERT" "$PVIT_ENDPOINT" "$PVIT_FHE" "$PVIT_REPORT"
echo 'Submit once. afterok gates remain blocked if structural smoke checks fail. Numeric failures in full reports are retained.'
