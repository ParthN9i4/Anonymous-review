#!/bin/bash
# closeout_v2 launcher. Run on the H200 login node from this directory:
#   bash launch_v2.sh night1     prepare run root; smoke; CIFAR pilot gate; E1, E3, F0, F1 capture + F1 smoke
#   bash launch_v2.sh night2     E2 (dev); medical pilots -> medical E1 + E2; F1 encrypted chains; replay
# Medical: export PVIT_MEDICAL_RUN (the medical_replication_v1 run root containing manifest.json,
# train/, conversion_fit/) and PVIT_MEDICAL_DATA (directory with bloodmnist.npz etc.) before night1.
#   bash launch_v2.sh freeze     freeze the screen on dev seed 42 (login node, seconds)
#   bash launch_v2.sh night3     confirm pass on the CIFAR-10 test split (requires the freeze)
#   bash launch_v2.sh report     screen evaluation + CSV tables into the repository
# Array concurrency: PVIT_GPU_PAR (default 2), PVIT_CPU_PAR (default 8).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
STAGE="${1:?stage required}"
GPU_PAR="${PVIT_GPU_PAR:-2}"; CPU_PAR="${PVIT_CPU_PAR:-8}"
REPO="$(cd ../../.. && pwd)"; RESULTS="${PVIT_RESULTS:-$HOME/pvit_results}"
module load anaconda-2025.12-2
submit() { local r; r="$(sbatch --parsable "$@")"; printf '%s' "${r%%;*}"; }
root() { cat "$RESULTS/latest_closeout_v2.txt"; }
count() { python -c "import json,sys;print(len(json.load(open(sys.argv[1]))[sys.argv[2]]))" "$(root)/manifest.json" "$1"; }
case "$STAGE" in
night1)
  PYTHONPATH="$REPO/results/campaign_2026-09-20/source_final:." python -m unittest test_v2.PlanTests -v
  MED=(); [[ -n "${PVIT_MEDICAL_RUN:-}" ]] && MED=(--medical-run "$PVIT_MEDICAL_RUN" --medical-data "${PVIT_MEDICAL_DATA:?set PVIT_MEDICAL_DATA}")
  R="$(python prepare_v2.py --repo "$REPO" --results "$RESULTS" "${MED[@]}" | tail -1)"
  ENV="$HOME/pvit_envs/final_assessment_ckks"
  if [[ ! -x "$ENV/bin/python" ]]; then python -m venv "$ENV"; "$ENV/bin/python" -m pip install --disable-pip-version-check numpy==2.5.3 tenseal==0.3.18; fi
  printf '%s\n' "$ENV/bin/python" > "$R/fhe_python.txt"
  G=$(submit -o "$R/logs/smoke_%j.out" -e "$R/logs/smoke_%j.err" "$R/code/run_gpu_v2.sbatch" "$R" smoke)
  C=$(submit -o "$R/logs/ckks_smoke_%j.out" -e "$R/logs/ckks_smoke_%j.err" "$R/code/run_cpu_v2.sbatch" "$R" smoke)
  P=$(submit --array=0 --dependency=afterok:$G -o "$R/logs/pilot_%A_%a.out" -e "$R/logs/pilot_%A_%a.err" "$R/code/run_gpu_v2.sbatch" "$R" pilot)
  E1=$(submit --array=0-$(( $(count e1)-1 ))%$GPU_PAR --dependency=afterok:$P -o "$R/logs/e1_%A_%a.out" -e "$R/logs/e1_%A_%a.err" "$R/code/run_gpu_v2.sbatch" "$R" e1)
  E3=$(submit --array=0-$(( $(count e3)-1 ))%$GPU_PAR --dependency=afterok:$P -o "$R/logs/e3_%A_%a.out" -e "$R/logs/e3_%A_%a.err" "$R/code/run_gpu_v2.sbatch" "$R" e3)
  F0=$(submit --array=0-1 --dependency=afterok:$C -o "$R/logs/f0_%A_%a.out" -e "$R/logs/f0_%A_%a.err" "$R/code/run_cpu_v2.sbatch" "$R" f0)
  FC=$(submit --dependency=afterok:$P -o "$R/logs/f1cap_%j.out" -e "$R/logs/f1cap_%j.err" "$R/code/run_gpu_v2.sbatch" "$R" f1_capture)
  FS=$(submit --dependency=afterok:$FC:$C -o "$R/logs/f1smoke_%j.out" -e "$R/logs/f1smoke_%j.err" "$R/code/run_cpu_v2.sbatch" "$R" f1_smoke)
  printf 'RUN %s\nsmoke gpu=%s cpu=%s | pilot=%s e1=%s e3=%s f0=%s f1_capture=%s f1_smoke=%s\n' "$R" "$G" "$C" "$P" "$E1" "$E3" "$F0" "$FC" "$FS" | tee "$R/night1_jobs.txt" ;;
night2)
  R="$(root)"
  grep -q '"status": "measured"' "$R/f1/f1_smoke/report.json" || { echo 'F1 smoke missing: inspect logs before night2'; exit 1; }
  E2=$(submit --array=0-$(( $(count e2)-1 ))%$GPU_PAR -o "$R/logs/e2_%A_%a.out" -e "$R/logs/e2_%A_%a.err" "$R/code/run_gpu_v2.sbatch" "$R" e2)
  F1=$(submit --array=0-$(( $(count f1)-1 ))%$CPU_PAR -o "$R/logs/f1_%A_%a.out" -e "$R/logs/f1_%A_%a.err" "$R/code/run_cpu_v2.sbatch" "$R" f1)
  NP=$(count pilot); MP=none; ME1=none; ME2=none
  if (( NP > 1 )); then
    MP=$(submit --array=1-$(( NP-1 )) -o "$R/logs/pilot_%A_%a.out" -e "$R/logs/pilot_%A_%a.err" "$R/code/run_gpu_v2.sbatch" "$R" pilot)
    ME1=$(submit --array=0-$(( $(count med_e1)-1 ))%$GPU_PAR --dependency=afterok:$MP -o "$R/logs/med_e1_%A_%a.out" -e "$R/logs/med_e1_%A_%a.err" "$R/code/run_gpu_v2.sbatch" "$R" med_e1)
    ME2=$(submit --array=0-$(( $(count med_e2)-1 ))%$GPU_PAR --dependency=afterok:$MP -o "$R/logs/med_e2_%A_%a.out" -e "$R/logs/med_e2_%A_%a.err" "$R/code/run_gpu_v2.sbatch" "$R" med_e2)
  fi
  FI=$(submit --dependency=afterany:$F1 -o "$R/logs/f1inj_%j.out" -e "$R/logs/f1inj_%j.err" "$R/code/run_gpu_v2.sbatch" "$R" f1_inject)
  printf 'RUN %s\ne2=%s med_pilot=%s med_e1=%s med_e2=%s f1=%s f1_inject=%s\n' "$R" "$E2" "$MP" "$ME1" "$ME2" "$F1" "$FI" | tee "$R/night2_jobs.txt" ;;
freeze)
  R="$(root)"; cd "$R/code"; python screen_v2.py freeze "$R" ;;
night3)
  R="$(root)"; [[ -f "$R/screen_frozen.json" ]] || { echo 'Freeze the screen first'; exit 1; }
  CF=$(submit --array=0-$(( $(count confirm)-1 ))%$GPU_PAR -o "$R/logs/confirm_%A_%a.out" -e "$R/logs/confirm_%A_%a.err" "$R/code/run_gpu_v2.sbatch" "$R" confirm)
  printf 'RUN %s\nconfirm=%s\n' "$R" "$CF" | tee "$R/night3_jobs.txt" ;;
report)
  R="$(root)"; cd "$R/code"
  [[ -f "$R/screen_frozen.json" ]] && python screen_v2.py evaluate "$R" > /dev/null || true
  python analyze_v2.py "$R" --out "$REPO/results/closeout_v2"
  python "$REPO/paper/scripts/make_v2_figures.py" --tables "$REPO/results/closeout_v2" --out "$REPO/paper/revised/figures"
  python "$REPO/paper/scripts/make_v2_tex.py"
  python "$REPO/paper/revised/verify_submission.py" || echo "verify_submission: see the list above (pending items are expected until every run is in)" ;;
*) echo "unknown stage $STAGE"; exit 2 ;;
esac
