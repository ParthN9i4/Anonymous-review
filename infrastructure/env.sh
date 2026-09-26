#!/usr/bin/env bash
# Source from anywhere:  source infrastructure/env.sh
# Puts core/ (maintained modules imported by bare name, e.g. `from fix2_bloodmnist
# import DeiTTiny`) and the repository root (for `tools.*`) on PYTHONPATH, then
# moves to the repository root so the scripts' relative default paths
# (./results/ablations, ./results/figures) resolve.
PVIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PVIT_ROOT
export PYTHONPATH="${PVIT_ROOT}/core:${PVIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PVIT_ROOT}" || return 1
