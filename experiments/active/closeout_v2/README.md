# closeout_v2 — mechanism, dose-response, screens, and one encrypted sub-block

This package answers the three strongest SaTML rejection risks for the fixed-weight
conversion study:

| Risk | Experiment |
|---|---|
| "You diagnosed your own weak converter." | E1 oracle attribution; E2 dose-response; F0 depth ledger (cost of fixing it) |
| "The CKKS evidence is disconnected from the failure." | F1: continuously encrypted MLP sub-block on the headline failing checkpoint (seed 43, budget, GELU) |
| "Nothing actionable; the screen rejects everything." | E4: pre-registered screens, frozen on dev seed 42, scored on seeds 43/44 and a locked test pass. **Developmental** evidence, not independent confirmation |
| "CIFAR-only mechanism." | E1-med and E2-med on BloodMNIST, PathMNIST and DermaMNIST (arm-000 checkpoints of the existing medical run) |

It reuses the frozen v1 campaign code (`results/campaign_2026-09-20/source_final/`) by
copying it into each run root. It never edits that code. `calibrate_v2` reproduces the v1
coefficients exactly (tested in `test_v2.py::test_v1_parity`).

Hypotheses, splits and stopping rules: [`PREREGISTRATION_v2.md`](PREREGISTRATION_v2.md).
Every value is fixed in [`protocol_v2.py`](protocol_v2.py).

## Files
| File | Role |
|---|---|
| `protocol_v2.py` | Splits, grids, job schedule (pilot 4, e1 48, e2 81, e3 72, confirm 81, med_e1 144, med_e2 243, f0 2, f1 32 jobs) |
| `medical_data.py` | MedMNIST loader, copied verbatim from the original medical pipeline |
| `variants.py` | Degree, domain and step overrides; attention oracle variants; per-block masks |
| `probes.py` | Per-site primitive errors (source path, converted path), reciprocal and row-mass errors; agreement metrics; Clopper–Pearson |
| `worker_v2.py` | One plaintext job: calibrate → convert → replay gate + dev/test → `report.json` + predictions |
| `screen_v2.py` | E4 freeze and evaluate |
| `f0_depth_ledger.py` | Measured CKKS levels per primitive, plus a labelled block composition |
| `fhe_common_v2.py`, `f1_mlp_block.py` | F1 capture (torch) → encrypt (TenSEAL) → hybrid replay (torch) |
| `analyze_v2.py` | CSV/JSON tables written into `results/closeout_v2/` |
| `prepare_v2.py`, `launch_v2.sh`, `run_*_v2.sbatch`, `status_v2.sh`, `collect_v2.sh` | H200 orchestration |
| `test_v2.py` | Plan, conversion, parity, screen and CKKS tests (skip cleanly without torch or TenSEAL) |

## Running on H200 (from this directory, on the login node)

```bash
git pull                                  # this repository, on the login node
cd experiments/active/closeout_v2
export PVIT_MEDICAL_RUN=~/pvit_results/medical_replication_v1    # contains manifest.json, train/, conversion_fit/
export PVIT_MEDICAL_DATA=/path/to/dir/with/bloodmnist.npz         # verify both paths first
bash launch_v2.sh night1                  # prepare + smoke + CIFAR pilot gate; E1 (48), E3 (72), F0 (2), F1 capture + smoke
bash status_v2.sh                         # progress; logs in <run_root>/logs/
bash launch_v2.sh night2                  # E2 (81); medical pilots -> med E1 (144) + med E2 (243); F1 (32 CPU); replay
bash launch_v2.sh freeze                  # freezes the screen on dev seed 42 (prints SHA-256)
bash launch_v2.sh night3                  # locked confirm pass on CIFAR-10 test (81)
bash launch_v2.sh report                  # screen evaluation + tables into results/closeout_v2/
bash collect_v2.sh                        # archive + SHA-256 for the artifact
```

Before the freeze, copy `screen_frozen.json` somewhere timestamped (commit it or e-mail it
to yourself). That is the evidence the thresholds predate the test pass.

Concurrency: `PVIT_GPU_PAR` (default 2 MIG jobs) and `PVIT_CPU_PAR` (default 8 CKKS jobs).
Measured locally: about 13 s per encrypted token-chain on 4 threads, so an F1 part (2 images × 65
tokens) takes about 30 min. Plaintext jobs take a few minutes each (not yet timed on H200; the
smoke logs report it).

## Local checks (no GPU, no data)

```bash
cd <copy containing source_final/* and these files>
python -m unittest test_v2 -v             # 16 tests, including pilot gate, medical parity and the two-pass calibration regression
python worker_v2.py --synthetic-selftest
python f0_depth_ledger.py --local-test
```
