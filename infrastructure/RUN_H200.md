# Running the substitution-factorial jobs on the institute H200 — step by step

New server as of 2026-09-09 (previous A6000 runbook: `infrastructure/RUN_FIX4.md`). This one
uses SLURM, not a bare `nohup`/`tmux` session — jobs survive your laptop and
even the login node rebooting.

Three jobs, all in `infrastructure/slurm/`:
- `cifar10_fg.sbatch` — CIFAR-10, all 7 configs (A–G), ~2–4h estimated
- `cifar100_full.sbatch` — CIFAR-100, all 7 configs, ~2–4h estimated
- `bloodmnist_full.sbatch` — BloodMNIST, all 7 configs × 150 epochs = 35 runs, likely the longest

**None of these estimates are measured** (Hard Rule 6) — they're by analogy to
similarly-shaped prior runs on the A6000. Watch the first job's actual pace
via `squeue`/log tail and adjust `--time=` in the others if it's running long.

---

## Step 0 — Connect and get the code

Auth over HTTPS with a password doesn't work (GitHub retired that) — clone
over SSH instead. One-time key setup, then:

```bash
ssh YOUR_USERNAME@YOUR_H200_HOST
git clone -b main <your-fork-url>
cd Private-ViT-FHE
```

(If the repo already exists there from a previous sync:
`git fetch origin && git checkout main && git pull`.)

## Step 1 — This cluster's real facts (confirmed 2026-09-10, already baked into infrastructure/slurm/*.sbatch)

`H200NODE` is a **single-node** cluster — the node names itself. No
`hpcusage` command here (that's specific to some other institution's cheat
sheet). Confirmed via `sinfo`, `scontrol show node H200NODE`, and
`sacctmgr show associations user=YOUR_USERNAME`:

- **Partition:** `debug` (the only one; `infinite` time limit despite the name)
- **Account:** `hpc`, **QOS:** `normal`
- **GPU:** one physical H200, split via NVIDIA MIG into 3 slices —
  `nvidia_h200_nvl_3g.71gb` (×1, larger) and `nvidia_h200_nvl_2g.35gb` (×2,
  smaller). All three job scripts request a `2g.35gb` slice — this model is
  2.69M params, nowhere near needing the larger one, and the larger slice
  was already occupied by someone else's job when checked. With only 2 free
  small slices and 3 jobs, one job will sit `PD` (pending) in `squeue` until
  a slice frees up — that's expected, not a problem.

**These values are already in `infrastructure/slurm/*.sbatch` in the repo you just
cloned** — no manual editing needed unless `sinfo`/`scontrol` output looks
different when you check (cluster state can change; re-run those commands
if a job fails at submission with a partition/account/QOS error).

## Step 2 — One-time conda environment

```bash
conda create -n pvit-fhe python=3.10 -y
conda activate pvit-fhe
pip install torch torchvision numpy scipy medmnist timm
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If a CUDA module needs loading first (cluster-dependent, check Step 1's
`module avail` output): `module load cuda/<version>` before the `pip install`.

## Step 3 — Smoke test locally on a login/interactive node first

Both smoke-test scripts run on CPU with synthetic data — no need to burn a
SLURM allocation for this one:

```bash
python experiments/active/smoke_test_substitution_ablation.py
python experiments/active/smoke_test_substitution_ablation_bloodmnist.py
```

Expect `ALL SMOKE CHECKS PASSED` from both. If either fails here, something
about the H200's environment differs from what was checked before you got
this repo (different torch/torchvision version, etc.) — fix that before
spending real GPU time, not after.

## Step 4 — Submit the three jobs

```bash
mkdir -p slurm_logs
sbatch infrastructure/slurm/cifar10_fg.sbatch
sbatch infrastructure/slurm/cifar100_full.sbatch
sbatch infrastructure/slurm/bloodmnist_full.sbatch
```

Each `sbatch` prints a job ID. Check status any time:

```bash
squeue -u YOUR_USERNAME
tail -f slurm_logs/subst_cifar10_<jobid>.out      # live progress
```

You can log off now — these run detached under SLURM, independent of your
SSH session.

## Step 5 — Collect results (tomorrow)

```bash
ls results/ablations/ | grep substitution_ablation
```

Expect three new files:
- `substitution_ablation_cifar10_seeds42_43_44_45_46.json`
- `substitution_ablation_cifar100_seeds42_43_44_45_46.json`
- `substitution_ablation_bloodmnist_seeds42_43_44_45_46.json`

Then, from the repo root:

```bash
git add results/ablations/substitution_ablation_*.json slurm_logs/
git commit -m "Full 2x2x2 substitution factorial: CIFAR-10, CIFAR-100, BloodMNIST"
git push
```

Bring these back to the conversation and real numbers replace every
projected/estimated figure in `docs/design/INTERACTION_ANALYSIS_STRENGTHENING.md`.

## What "done" looks like vs. what needs a second look

| Observation | Meaning |
|---|---|
| All three JSONs land, no `.err` file has a traceback | Clean run — safe to build the factorial table from real numbers |
| Any `.err` has `CUDA out of memory` | Lower `--batch-size` (128 default) and re-submit just that job |
| A `.err` shows the `TypeError: Expected state_dict... NoneType` we fixed in `core/fix2_bloodmnist.py` | Means the running code is stale — re-`git pull` before re-submitting, the fix (commit `aa076e9`) should already be in `main` |
| `core/substitution_ablation.py`'s job finishes in far less than 2-4h | Worth double-checking a full 100-epoch, 5-seed, 7-config run actually happened and wasn't silently truncated — check the seed count in the JSON's `summary` block |

## Notes

- These jobs re-run configs A–E too, not just the new F/G — because the
  leakage-fix commit (`7b3af21`) that produced the current, correct
  `get_dataset()`/train_model() code was **never actually executed**: the
  results or log files that were on disk predate it. This run is the first real
  execution of the corrected protocol, not an incremental add.
- `--skip-no-kd` is passed on the CIFAR jobs to keep runtime down (matches
  the precedent set by the prior, now-superseded run). Drop it if you want
  the CE-only arm too — doubles run count for a question the KD-arm data
  already mostly answers.
- BloodMNIST's script re-trains "E: all three" fresh rather than reusing the
  cached `Config_E` checkpoint from `core/fix2_bloodmnist.py`, for self-containment.
  Pass `--skip-config-e` to `core/substitution_ablation_bloodmnist.py` (edit the
  sbatch file) if GPU time gets tight and you'd rather splice in the cached
  numbers from `results/ablations/fix2_bloodmnist_seeds*.json` by hand.
