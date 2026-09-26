# Rectification and re-run plan

Generated 2026-09-10. Run `python tools/audit_results.py` for the live verdict on
every artifact; this file explains the plan behind it.

## What was wrong

| Artifact | Verdict | Root cause (file:line) |
|---|---|---|
| `verification_summary.json`, `verification_curves.json` | **TAINTED** | `experiments/active/verify_fixes.py` selected its reported checkpoint on the **test set** every epoch (`best_acc = max(best_acc, acc)`), and its `get_cifar10` returned **no validation split at all**. This is the source of the paper's headline CIFAR-10 table *and every p-value*. |
| `substitution_ablation_seeds42_43_44_45_46.json` | **SUPERSEDED** | Produced before commit `7b3af21` added the 45k/5k val split. Confirmed by git: that commit rewrote the script but never touched this artifact, and the log's `► Best:` print format no longer exists in the code. |
| `activation_magnitudes.json` | **SUPERSEDED** | Mechanism probe measured alongside test-selected training; also has no `grad_norm` field at all (`docs/status/P1_AUDIT.md` F06). |
| BloodMNIST / RetinaMNIST / Gate 1 / diagnostics | **SAFE** | `core/fix2_bloodmnist.py:469` and `experiments/active/fix5_evenpower_ablation.py:414` both select on a real `val_loader`. No re-run needed. |

`leakage_sensitivity.json` quantified the damage: the bias tracks variance, so
the collapsing arm was inflated **+8.18 pts** while stable arms moved 0.18–0.32.
**Selection was masking the collapse, not manufacturing it** — the finding
survives, the numbers do not.

## What was fixed

1. **`experiments/active/verify_fixes.py` leak closed.** `get_cifar10` now delegates to
   `substitution_ablation.get_dataset` (fixed 45k/5k split, split seed
   independent of the model seed). Delegating rather than reimplementing keeps
   both experiments on an identical split. `train_one` selects on val and
   touches test exactly twice, after training. Verified programmatically: the
   training loop references `val_loader` and never `test_loader`.
2. **Two array-safety bugs caught before launch.** `--seeds` in
   `experiments/active/verify_fixes.py` is a *count* (`range(42, 42+N)`), so `--seeds 42` would
   have launched **42 seeds**. Added `--seed-list`. And both outputs used fixed
   filenames, so five array tasks would have clobbered each other *and*
   overwritten the historical artifact — outputs are now per-seed tagged.
3. **Monolithic jobs replaced by arrays.** Measured rate is ~34 min/run, so a
   40-run job needs ~22.7h; the 6–8h walls killed jobs 468/469 and `scontrol`
   cannot extend a limit here. One seed per array task fits comfortably.
4. **Checkpointing + log recovery** so a wall-clock kill is survivable.

## Re-run commands

```bash
# Safe to pull HERE: nothing is queued yet. Once the arrays are submitted, do
# not pull again until squeue is empty -- see the warning further down.
cd Private-ViT-FHE && git pull

sbatch infrastructure/slurm/rerun_verify_fixes.sbatch                              # headline CIFAR-10 table, leak-free
sbatch infrastructure/slurm/cifar_array.sbatch                                     # factorial A-I, CIFAR-10
sbatch --export=ALL,DATASET=cifar100 infrastructure/slurm/cifar_array.sbatch       # factorial A-I, CIFAR-100
# BloodMNIST factorial is job 470, already queued with checkpointing
```

Each is a 5-task array (seeds 42–46), 2 running at a time on the two free MIG
slices. Per-seed JSONs land in `results/ablations/`; aggregate across seeds
once all tasks finish.

## Getting results OFF the cluster (do this first)

Analysis does not happen on the H200, and relaying numbers by hand is slow and
lossy. `infrastructure/push_results.sh` ships artifacts to a dedicated `results-h200` branch:

```bash
bash infrastructure/push_results.sh                 # after any batch of tasks lands
# or leave it running:  watch -n 900 bash infrastructure/push_results.sh
```

[Certain] **Safe to run while an array is still going.** It never pulls, merges,
or checks anything out, and it stages *only* `results/ablations/*.json` and
`slurm_logs/*.out` — never a source file, so it cannot ship a half-finished code
change off the cluster. It pushes to a branch you do not have checked out, so
your local ref is untouched and you never need to pull to catch up. "Nothing
new" exits 0, so it is safe in a loop.

Each commit records the **training** code commit — resolved by walking back past
this script's own commits, not by reading HEAD, which becomes a results commit
after the first push. Pass that value to `tools/aggregate_seeds.py --source-commit`.

Verified against a throwaway repo: pushes to the results branch without
touching the checked-out branch, reports "nothing new" on a no-op run, leaves a
modified source file unstaged and unpushed, and reports the same training commit
across three successive pushes.

---

## When the jobs land: one block, run it on the H200

Paste this whole thing. It is idempotent — run it after every batch of tasks
finishes and it will tell you what is still missing.

> ### Do NOT `git pull` on the H200 while an array is still running
>
> [Certain] A queued array task starts a fresh `python core/substitution_ablation.py`,
> which reads the file **from disk at task-start time** — not at `sbatch` time.
> So a pull that lands between seed 43 finishing and seed 44 starting silently
> splits one 5-seed experiment across two code versions, and nothing in the
> output records that it happened.
>
> That is the same failure mode that produced the mess these re-runs exist to
> fix: `verification_summary.json` and `substitution_ablation_seeds*.json` are
> two different scripts' Config E numbers, and merging them was only caught
> after the fact.
>
> The tools below (`tools/recover_results_from_log.py`, `tools/aggregate_seeds.py`,
> `tools/audit_results.py`, `tools/mark_tainted_artifacts.py`) are read-only over
> `results/ablations/` and touch nothing the jobs use. If your checkout
> predates them, copy just those files across (`scp`) rather than pulling, or
> wait until `squeue` is empty and pull then.
>
> Record the commit the array is actually running, so the provenance is not a
> guess later: `git -C ~/Private-ViT-FHE rev-parse --short HEAD`.

```bash
cd ~/Private-ViT-FHE && module load anaconda-2025.12-2
git rev-parse --short HEAD    # the commit these results belong to -- write it down

# 1. What is still queued or running?
squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %.10L %R"

# 2. Rescue any task that hit its wall. A killed job is NOT worthless: every
#    completed run printed its result line, and this parses them back out.
#    (Safe to run on a live log too; unfinished runs are dropped, not zeroed.)
for f in slurm_logs/subst_*_*.out; do
  [ -e "$f" ] || continue
  python tools/recover_results_from_log.py "$f" \
      --dataset cifar10 \
      --out "results/ablations/$(basename "${f%.out}")_RECOVERED.json" || true
done

# 3. Merge the per-seed JSONs into one n=5 result per experiment.
#    --require-seeds exits non-zero if ANY (config, seed) cell is missing, so
#    this cannot quietly hand you a 4-seed mean.
python tools/aggregate_seeds.py --pattern 'verification_summary_seeds*'          --require-seeds --out results/ablations/MERGED_verify_fixes.json
python tools/aggregate_seeds.py --pattern 'substitution_ablation_cifar10_seeds*'  --require-seeds --out results/ablations/MERGED_cifar10.json
python tools/aggregate_seeds.py --pattern 'substitution_ablation_cifar100_seeds*' --require-seeds --out results/ablations/MERGED_cifar100.json
python tools/aggregate_seeds.py --pattern 'substitution_ablation_bloodmnist_seeds*' --require-seeds --out results/ablations/MERGED_bloodmnist.json

# 4. Ledger check. Exits non-zero while anything still needs a re-run.
python tools/audit_results.py
python tools/mark_tainted_artifacts.py --check
```

`tools/aggregate_seeds.py` refuses to merge any artifact carrying a TAINTED or
SUPERSEDED status, uses a `.PARTIAL` checkpoint only when that seed has no final
JSON (and labels it), and marks a `!` on any vs-teacher delta where the config
and the teacher were not run on the same seed set.

**Check `n_seeds` before quoting any mean** — a single-seed file is not a result.

### What H and I are, and why they are the ones to protect

Configs **H (`BPMaxsoftmax+norm`)** and **I (`BPMax all three`)** are runs 9 and
10 of 10 in every array task, so they are the first casualties of a wall hit.
They are also the only arms that can be cited against Powerformer: config F is
the same two categories but with **no denominator at all**, which makes it a
strictly harsher substitution than BPMax and therefore not a matched comparison
(`docs/design/INTERACTION_ANALYSIS_STRENGTHENING.md` §3). If a task dies, check the
recovery output's `MISSING` line for H/I specifically before treating that seed
as usable.

### The prediction these re-runs should test

`leakage_sensitivity.json` measured the selection bias directly and found it
tracks variance: Config E was inflated **+8.18 pts** while every stable arm moved
0.18–0.32, and the teacher-vs-E gap widens 36.46 → 44.46 under the final-epoch
comparator. So selection was **masking** the collapse. The leak-free re-run
should therefore make Config E look *worse* than 37.06 ± 18.11 and the gap
*wider*. **If E comes back less collapsed, do not celebrate — something else is
wrong**, and the split logic in `core/substitution_ablation.py` is the first thing to
re-read.

## Not re-run, and why

- **BloodMNIST (`fix2_bloodmnist_seeds*.json`)** — clean protocol, and it
  anchors the paper (`docs/design/BASELINE_AND_DATASETS.md`).
- **Gate 1 (`fix5_evenpower_ablation.json`)** — clean protocol. The *description*
  still needs correcting to `(a*s + b)^2`: the offset `c` is inert (init 0.0,
  enters as `c*c`, so the gradient is 0 at init and it never trains). The result
  stands; the wording does not.
- **Encrypted/CKKS artifacts** — numerical-fidelity checks, unaffected by
  training-protocol issues.
