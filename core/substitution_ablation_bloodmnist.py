"""
substitution_ablation_bloodmnist.py
====================================
BloodMNIST completion of the CIFAR-10 single/pair/triple substitution
ablation (substitution_ablation.py). Added 2026-09-09.

WHY THIS SCRIPT EXISTS
-----------------------
fix2_bloodmnist.py only ever trained 3 configs on BloodMNIST: Teacher,
Config_E (all three substitutions), Fix2_Normalized (the repair). It never
ran the CIFAR-10 script's single/pair breakdown (A: GELU only, B: softmax
only, C: norm only, D: GELU+softmax), so there has never been a BloodMNIST
answer to "which substitution actually hurts" -- only a CIFAR-10 one.

Separately, `substitution_ablation.py`'s CONFIGS list was missing 2 of the 8
points in the full 2x2x2 factorial (GELU x softmax x norm): "softmax+norm"
and "GELU+norm". F ("softmax+norm") is the structural analogue of
Powerformer's Table 5 (ACL 2025, p.11097: BPMax+BatchLN, 2 of our 3
substitution categories, on a real trained BERT-base -- NO collapse, and a net
gain over the baseline, 82.86% -> 83.08%). Note the full row, though: BPMax
alone 83.01 and Batch LN alone 83.81, so their COMBINATION is worse than the
better single substitution, and worse on all three tasks individually
(71.48->70.52, 87.91->86.76, 92.05->91.97). The sub-additive interaction is
present in their numbers too; the difference is magnitude, not direction.
This script completes that same factorial on BloodMNIST, the project's medical
dataset, not just CIFAR-10.

ARCHITECTURE / PROTOCOL -- reused verbatim from fix2_bloodmnist.py
--------------------------------------------------------------------
Imports DeiTTiny, get_bloodmnist_loaders, train_one, and the KD loss
straight from fix2_bloodmnist.py rather than reimplementing them, so this
run is bit-identical in architecture/hyperparameters to the existing
Teacher/Config_E/Fix2_Normalized numbers in
results/ablations/fix2_bloodmnist_seeds*.json: 32x32 input (BloodMNIST's
native 28x28 resized up), patch 4, 6 layers, 192 dim, 3 heads, T=4,
alpha=0.1, no T^2 scaling, grad clip 5.0, AdamW lr=1e-3 wd=0.05, 150 epochs,
seeds 42-46, ONE shared teacher checkpoint (--teacher-seed, default 42) used
for KD across all student seeds -- matching fix2_bloodmnist.py's protocol,
NOT substitution_ablation.py's per-seed-teacher protocol.

SCOPE NOTE: KD arm only, no CE-only ("no KD") arm. fix2_bloodmnist.py never
ran a no-KD arm either; adding one here would double the run count (70 runs
instead of 35) for a question (does the substitution break training even
without distillation signal) the CIFAR-10 script already answers. If that
question turns out to matter specifically for BloodMNIST, extend later.

Every config below except "E: all three" is a fresh, previously-unrun
combination on this dataset. "E: all three" == the existing "Config_E" in
fix2_bloodmnist.py; it is re-run here (not reused from the cached JSON/
checkpoint) so this script's output is self-contained and independently
reproducible without depending on another script's checkpoint file existing
on disk -- at the cost of 5 redundant runs. If GPU time is tight, an
operator can skip it and splice in the cached Config_E numbers instead.

Usage
-----
  # Smoke test
  python core/substitution_ablation_bloodmnist.py --seeds 42 --epochs 30

  # Full run (7 configs x 5 seeds = 35 runs, ~150 epochs each)
  python core/substitution_ablation_bloodmnist.py
"""

import argparse
import json
import os

import numpy as np
import torch

from fix2_bloodmnist import (
    DeiTTiny,
    get_bloodmnist_loaders,
    train_one,
    load_checkpoint,
    HAVE_SCIPY,
)

if HAVE_SCIPY:
    from scipy import stats as scipy_stats


# =====================================================================
# CONFIGS -- the full 2x2x2 factorial (GELU x softmax x norm), matching
# substitution_ablation.py's CIFAR-10 naming exactly so tables line up.
# =====================================================================

CONFIGS = {
    "Teacher":          {"norm": "layernorm", "attn": "standard", "gelu": "standard", "kd": False},
    # The 000 cell. Added 2026-09-14 -- the factorial ran A-G without it, so
    # every contrast needing it (3 of 12 edges, 3 of 6 faces, the three-way)
    # was uncomputable, and the CE-only Teacher is NOT a substitute: it differs
    # from the students by KD as well as by substitution, so using it here
    # charges the whole distillation gap to the polynomial replacements.
    # Matched by construction: identical to Teacher except kd=True.
    "0: unchanged KD":  {"norm": "layernorm", "attn": "standard", "gelu": "standard", "kd": True},
    "A: GELU only":     {"norm": "layernorm", "attn": "standard", "gelu": "poly",     "kd": True},
    "B: softmax only":  {"norm": "layernorm", "attn": "poly",     "gelu": "standard", "kd": True},
    "C: norm only":     {"norm": "batchnorm", "attn": "standard", "gelu": "standard", "kd": True},
    "D: GELU+softmax":  {"norm": "layernorm", "attn": "poly",     "gelu": "poly",     "kd": True},
    "E: all three":     {"norm": "batchnorm", "attn": "poly",     "gelu": "poly",     "kd": True},
    "F: softmax+norm":  {"norm": "batchnorm", "attn": "poly",     "gelu": "standard", "kd": True},
    "G: GELU+norm":     {"norm": "batchnorm", "attn": "standard", "gelu": "poly",     "kd": True},
}

STUDENT_CONFIGS = [name for name in CONFIGS if name != "Teacher"]


def summarize_results(all_results: dict) -> dict:
    summary = {"per_config": {}, "tests_vs_teacher": {}}
    for name, runs in all_results.items():
        accs = np.array([r["test_acc"] for r in runs])
        bal = np.array([r["test_bal_acc"] for r in runs])
        summary["per_config"][name] = {
            "test_acc_mean": float(accs.mean()),
            "test_acc_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "test_acc_min": float(accs.min()),
            "test_acc_max": float(accs.max()),
            "test_acc_seeds": accs.tolist(),
            "test_bal_acc_mean": float(bal.mean()),
            "test_bal_acc_std": float(bal.std(ddof=1)) if len(bal) > 1 else 0.0,
            "test_bal_acc_seeds": bal.tolist(),
            "n_seeds": int(len(accs)),
        }
    if HAVE_SCIPY and "Teacher" in all_results:
        t_accs = np.array([r["test_acc"] for r in all_results["Teacher"]])
        for name in STUDENT_CONFIGS:
            if name not in all_results:
                continue
            c_accs = np.array([r["test_acc"] for r in all_results[name]])
            if len(c_accs) > 1 and len(t_accs) > 1:
                t, p = scipy_stats.ttest_ind(c_accs, t_accs, equal_var=False)
                summary["tests_vs_teacher"][name] = {
                    "t": float(t), "p": float(p),
                    "diff_pp": float((c_accs.mean() - t_accs.mean()) * 100),
                }
    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 78)
    print("RESULTS  (BloodMNIST, full 2x2x2 substitution factorial)")
    print("=" * 78)
    tm = summary["per_config"]["Teacher"]["test_acc_mean"] * 100
    ts = summary["per_config"]["Teacher"]["test_acc_std"] * 100
    print(f"  Teacher (standard): {tm:.2f}% +/- {ts:.2f}%\n")
    print(f"  {'Configuration':<22} {'test_acc':>16} {'Drop vs teacher':>16}")
    print(f"  {'-'*22} {'-'*16} {'-'*16}")
    for name in STUDENT_CONFIGS:
        if name not in summary["per_config"]:
            continue
        s = summary["per_config"][name]
        m, sd = s["test_acc_mean"] * 100, s["test_acc_std"] * 100
        print(f"  {name:<22} {m:>8.2f}+/-{sd:<6.2f} {tm - m:>15.2f}%")
    if summary["tests_vs_teacher"]:
        print("\n  Welch's t vs teacher (two-sided, scipy t-distribution):")
        for name, t in summary["tests_vs_teacher"].items():
            star = " *" if t["p"] < 0.05 else ""
            print(f"    {name:<22} dAcc={t['diff_pp']:+7.2f}pp  p={t['p']:.4f}{star}")
    print("=" * 78)



def _save_partial(args, all_results):
    """Checkpoint after every completed run -- see substitution_ablation.py.

    This job is 7 configs x 5 seeds x 150 epochs = 35 runs against a 10h SLURM
    wall. Without this, a timeout loses every completed run. Atomic rename, so
    a kill mid-write cannot leave a corrupt file.
    """
    try:
        os.makedirs(args.results_dir, exist_ok=True)
        done = sum(len(v) for v in all_results.values())
        out = os.path.join(
            args.results_dir,
            f"substitution_ablation_bloodmnist_seeds"
            f"{'_'.join(str(s) for s in args.seeds)}.PARTIAL.json")
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"STATUS": "PARTIAL -- running or killed before finishing",
                       "runs_completed": done, "args": vars(args),
                       "dataset": "BloodMNIST", "results": all_results},
                      f, indent=2, default=float)
        os.replace(tmp, out)
    except Exception as e:
        print(f"  [WARN] partial-save failed (continuing): {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Full 2x2x2 substitution factorial on BloodMNIST "
                     "(completes substitution_ablation.py's CIFAR-10 design).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--teacher-seed", type=int, default=42,
                        help="Which seed's teacher is shared for KD across all "
                             "student seeds (matches fix2_bloodmnist.py's protocol).")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints_substitution_bloodmnist")
    parser.add_argument("--results-dir", type=str, default="./results/ablations")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-config-e", action="store_true",
                        help="Skip 'E: all three' (5 runs) and splice in the "
                             "already-published Config_E numbers from "
                             "results/ablations/fix2_bloodmnist_seeds*.json "
                             "instead, to save GPU time. Off by default -- see "
                             "module docstring for the self-containment tradeoff.")
    parser.add_argument("--only-configs", type=str, nargs="+", default=None,
                        metavar="NAME",
                        help="Run only these student arms (exact CONFIGS keys, "
                             "quoted). Lets a cohort be EXTENDED -- e.g. five "
                             "more seeds on just '011' and '111', or the new "
                             "'0: unchanged KD' control -- without spending GPU "
                             "re-running cells that already have results. "
                             "Merge the resulting JSONs at analysis time; "
                             "analyze_factorial.py accepts several files and "
                             "intersects their seed blocks.")
    parser.add_argument("--skip-teacher-phase", action="store_true",
                        help="Reuse the teacher checkpoint at --ckpt-dir instead "
                             "of retraining it. Only valid when that checkpoint "
                             "was produced by THIS script's Phase 1 -- students "
                             "distilled from a different teacher are not "
                             "comparable to the existing arms.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs_to_run = [n for n in STUDENT_CONFIGS if not (args.skip_config_e and n == "E: all three")]
    if args.only_configs:
        unknown = [n for n in args.only_configs if n not in CONFIGS]
        if unknown:
            raise SystemExit(
                f"[FATAL] --only-configs names no such arm: {unknown}\n"
                f"        valid: {list(STUDENT_CONFIGS)}\n"
                f"        Names must match CONFIGS exactly -- a typo would "
                f"silently run nothing and look like a completed cohort.")
        configs_to_run = [n for n in configs_to_run if n in args.only_configs]
        if not configs_to_run:
            raise SystemExit("[FATAL] --only-configs selected zero arms to run.")
    print(f"Device        : {device}")
    print(f"Seeds         : {args.seeds}")
    print(f"Epochs        : {args.epochs}   LR: {args.lr}   WD: {args.wd}   BS: {args.batch_size}")
    print(f"Teacher-for-KD: seed={args.teacher_seed} (shared across all students)")
    print(f"Configs       : {configs_to_run}")

    train_loader, val_loader, test_loader = get_bloodmnist_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        data_root=args.data_root,
    )
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  "
          f"Test: {len(test_loader.dataset)}")

    all_results = {"Teacher": []}

    # PHASE 1: Teacher at all seeds (identical to fix2_bloodmnist.py Phase 1)
    if args.skip_teacher_phase:
        # Extending a cohort: the students must distil from the SAME teacher
        # the existing arms used, so retraining it is not merely wasteful, it
        # would break comparability. Requires the checkpoint to be present.
        need = os.path.join(args.ckpt_dir, f"Teacher_seed{args.teacher_seed}.pt")
        if not os.path.exists(need):
            raise SystemExit(
                f"[FATAL] --skip-teacher-phase needs the existing teacher at\n"
                f"        {need}\n"
                f"        It is absent. Retraining it here would give these "
                f"students a DIFFERENT teacher from the completed arms, which "
                f"is exactly the confound this flag exists to avoid.")
        print(f"\nPhase 1 skipped: reusing teacher checkpoint {need}")
        all_results.pop("Teacher", None)
    else:
        for seed in args.seeds:
            res = train_one(
                "Teacher", CONFIGS["Teacher"], seed,
                train_loader, val_loader, test_loader,
                teacher_model=None, epochs=args.epochs, lr=args.lr, wd=args.wd,
                device=device, ckpt_dir=args.ckpt_dir,
            )
            all_results["Teacher"].append(res)
            _save_partial(args, all_results)

        teacher_bals = [r["test_bal_acc"] for r in all_results["Teacher"]]
        if np.mean(teacher_bals) < 0.55:
            print(f"\n[ABORT] Teacher mean bal_acc={np.mean(teacher_bals)*100:.1f}% is too low.")
            print("        Investigate before proceeding -- see fix2_bloodmnist.py's own guardrail.")
            return

    teacher_ckpt = os.path.join(args.ckpt_dir, f"Teacher_seed{args.teacher_seed}.pt")
    teacher_model, teacher_obj = load_checkpoint(teacher_ckpt, device)
    print(f"\nLoaded shared teacher (seed={args.teacher_seed}): "
          f"test_acc={teacher_obj['test_acc']*100:.2f}% | "
          f"bal_acc={teacher_obj['test_bal_acc']*100:.2f}%")

    # PHASE 2: each substitution config, WITH KD, all seeds
    for cfg_name in configs_to_run:
        all_results[cfg_name] = []
        for seed in args.seeds:
            res = train_one(
                cfg_name, CONFIGS[cfg_name], seed,
                train_loader, val_loader, test_loader,
                teacher_model=teacher_model, epochs=args.epochs,
                lr=args.lr, wd=args.wd,
                device=device, ckpt_dir=args.ckpt_dir,
            )
            all_results[cfg_name].append(res)
            _save_partial(args, all_results)

    if args.skip_config_e:
        print("\n[NOTE] 'E: all three' was skipped per --skip-config-e; splice in "
              "results/ablations/fix2_bloodmnist_seeds*.json's Config_E numbers "
              "manually before treating this JSON as complete.")

    # PHASE 3: aggregate, print, save
    summary = summarize_results(all_results)
    print_summary(summary)

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(
        args.results_dir,
        f"substitution_ablation_bloodmnist_seeds{'_'.join(str(s) for s in args.seeds)}.json",
    )
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "results": all_results,
            "summary": summary,
            "dataset": "BloodMNIST",
            "num_classes": 8,
            "img_size": 32,
            "patch_size": 4,
            "depth": 6,
            "embed_dim": 192,
            "num_heads": 3,
            "kd_T": 4.0,
            "kd_alpha": 0.1,
            "grad_clip": 5.0,
            "note": "Completes substitution_ablation.py's factorial (2x2x2, GELU x "
                    "softmax x norm) on BloodMNIST. 'E: all three' == fix2_bloodmnist.py's "
                    "Config_E, re-run fresh here for self-containment unless --skip-config-e.",
        }, f, indent=2, default=float)
    print(f"\nResults JSON: {out_path}")


if __name__ == "__main__":
    main()
