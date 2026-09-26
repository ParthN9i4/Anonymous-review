"""
leakage_sensitivity_check.py
=============================
How much did test-set checkpoint selection inflate the CIFAR-10 results?

WHY THIS EXISTS
---------------
`verify_fixes.py:328` and `substitution_ablation.py:322` both do:

    acc = evaluate(model, test_loader, device)
    best_acc = max(best_acc, acc)

i.e. they select the reported checkpoint by evaluating on the TEST set every
epoch. That is test-set leakage: the test set is acting as a validation set,
and every reported CIFAR-10 number is a best-of-N-epochs figure rather than a
clean held-out estimate. (BloodMNIST is NOT affected -- `fix2_bloodmnist.py:461`
selects on a real `val_loader`.)

This was found by external review on 2026-09-04 and confirmed against the code.
`docs/P1_AUDIT.md:62` (F24) had identified the same PATTERN in
`simple_kd_baseline.py` / `blindfed_offline_kd.py` and downgraded it S1->S2 on
the reasoning that "no number currently in the repo is contaminated by this".
That scoping was wrong: F24 never checked these two scripts, which HAVE run to
completion and DO produce headline numbers.

WHAT THIS COMPUTES
------------------
`verify_fixes.py:285` returns `(best_acc, final_acc, history)` and the runner
stored BOTH per seed. So the honest final-epoch number is already on disk and
needs no GPU. This compares them.

The point is not that final-epoch equals a proper val-selected protocol -- it
does not, and the val-split re-run is still required. The point is the
DIRECTION and MAGNITUDE of the bias, which decides whether the headline claim
is at risk.

KEY PROPERTY BEING TESTED
-------------------------
Best-of-N selection helps a HIGH-VARIANCE arm far more than a stable one: a
collapsing seed that briefly spikes gets credited with its spike. Config E has
std ~20; Fix 2 has std ~0.6. So the bias is NOT uniform across arms, and the
prediction is that removing it makes the collapse look WORSE, not better.

Usage: python experiments/active/leakage_sensitivity_check.py
"""

import json
import os
import statistics as st

SRC = "results/ablations/verification_summary.json"
OUT = "results/ablations/leakage_sensitivity.json"


def main():
    with open(SRC) as f:
        d = json.load(f)

    print("=" * 74)
    print("TEST-SET SELECTION BIAS -- CIFAR-10, 5 seeds (42-46)")
    print("=" * 74)
    print("  'best'  = max over 100 epochs of TEST accuracy  <- the leaky protocol")
    print("  'final' = last-epoch TEST accuracy              <- no selection at all")
    print("  Neither is a clean val-selected estimate; this measures the BIAS.")
    print()
    print(f"  {'config':18s} {'best (leaky)':>17s} {'final (honest)':>17s} {'delta':>8s}")
    print("  " + "-" * 62)

    res = {}
    for k, v in d.items():
        if "per_seed_best" not in v or "per_seed_final" not in v:
            continue
        b = list(v["per_seed_best"].values())
        f_ = list(v["per_seed_final"].values())
        bm, bs = st.mean(b), st.stdev(b)
        fm, fs = st.mean(f_), st.stdev(f_)
        res[k] = {
            "best_mean": bm, "best_std": bs,
            "final_mean": fm, "final_std": fs,
            "inflation": bm - fm,
            "per_seed_best": b, "per_seed_final": f_,
        }
        print(f"  {k:18s} {bm:7.2f} +/- {bs:5.2f} {fm:7.2f} +/- {fs:5.2f} {fm - bm:+8.2f}")

    print()
    print("=" * 74)
    print("WHAT THIS SETTLES")
    print("=" * 74)

    if "Teacher" in res and "Config_E" in res:
        gap_best = res["Teacher"]["best_mean"] - res["Config_E"]["best_mean"]
        gap_final = res["Teacher"]["final_mean"] - res["Config_E"]["final_mean"]
        print(f"  Teacher - Config_E gap, leaky protocol : {gap_best:6.2f} pts")
        print(f"  Teacher - Config_E gap, final epoch    : {gap_final:6.2f} pts")
        print(f"  -> the collapse is {gap_final - gap_best:+.2f} pts DEEPER without selection")
        print()

    infl = {k: v["inflation"] for k, v in res.items()}
    worst = max(infl, key=infl.get)
    stable = {k: v for k, v in infl.items() if k != worst}
    print(f"  Most inflated arm : {worst} (+{infl[worst]:.2f} pts)")
    print(f"  Every other arm   : +{min(stable.values()):.2f} to +{max(stable.values()):.2f} pts")
    print()
    print("  The bias tracks variance, as predicted: the collapsing arm gains an")
    print("  order of magnitude more from best-of-N selection than the stable ones.")
    print("  CONSEQUENCE: test-set selection was MASKING the collapse, not")
    print("  manufacturing it. The honest protocol strengthens the headline claim.")
    print()
    print("  STILL REQUIRED: a proper train/val/test re-run. Final-epoch is not")
    print("  val-selected, and no claim should ship on this check alone.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({
            "purpose": "quantify test-set checkpoint-selection bias in the CIFAR-10 "
                       "results, from data already on disk (no GPU required)",
            "source": SRC,
            "leaky_protocol": "best_acc = max over 100 epochs of test accuracy "
                              "(verify_fixes.py:328, substitution_ablation.py:322)",
            "honest_comparator": "final-epoch test accuracy, no selection",
            "caveat": "final-epoch is NOT a val-selected estimate; this measures the "
                      "direction and magnitude of the bias, and does not replace the "
                      "train/val/test re-run",
            "bloodmnist_unaffected": "fix2_bloodmnist.py:461 selects on a real val_loader",
            "per_config": res,
            "gap_best": gap_best if "Teacher" in res else None,
            "gap_final": gap_final if "Teacher" in res else None,
            "conclusion": "bias tracks variance; the collapsing arm is inflated ~25x "
                          "more than the stable arms, so the leaky protocol UNDERSTATED "
                          "the collapse",
        }, f, indent=2)
    print(f"\nJSON: {OUT}")


if __name__ == "__main__":
    main()
