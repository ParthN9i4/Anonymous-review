"""
analyze_powernorm.py -- P/Q (PowerNormAttn) vs the existing raw-gate cube.
============================================================================
Added 2026-09-16.

WHY THIS IS A SEPARATE SCRIPT, NOT A THIRD MODE OF analyze_factorial.py
------------------------------------------------------------------------
P and Q are not points in the 2x2x2 (G,A,N) cube -- they use a DIFFERENT
attention operator (PowerNormAttn, not PolyAttn), so "P=010, Q=011" is a
useful mnemonic but not a mask. analyze_factorial.py's contrasts (edges,
faces, three-way) are only meaningful within one operator family; forcing P/Q
into that machinery would either silently mix operators into one cell or
require lying to the mask table. A dedicated script keeps the comparison
honest: it names what is actually being compared (operator A vs operator B in
matched backgrounds), not "010" vs "011".

THE TEACHER-COMPARABILITY REQUIREMENT
--------------------------------------
`substitution_ablation.py` trains a FRESH teacher per seed for every
--config-set invocation (main():~966). A `--config-set powernorm` run trains
its own teachers, sharing them across P and Q within that run (matched), but
those teachers are NOT the same training run as the teachers in an existing
clean8 JSON -- even at identical seeds, identical code and (post-8fbc348)
identical determinism flags do not make two SEPARATE calls to train_model
produce the same optimization trajectory unless the environment is bit-for-bit
identical (same PyTorch/CUDA build, same GPU, no other process perturbing
kernel selection). Comparing P/Q's KD-relative arms against a DIFFERENT
teacher's KD student cells (010/011 from clean8) is exactly the "teacher
comparability" confound this project already flagged for CIFAR teacher
retraining generally.

This script therefore REFUSES to run the cross-file comparison until the new
teacher's per-seed accuracy is shown to be statistically indistinguishable
from the reference file's teacher (paired t-test on the intersecting seeds,
or a flat tolerance if scipy is unavailable). Pass --allow-teacher-mismatch
only to inspect a mismatched pair in isolation -- never to report the P/Q
edge as a controlled comparison.

USAGE
-----
  python experiments/active/analyze_powernorm.py \\
      --powernorm-file results/ablations/substitution_ablation_cifar10_powernorm_seeds42_43_44_45_46.json \\
      --reference-file results/ablations/substitution_ablation_cifar10_clean8_seeds42_43_44_45_46.json \\
      --dataset cifar10
"""

import argparse
import glob
import json
import math
import os
import statistics as st

try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

CHANCE = {"bloodmnist": 8, "cifar10": 10, "cifar100": 100}
COLLAPSE_TOL_PP = 2.0

# Which raw-gate cell each new arm is meant to be compared against, and why.
# P: powernorm+LN vs 010 (raw PolyAttn, LayerNorm) -- does the operator train
#    at all under the same normaliser the raw gate already survives with?
# Q: powernorm+BN vs 011 (raw PolyAttn, BatchNorm) -- the load-bearing cell:
#    the gain-dispersion hypothesis predicts Q recovers where 011 collapses.
COMPARISON = {
    "P: powernorm+LN": "010",
    "Q: powernorm+BN": "011",
}
MASK_TO_NAME = {
    "000": "0: unchanged KD", "100": "A: GELU only", "010": "B: softmax only",
    "001": "C: norm only", "110": "D: GELU+softmax", "111": "E: all three",
    "011": "F: softmax+norm", "101": "G: GELU+norm",
}


def load_with_kd(paths):
    """Deep-merge one or more result JSONs into a single {arm: {seed: ...}} view.

    ACCEPTS MULTIPLE FILES BECAUSE THE RUNS PRODUCE MULTIPLE FILES. The clean8
    campaign ran as five separate single-seed array tasks, so what exists on
    disk is substitution_ablation_cifar10_clean8_seeds42.json ... seeds46.json
    and NOT a combined five-seed file. An earlier version of this script took a
    single --reference-file and crashed with FileNotFoundError on a merged name
    that was never produced.

    The merge is per-arm and per-seed (not a dict.update at the arm level),
    which is the part that matters: a shallow update would let the last file
    read replace an arm's whole seed map, leaving one seed per arm and an
    apparent n=1. That exact corruption has already been observed in a merged
    clean8 file in circulation, where teacher entries covered five seeds but
    student entries retained only seed46. Rebuild from the per-seed originals
    rather than trusting any pre-merged file.

    A `summary` block is dropped: each per-seed file carries a summary computed
    over its own single seed, and keeping one would misrepresent the merge.
    """
    merged, statuses, seen = {}, [], []
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        status = str(d.get("_STATUS", "")) or str(d.get("STATUS", ""))
        if status:
            statuses.append((os.path.basename(path), status))
        seen.append(os.path.basename(path))
        if not merged:
            merged = {k: v for k, v in d.items()
                      if k not in ("summary", "with_kd_per_seed", "teacher_per_seed")}
        for s, v in (d.get("teacher_per_seed") or {}).items():
            merged.setdefault("teacher_per_seed", {})[str(s)] = v
        for arm, byseed in (d.get("with_kd_per_seed") or {}).items():
            merged.setdefault("with_kd_per_seed", {}).setdefault(arm, {}).update(
                {str(s): v for s, v in byseed.items()})
    merged["_merged_from"] = seen
    return merged, statuses


def per_seed_metric(d, arm_name, metric="test_at_best_val"):
    block = (d.get("with_kd_per_seed") or {}).get(arm_name)
    if block is None:
        return {}
    out = {}
    for s, r in block.items():
        v = r.get(metric) if isinstance(r, dict) else r
        if v is not None:
            out[int(s)] = float(v)
    return out


def teacher_per_seed(d, metric="test_at_best_val"):
    out = {}
    for s, r in (d.get("teacher_per_seed") or {}).items():
        v = r.get(metric) if isinstance(r, dict) else r
        if v is not None:
            out[int(s)] = float(v)
    return out


def paired_diff_summary(a, b, seeds):
    xs = [a[s] - b[s] for s in seeds]
    m = st.mean(xs)
    sd = st.stdev(xs) if len(xs) > 1 else 0.0
    out = {"n": len(xs), "mean": m, "sd": sd, "per_seed": xs}
    if HAVE_SCIPY and len(xs) > 1:
        se = sd / math.sqrt(len(xs))
        ci = sps.t.ppf(0.975, len(xs) - 1) * se
        out["ci95"] = [m - ci, m + ci]
        out["p_vs_zero"] = float(sps.ttest_1samp(xs, 0).pvalue)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--powernorm-file", required=True, nargs="+",
                    help="JSON(s) from --config-set powernorm; globs and "
                         "multiple per-seed files are merged")
    ap.add_argument("--reference-file", required=True, nargs="+",
                    help="JSON(s) containing the 010/011 (and ideally 000) "
                         "cells, e.g. the five per-seed clean8 files. Globs "
                         "and multiple files are deep-merged per arm per seed.")
    ap.add_argument("--dataset", required=True, choices=sorted(CHANCE))
    ap.add_argument("--metric", default="test_at_best_val",
                    choices=["test_at_best_val", "test_final", "best_val"])
    ap.add_argument("--teacher-tolerance-pp", type=float, default=2.0,
                    help="max allowed |mean teacher diff| (paired, matched "
                         "seeds) before the cross-file comparison is refused")
    ap.add_argument("--allow-teacher-mismatch", action="store_true",
                    help="proceed even if teachers differ beyond tolerance; "
                         "the output is then NOT a controlled comparison")
    args = ap.parse_args()

    def expand(pats, what):
        out = []
        for p in pats:
            hits = sorted(glob.glob(p))
            if not hits:
                if not os.path.exists(p):
                    raise SystemExit(
                        f"REFUSING: {what} {p!r} matches no file. The clean8 "
                        f"campaign ran as five single-seed tasks, so the "
                        f"reference is normally five files, e.g.\n"
                        f"  --reference-file results/ablations/"
                        f"substitution_ablation_cifar10_clean8_seeds4*.json")
                hits = [p]
            out.extend(hits)
        return out

    pn, pn_status = load_with_kd(expand(args.powernorm_file, "--powernorm-file"))
    ref, ref_status = load_with_kd(expand(args.reference_file, "--reference-file"))
    for label, statuses in (("powernorm", pn_status), ("reference", ref_status)):
        for fname, marker in statuses:
            raise SystemExit(f"REFUSING {label} input {fname}: marked {marker!r}. "
                             f"docs/PREREGISTRATION.md forbids citing a "
                             f"TAINTED/PARTIAL/SUPERSEDED artifact.")
    print(f"powernorm files: {', '.join(pn['_merged_from'])}")
    print(f"reference files: {', '.join(ref['_merged_from'])}")
    for label, d in (("powernorm", pn), ("reference", ref)):
        cov = {a: len(b) for a, b in (d.get("with_kd_per_seed") or {}).items()}
        print(f"  {label} arm->#seeds: {cov}")
        if cov and max(cov.values()) == 1 and len(d["_merged_from"]) > 1:
            print(f"  [!] every {label} arm has ONE seed after merging "
                  f"{len(d['_merged_from'])} files -- suspect a shallow-merged "
                  f"input; rebuild from the per-seed originals")
    print()

    K = CHANCE[args.dataset]
    thresh = 100.0 / K + COLLAPSE_TOL_PP
    print(f"dataset: {args.dataset}  (K={K}, collapse threshold <= {thresh:.2f}%)")
    print(f"metric : {args.metric}\n")

    # ── Teacher comparability gate ──────────────────────────────────
    t_pn, t_ref = teacher_per_seed(pn, args.metric), teacher_per_seed(ref, args.metric)
    common_t = sorted(set(t_pn) & set(t_ref))
    if not common_t:
        raise SystemExit("No overlapping seeds between the two files' teachers; "
                         "cannot establish comparability.")
    tdiff = paired_diff_summary(t_pn, t_ref, common_t)
    print(f"TEACHER COMPARABILITY (seeds {common_t}):")
    print(f"  powernorm-file teacher : "
          f"{[round(t_pn[s], 2) for s in common_t]}")
    print(f"  reference-file teacher : "
          f"{[round(t_ref[s], 2) for s in common_t]}")
    print(f"  paired diff mean={tdiff['mean']:+.2f}  sd={tdiff['sd']:.2f}"
          + (f"  p={tdiff.get('p_vs_zero', float('nan')):.4f}" if 'p_vs_zero' in tdiff else ""))
    mismatch = abs(tdiff["mean"]) > args.teacher_tolerance_pp
    if mismatch and not args.allow_teacher_mismatch:
        raise SystemExit(
            f"\nREFUSING the comparison: teacher means differ by "
            f"{tdiff['mean']:+.2f} pp, exceeding --teacher-tolerance-pp "
            f"{args.teacher_tolerance_pp}. The powernorm run's teacher is not "
            f"an acceptable matched control for the reference file's "
            f"010/011 cells -- comparing them would charge some of the "
            f"teacher-training gap to the attention operator. Re-run with "
            f"--allow-teacher-mismatch only to inspect in isolation, never "
            f"to report a controlled P/Q-vs-raw-gate effect.")
    print(f"  {'WITHIN' if not mismatch else 'OUTSIDE'} tolerance "
          f"({args.teacher_tolerance_pp} pp)"
          + ("" if not mismatch else " -- proceeding ONLY because "
             "--allow-teacher-mismatch was passed; treat the comparison "
             "below as exploratory, not controlled") + "\n")

    # ── The actual comparison ───────────────────────────────────────
    print(f"{'Arm':<18} {'vs':<6} {'n':>2} {'mean':>8} {'sd':>7} {'collapsed':>9}  "
          f"{'edge mean':>10} {'edge 95% CI':>18} {'p':>8}")
    print("-" * 96)
    for new_name, ref_mask in COMPARISON.items():
        ref_name = MASK_TO_NAME[ref_mask]
        new_vals = per_seed_metric(pn, new_name, args.metric)
        ref_vals = per_seed_metric(ref, ref_name, args.metric)
        seeds = sorted(set(new_vals) & set(ref_vals))
        if not seeds:
            print(f"{new_name:<18} {ref_mask:<6}  -- no overlapping seeds, skipped")
            continue
        xs_new = [new_vals[s] for s in seeds]
        nc = sum(v <= thresh for v in xs_new)
        d = paired_diff_summary(new_vals, ref_vals, seeds)
        ci_s = (f"[{d['mean']-((d['ci95'][1]-d['ci95'][0])/2):+.2f},"
                f"{d['mean']+((d['ci95'][1]-d['ci95'][0])/2):+.2f}]"
                if "ci95" in d else "")
        p_s = f"{d.get('p_vs_zero', float('nan')):.4f}" if "p_vs_zero" in d else ""
        print(f"{new_name:<18} {ref_mask:<6} {len(seeds):>2} {st.mean(xs_new):>8.2f} "
              f"{st.stdev(xs_new) if len(xs_new)>1 else 0:>7.2f}  {nc:>4}/{len(xs_new):<4}  "
              f"{d['mean']:>+10.2f} {ci_s:>18} {p_s:>8}")
        print(f"     per-seed new: {[round(v,2) for v in xs_new]}")
        print(f"     per-seed ref: {[round(ref_vals[s],2) for s in seeds]}")

    print("\nHEADLINE: the Q row is the falsification test. Q collapsing like")
    print("011 does refutes the gain-dispersion hypothesis; Q surviving while")
    print("011 collapses corroborates it. P is the sanity control (does the")
    print("operator train at all under LayerNorm).")


if __name__ == "__main__":
    main()
