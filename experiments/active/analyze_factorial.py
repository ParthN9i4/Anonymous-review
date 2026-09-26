"""Paired factorial contrasts for the (G, A, N) substitution cube.

Consolidated 2026-09-14 from two scripts that computed overlapping subsets of
the same analysis (`analyze_factorial.py` + `analysis/interactions.py`). Two
analyzers meant two mask tables, and they disagreed: one keyed the 000 cell on
"0: unchanged KD" (the name substitution_ablation.py actually emits) and the
other on "Z: none (KD only)" (a name from the plan that was never implemented).
The second would have silently dropped the entire control arm. Mask names are
now read from substitution_ablation.py itself, so a rename there cannot leave a
stale copy behind here.

WHY THIS ANALYSIS AND NOT MAIN EFFECTS
--------------------------------------
One-at-a-time ablations cannot identify conditional effects. Each replacement
has an effect in FOUR backgrounds (twelve edges), each pair has TWO conditional
interactions (six faces), and the three-way term is the difference between a
pair's two faces. Averaged main effects hide exactly the structure this project
studies: on BloodMNIST the GELU edge is -0.31 / -0.20 pp in two backgrounds and
+46.84 pp in a third.

Three of the six faces need NO 000 arm, so they are computable on data that
already exists:

    I_AN|G=1 = M111 - M110 - M101 + M100
    I_GA|N=1 = M111 - M101 - M011 + M001
    I_GN|A=1 = M111 - M110 - M011 + M010

Everything is computed PER SEED inside a seed block and only then summarised;
unpaired cohort means are never subtracted. Collapse is a STATUS, not a number:
a collapsed run is counted in a rate, never averaged in as an ordinary
measurement. See docs/PREREGISTRATION.md for the criterion, the declared
equivalence margin, and the primary family.

Usage:
  python experiments/active/analyze_factorial.py results/ablations/substitution_ablation_bloodmnist_seeds*.json
  python experiments/active/analyze_factorial.py --dataset cifar10 --json-out analysis_out/cifar10.json \
      results/ablations/substitution_ablation_cifar10_clean8_seeds*.json
"""

import argparse
import ast
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

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_mask_table():
    """Read CONFIG_MASKS out of substitution_ablation.py without importing it.

    A plain `import` would pull in torch and timm just to read a dict of eight
    strings -- too heavy for a login node, and it would fail outright in an
    environment where only the results were copied. An AST parse costs
    milliseconds and still makes the experiment script the single source of
    truth for arm names, which is the point: the duplicate table is what let
    the 000 cell be keyed on a name nothing emits.
    """
    path = os.path.join(HERE, "substitution_ablation.py")
    try:
        tree = ast.parse(open(path).read())
    except (OSError, SyntaxError) as exc:
        raise SystemExit(f"cannot read arm names from {path}: {exc}")
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "CONFIG_MASKS"
                for t in node.targets):
            table = ast.literal_eval(node.value)
            missing = set(MASKS) - set(table.values())
            if missing:
                raise SystemExit(
                    f"substitution_ablation.py:CONFIG_MASKS does not cover the "
                    f"full cube; missing {sorted(missing)}")
            return table
    raise SystemExit(f"no CONFIG_MASKS assignment found in {path}")


# ── The cube. Mask order is (G, A, N). ───────────────────────────────
# The CE-only Teacher sits OUTSIDE the cube and is NOT the 000 cell: 000 is an
# unsubstituted student that still receives KD. Conflating them would charge the
# distillation gap to the substitutions.
MASKS = ("000", "100", "010", "001", "110", "101", "011", "111")
NAME_TO_MASK = _load_mask_table()
OUTSIDE_CUBE = ("Teacher", "H:", "I:")   # global-denominator arms: separate family

# Chance level per dataset. Balanced accuracy of a single-class predictor is
# exactly 1/K, so the collapse floor is dataset-determined, not chosen.
CHANCE = {"bloodmnist": 8, "cifar10": 10, "cifar100": 100}
COLLAPSE_TOL_PP = 2.0
DEFAULT_DELTA_PP = 1.0      # declared equivalence margin, docs/PREREGISTRATION.md

# 12 edges: (label, plus_mask, minus_mask, factor)
EDGES = [
    ("G in 000 bg", "100", "000", "G"), ("G in 010 bg", "110", "010", "G"),
    ("G in 001 bg", "101", "001", "G"), ("G in 011 bg", "111", "011", "G"),
    ("A in 000 bg", "010", "000", "A"), ("A in 100 bg", "110", "100", "A"),
    ("A in 001 bg", "011", "001", "A"), ("A in 101 bg", "111", "101", "A"),
    ("N in 000 bg", "001", "000", "N"), ("N in 100 bg", "101", "100", "N"),
    ("N in 010 bg", "011", "010", "N"), ("N in 110 bg", "111", "110", "N"),
]
# 6 conditional two-way faces: label -> {mask: coefficient}
FACES = {
    "I_AN | G=0": {"011": 1, "010": -1, "001": -1, "000": 1},
    "I_AN | G=1": {"111": 1, "110": -1, "101": -1, "100": 1},
    "I_GA | N=0": {"110": 1, "100": -1, "010": -1, "000": 1},
    "I_GA | N=1": {"111": 1, "101": -1, "011": -1, "001": 1},
    "I_GN | A=0": {"101": 1, "100": -1, "001": -1, "000": 1},
    "I_GN | A=1": {"111": 1, "110": -1, "011": -1, "010": 1},
}
THREE_WAY = {"111": 1, "110": -1, "101": -1, "011": -1,
             "100": 1, "010": 1, "001": 1, "000": -1}

# Metric aliases per result schema. BloodMNIST reports balanced accuracy as a
# FRACTION; the CIFAR family reports plain accuracy in PERCENT. The collapse
# criterion is stated on balanced accuracy, which coincides with plain accuracy
# only on a balanced test set -- true for CIFAR-10/100, false for BloodMNIST,
# which is why the blood path must not fall back to test_acc silently.
BLOOD_METRICS = {"balanced": ("test_bal_acc",), "plain": ("test_acc",)}
CIFAR_METRICS = ("test_at_best_val", "test_final", "best_val")


def load(paths, metric, allow_tainted=False):
    """Return ({mask: {seed: acc_pct}}, {seed: teacher_pct}, [filenames]).

    Handles both result schemas in this repo:
      - substitution_ablation.py         {"with_kd_per_seed": {name: {seed: {...}}}}
      - substitution_ablation_bloodmnist {"results": {name: [run, ...]}}

    Refuses artifacts carrying an in-band TAINTED/SUPERSEDED marker. Those were
    produced under test-set checkpoint selection; docs/PREREGISTRATION.md
    forbids pooling them with corrected runs to inflate n. The marker is written
    by mark_tainted_artifacts.py from the audit_results.py ledger, so this guard
    is mechanical rather than a convention someone has to remember.
    """
    per, teacher, files, unknown = {}, {}, [], set()
    for p in sorted(paths):
        with open(p) as fh:
            d = json.load(fh)
        status = str(d.get("_STATUS", ""))
        if status and not allow_tainted:
            raise SystemExit(
                f"REFUSING {os.path.basename(p)}\n"
                f"  _STATUS : {status}\n"
                f"  _WHY    : {d.get('_WHY','(none)')}\n"
                f"  _ACTION : {d.get('_ACTION','(none)')}\n"
                f"  docs/PREREGISTRATION.md forbids pooling this with corrected runs.\n"
                f"  Pass --allow-tainted only to inspect it in isolation, never to "
                f"increase sample size.")
        if status:
            print(f"  [!] {os.path.basename(p)} is {status.split('--')[0].strip()} "
                  f"- inspecting in isolation, NOT citable")
        if str(d.get("STATUS", "")).startswith("PARTIAL"):
            print(f"  [!] {os.path.basename(p)} is a PARTIAL file from an "
                  f"unfinished job ({d.get('runs_completed','?')} runs)")
        files.append(os.path.basename(p))

        if isinstance(d.get("results"), dict):          # BloodMNIST family
            keys = BLOOD_METRICS["plain" if metric == "test_acc" else "balanced"]
            for name, runs in d["results"].items():
                if not isinstance(runs, list):
                    continue
                if name not in NAME_TO_MASK and not name.startswith(OUTSIDE_CUBE):
                    unknown.add(name)
                for r in runs:
                    acc = next((r[k] for k in keys if r.get(k) is not None), None)
                    if acc is None:
                        continue
                    acc *= 100.0                        # stored as a fraction here
                    if name == "Teacher":
                        teacher[int(r["seed"])] = acc
                    elif name in NAME_TO_MASK:
                        per.setdefault(NAME_TO_MASK[name], {})[int(r["seed"])] = acc

        if isinstance(d.get("with_kd_per_seed"), dict):  # CIFAR family
            for name, bysd in d["with_kd_per_seed"].items():
                if name not in NAME_TO_MASK:
                    if not name.startswith(OUTSIDE_CUBE):
                        unknown.add(name)
                    continue
                for s, r in bysd.items():
                    v = r.get(metric) if isinstance(r, dict) else r
                    if v is not None:
                        per.setdefault(NAME_TO_MASK[name], {})[int(s)] = float(v)
            for s, r in (d.get("teacher_per_seed") or {}).items():
                v = r.get(metric) if isinstance(r, dict) else r
                if v is not None:
                    teacher[int(s)] = float(v)

    if unknown:
        # Loud, because the failure mode this replaces was silent: an arm whose
        # name no table knows is simply absent from every contrast that needs it.
        print(f"  [!] UNRECOGNISED ARM NAMES, excluded from the cube: "
              f"{sorted(unknown)}")
    return per, teacher, files


def contrast(per, coefs, seeds):
    """Per-seed paired contrast. Returns None if any required cell is absent."""
    if any(m not in per for m in coefs):
        return None
    out = []
    for s in seeds:
        if any(s not in per[m] for m in coefs):
            return None
        out.append(sum(k * per[m][s] for m, k in coefs.items()))
    return out


def summarise(xs, delta):
    """mean, sd, 95% half-width, two-sided p vs 0, TOST verdict, achieved margin.

    TOST, not a large p-value: equivalence holds iff the 90% CI lies inside
    [-delta, +delta]. The achieved margin (|mean| + 90% half-width) is the
    tightest delta that would have passed, reported so the declared delta is
    visibly not chosen after seeing the data.
    """
    m, n = st.mean(xs), len(xs)
    sd = st.stdev(xs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    if not HAVE_SCIPY or n < 2:
        return m, sd, None, None, None, None
    ci = sps.t.ppf(0.975, n - 1) * se
    p = float(sps.ttest_1samp(xs, 0).pvalue)
    achieved = abs(m) + sps.t.ppf(0.95, n - 1) * se
    return m, sd, float(ci), p, bool(achieved <= delta), float(achieved)


def _record(store, label, xs, delta):
    m, sd, ci, p, tost, ach = summarise(xs, delta)
    store[label] = {
        "n": len(xs), "mean": m, "sd": sd, "per_seed": list(xs),
        "ci95": [m - ci, m + ci] if ci is not None else [None, None],
        "p_vs_zero": p, "tost_equivalent": tost, "achieved_margin_pp": ach,
    }
    return m, sd, ci, p, tost, ach


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+",
                    help="result JSONs (globs allowed; per-seed files merge)")
    ap.add_argument("--dataset", default="bloodmnist", choices=sorted(CHANCE))
    ap.add_argument("--metric", default="test_at_best_val",
                    choices=sorted(set(CIFAR_METRICS) | {"test_bal_acc", "test_acc"}),
                    help="CIFAR family: test_at_best_val (headline) / test_final "
                         "(selection-free sensitivity) / best_val. BloodMNIST "
                         "family: test_bal_acc (default, balanced) / test_acc.")
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA_PP,
                    help="equivalence margin in pp [docs/PREREGISTRATION.md]")
    ap.add_argument("--json-out", default=None,
                    help="also write the full contrast set as machine-readable JSON")
    ap.add_argument("--allow-tainted", action="store_true",
                    help="inspect a TAINTED/SUPERSEDED artifact in isolation; "
                         "never use this to pool with corrected runs")
    a = ap.parse_args()

    paths = [q for p in a.files for q in (sorted(glob.glob(p)) or [p])]
    per, teacher, files = load(paths, a.metric, a.allow_tainted)
    if not per:
        raise SystemExit("no factorial arms found in the given files")
    K = CHANCE[a.dataset]
    thresh = 100.0 / K + COLLAPSE_TOL_PP

    seeds = sorted(set.intersection(*[set(v) for v in per.values()]))
    print(f"files      : {', '.join(files)}")
    print(f"dataset    : {a.dataset}  (K={K}, chance={100.0/K:.2f}%, "
          f"collapse threshold <= {thresh:.2f}%)")
    print(f"metric     : {a.metric}")
    print(f"seed block : {seeds}   (paired contrasts use this block only)")
    print(f"delta      : {a.delta} pp   [docs/PREREGISTRATION.md]")
    missing = [m for m in MASKS if m not in per]
    if missing:
        print(f"ABSENT ARMS: {sorted(missing)}  -> contrasts needing them are skipped")

    out = {"dataset": a.dataset, "metric": a.metric, "delta_pp": a.delta,
           "files": files, "seeds": seeds,
           "mask_order": ["GELU", "attention", "normalization"],
           "collapse_threshold_pct": thresh,
           "cells": {}, "edges": {}, "faces": {}, "three_way": {},
           "collapse": {}}

    print(f"\n{'mask':<6} {'n':>2} {'mean':>8} {'sd':>7}  {'collapsed':>9}  per-seed")
    print("-" * 92)
    if teacher:
        t = [teacher[s] for s in seeds if s in teacher]
        if t:
            print(f"{'(CE)':<6} {len(t):>2} {st.mean(t):>8.2f} "
                  f"{st.stdev(t) if len(t)>1 else 0:>7.2f}  {'--':>9}  "
                  + " ".join(f"{x:.2f}" for x in t) + "   <- teacher, OUTSIDE the cube")
            out["teacher"] = {"n": len(t), "mean": st.mean(t),
                              "per_seed": {str(s): teacher[s]
                                           for s in seeds if s in teacher}}
    for mask in MASKS:
        if mask not in per:
            continue
        xs = [per[mask][s] for s in seeds]
        nc = sum(x <= thresh for x in xs)
        print(f"{mask:<6} {len(xs):>2} {st.mean(xs):>8.2f} "
              f"{st.stdev(xs) if len(xs)>1 else 0:>7.2f}  {nc:>4}/{len(xs):<4}  "
              + " ".join(f"{x:.2f}" for x in xs))
        out["cells"][mask] = {
            "n": len(xs), "mean": st.mean(xs),
            "sd": st.stdev(xs) if len(xs) > 1 else 0.0,
            "collapsed": nc, "per_seed": {str(s): per[mask][s] for s in seeds}}

    # ── collapse-rate contrast, the declared primary test ─────────────
    if "011" in per and "111" in per and HAVE_SCIPY:
        k1 = sum(per["011"][s] <= thresh for s in seeds)
        k2 = sum(per["111"][s] <= thresh for s in seeds)
        n = len(seeds)
        p = float(sps.fisher_exact([[k1, n - k1], [k2, n - k2]])[1])
        floor = 2 / 2 ** n
        print(f"\nPRIMARY  collapse rate 011 vs 111 : {k1}/{n} vs {k2}/{n}   "
              f"Fisher exact two-sided p = {p:.4f}"
              f"{'' if p < 0.05 else '   (NOT significant at this n)'}")
        print(f"         distribution-free floor at n={n}: smallest attainable "
              f"two-sided sign-test p = {floor:.4f}")
        out["collapse"] = {"011": k1, "111": k2, "n": n, "fisher_p": p,
                           "sign_test_floor": floor}
        for kk, nm in ((k1, "011"), (k2, "111")):
            if kk == 0:
                bound = (1 - 0.05 ** (1 / n)) * 100
                print(f"         {nm}: 0/{n} collapsed -> one-sided 95% bound on the "
                      f"true rate is {bound:.1f}%, not 'reliable'")
                out["collapse"][f"{nm}_zero_failure_upper_bound_pct"] = bound

    print(f"\n{'TWELVE EDGES':<16} {'mean':>8} {'sd':>7} {'95% CI':>20} {'p':>8}")
    print("-" * 92)
    for label, plus, minus, _f in EDGES:
        xs = contrast(per, {plus: 1, minus: -1}, seeds)
        if xs is None:
            need = [m for m in (plus, minus) if m not in per]
            print(f"  {label:<14} {'--':>8}   SKIPPED - needs arm(s) {need}")
            out["edges"][label] = {"skipped_needs": need}
            continue
        m, sd, ci, p, _t, _a = _record(out["edges"], label, xs, a.delta)
        ci_s = f"[{m-ci:+7.2f},{m+ci:+7.2f}]" if ci is not None else ""
        print(f"  {label:<14} {m:>8.2f} {sd:>7.2f} {ci_s:>20} "
              f"{p if p is not None else float('nan'):>8.4f}")

    print(f"\n{'SIX FACES + 3-WAY':<16} {'mean':>8} {'sd':>7} {'95% CI':>20} "
          f"{'p':>8}  equivalence(d={a.delta})")
    print("-" * 92)
    for label, coefs in list(FACES.items()) + [("I_GAN (3-way)", THREE_WAY)]:
        store = out["three_way"] if "3-way" in label else out["faces"]
        xs = contrast(per, coefs, seeds)
        if xs is None:
            need = [m for m in coefs if m not in per]
            print(f"  {label:<14} {'--':>8}   SKIPPED - needs arm(s) {need}")
            store[label] = {"skipped_needs": need}
            continue
        m, sd, ci, p, tost, ach = _record(store, label, xs, a.delta)
        ci_s = f"[{m-ci:+7.2f},{m+ci:+7.2f}]" if ci is not None else ""
        eq = ""
        if tost is not None:
            eq = (f"EQUIVALENT (achieved {ach:.2f})" if tost
                  else f"not equivalent (needs d>={ach:.2f})")
        print(f"  {label:<14} {m:>8.2f} {sd:>7.2f} {ci_s:>20} "
              f"{p if p is not None else float('nan'):>8.4f}  {eq}")

    print("\nNOTE: an insignificant contrast is NOT equivalence. Nulls are reported")
    print("      as 'no clear effect detected' unless TOST passes at the declared delta.")

    if a.json_out:
        d = os.path.dirname(os.path.abspath(a.json_out))
        os.makedirs(d, exist_ok=True)
        tmp = a.json_out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, a.json_out)
        print(f"\nContrast JSON: {a.json_out}")


if __name__ == "__main__":
    main()
