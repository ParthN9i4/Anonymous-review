"""
npre_cost_analysis.py
=====================
What would a MODEL-PRESERVING (SoK Class N-PRE) version of our ViT cost under
CKKS, and how does that compare to our model-modifying (Class N-MOD) Fix 2?

WHY THIS EXISTS
---------------
The Duality SoK (ePrint 2026/935) splits the field on one axis: model-preserving
(N-PRE) vs model-modifying (N-MOD). It reports that N-PRE frameworks never reach
Major accuracy loss, while *every* retraining/distillation framework it surveys
does -- because its own rubric grades ">1.5% OR requires retraining" as Major.
Our work retrains, so we are graded Major by construction.

The only honest reply is to show what model preservation would have COST us.
This script computes that from POLARIS's published configuration (SoK Table 1,
p. 12) applied to our architecture, and puts it beside our measured/estimated
Fix 2 budget.

It is an ANALYTICAL cost model, not a measurement. Every assumption is printed.
No CKKS library is invoked; nothing here is a timing.

Usage:  python experiments/active/npre_cost_analysis.py [--json out.json]
"""

import argparse
import json
import math

# ---------------------------------------------------------------------------
# Assumption 1: Paterson-Stockmeyer depth for a degree-d polynomial.
# Evaluating a degree-d Chebyshev series costs ceil(log2(d)) + 1 multiplicative
# levels. This is the standard PS bound and matches OpenFHE's
# EvalChebyshevSeries. Uncertainty is +/-1 level (domain mapping can add one).
# ---------------------------------------------------------------------------
def cheb_depth(degree: int) -> int:
    return math.ceil(math.log2(degree)) + 1


# ---------------------------------------------------------------------------
# Assumption 2: POLARIS's published recipe (SoK Table 1, BERT-Mini / SST-2).
# Degrees and intervals are quoted verbatim; the depth column is derived by
# cheb_depth() above, not published by them.
# ---------------------------------------------------------------------------
POLARIS = {
    "N": 2 ** 16,
    "slots": 2 ** 14,
    "scaling_bits": 50,
    "first_mod_bits": 53,
    "levels_between_bootstraps": 10,   # "Levels / L_BT / depth  10/16/26"
    "bootstrap_levels": 16,
    "total_depth": 26,
    "model": "BERT-Mini (4 layers, d=256, H=4, ~11M params)",
    "measured_latency_s": {"BERT-Tiny": 18.0, "BERT-Mini": 83.0},
    "seq_len": 24,
    "hardware": "1x NVIDIA A100, OpenFHE + FIDESLIB",
    # (name, degree) from Table 1
    "cheb": {
        "exp": 6,
        "inv_sqrt_softmax_iter1": 66,
        "inv_sqrt_softmax_iter2": 24,
        "inv_sqrt_softmax_iter3": 29,
        "inv_sqrt_layernorm": 59,
        "gelu": 119,
        "tanh_pooler": 31,
    },
    "softmax_squaring_iters": 3,
}

# ---------------------------------------------------------------------------
# Assumption 3: our architecture (measured, fix2_bloodmnist.py:265).
# ---------------------------------------------------------------------------
OURS = {
    "blocks": 6,
    "d": 192,
    "heads": 3,
    "tokens": 65,
    "params": 2_693_192,
}

# ---------------------------------------------------------------------------
# Assumption 4: our Fix 2 / raw-poly per-block budget, from paper/main.tex
# "Per-block depth budget" and fix5_evenpower_ablation.py::_depth().
# The ReLU and division figures are LITERATURE ESTIMATES, never measured:
#   minimax sign polynomial 13-14 levels (Lee et al., IEEE TDSC 2022)
#   Goldschmidt division ~2 levels/iter x 3-5 iters
# ---------------------------------------------------------------------------
LINEAR_PER_BLOCK = {
    "QKV projection": 1,
    "QK^T (ct-ct)": 1,
    "attention x V": 1,
    "output projection": 1,
    "MLP up-projection": 1,
    "MLP down-projection": 1,
}


def npre_softmax_depth():
    """Normalize-and-square softmax, POLARIS's choice, costed per attention."""
    c = POLARIS["cheb"]
    steps = [("exp (Chebyshev deg %d)" % c["exp"], cheb_depth(c["exp"]))]
    for i, key in enumerate(
            ["inv_sqrt_softmax_iter1", "inv_sqrt_softmax_iter2",
             "inv_sqrt_softmax_iter3"], start=1):
        d = c[key]
        steps.append((f"iter {i}: 1/sqrt(x) (deg {d})", cheb_depth(d)))
        steps.append((f"iter {i}: squaring", 1))
        steps.append((f"iter {i}: renormalize multiply", 1))
    return steps


def npre_layernorm_depth():
    d = POLARIS["cheb"]["inv_sqrt_layernorm"]
    return [
        ("mean / centring", 1),
        (f"1/sqrt(var) (Chebyshev deg {d})", cheb_depth(d)),
        ("scale multiply", 1),
    ]


def npre_gelu_depth():
    d = POLARIS["cheb"]["gelu"]
    return [(f"GELU (Chebyshev deg {d})", cheb_depth(d))]


def build_budgets():
    # ---- N-PRE block (POLARIS recipe on our architecture) ----
    npre = []
    npre += [(k, v) for k, v in LINEAR_PER_BLOCK.items()]
    npre += [("LayerNorm 1: " + n, v) for n, v in npre_layernorm_depth()]
    npre += [("Softmax: " + n, v) for n, v in npre_softmax_depth()]
    npre += [("LayerNorm 2: " + n, v) for n, v in npre_layernorm_depth()]
    npre += [("MLP " + n, v) for n, v in npre_gelu_depth()]

    # ---- N-MOD: raw polynomial block (Config E / homotopy target) ----
    raw = [(k, v) for k, v in LINEAR_PER_BLOCK.items()]
    raw += [("PolyAttn (degree 2)", 2),
            ("PolyGELU (degree 2)", 2),
            ("BatchNorm folds to affine", 0),
            ("BatchNorm folds to affine", 0)]

    # ---- N-MOD: Fix 2 block (raw + ReLU surrogate + row division) ----
    # Division cost is now MEASURED, not estimated: 6 Goldschmidt iterations
    # (13 levels) to reach <1% relative error over this model's real row-sum
    # range [2.947, 46.01], confirmed both in plaintext and inside a real CKKS
    # ciphertext at N=2^15 (measure_encrypted_primitives.py,
    # results/ablations/encrypted_primitives.json, 2026-08-31). The prior
    # 6-10 literature estimate was optimistic by 3 levels at its low end.
    DIV_MEASURED = 13
    fix2_lo = list(raw) + [("minimax sign/ReLU (est. lo)", 13),
                           ("Goldschmidt row division (MEASURED)", DIV_MEASURED)]
    fix2_hi = list(raw) + [("minimax sign/ReLU (est. hi)", 14),
                           ("Goldschmidt row division (MEASURED)", DIV_MEASURED)]

    # ---- N-MOD: EvenPow_RowSum -- the Gate 1 winner, 2026-08-27 ----
    # The perfect square (a*s+b)^2 + c^2 is non-negative by construction, so the
    # minimax sign polynomial disappears entirely. Measured to match Fix 2:
    # 94.92 +/- 0.28 vs 94.80 +/- 0.46, p=0.627, 5 seeds, BloodMNIST
    # (results/ablations/fix5_evenpower_ablation.json). The row division is
    # retained because removing it collapses the model (63.65 +/- 39.88).
    ep_lo = list(raw) + [("Goldschmidt row division (MEASURED)", DIV_MEASURED)]
    ep_hi = list(raw) + [("Goldschmidt row division (MEASURED)", DIV_MEASURED)]
    return npre, raw, fix2_lo, fix2_hi, ep_lo, ep_hi


def total(items):
    return sum(v for _, v in items)


def bootstraps(depth_per_block, blocks, usable):
    """Bootstraps needed for a full forward pass at `usable` levels per run."""
    total_levels = depth_per_block * blocks
    return max(0, math.ceil(total_levels / usable) - 1), total_levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    npre, raw, fix2_lo, fix2_hi, ep_lo, ep_hi = build_budgets()
    usable = POLARIS["levels_between_bootstraps"]
    B = OURS["blocks"]

    print("=" * 78)
    print("N-PRE vs N-MOD COST ANALYSIS")
    print("=" * 78)
    print("\nASSUMPTIONS (all of them):")
    print(f"  1. Chebyshev degree d costs ceil(log2(d))+1 levels (Paterson-")
    print(f"     Stockmeyer / OpenFHE EvalChebyshevSeries). Uncertainty +/-1.")
    print(f"  2. N-PRE approximation degrees are POLARIS's published values")
    print(f"     (SoK Table 1, p.12), applied unchanged to our architecture.")
    print(f"  3. Our model: {B} blocks, d={OURS['d']}, {OURS['tokens']} tokens,")
    print(f"     {OURS['params']:,} params (measured).")
    print(f"  4. ReLU (13-14) and Goldschmidt (6-10) are LITERATURE ESTIMATES,")
    print(f"     never measured on this model's value ranges.")
    print(f"  5. Compute budget between bootstraps = {usable} levels (POLARIS).")
    print("  Nothing below is a timing or a measurement.")

    print("\n" + "-" * 78)
    print("A. MODEL-PRESERVING (Class N-PRE) block, POLARIS recipe")
    print("-" * 78)
    for name, v in npre:
        print(f"  {name:<52} {v:>3}")
    npre_tot = total(npre)
    print(f"  {'TOTAL per block':<52} {npre_tot:>3}")

    sm = total([(n, v) for n, v in npre if n.startswith("Softmax")])
    ln = total([(n, v) for n, v in npre if n.startswith("LayerNorm")])
    gl = total([(n, v) for n, v in npre if n.startswith("MLP GELU")])
    print(f"\n  of which: softmax {sm}, both LayerNorms {ln}, GELU {gl}, "
          f"linear {total([(k, v) for k, v in LINEAR_PER_BLOCK.items()])}")
    print(f"  -> model-preserving softmax ALONE ({sm} levels) exceeds the entire")
    print(f"     {usable}-level compute budget between bootstraps by "
          f"{sm / usable:.1f}x.")

    print("\n" + "-" * 78)
    print("B. MODEL-MODIFYING (Class N-MOD) blocks, ours")
    print("-" * 78)
    raw_tot, f2lo, f2hi = total(raw), total(fix2_lo), total(fix2_hi)
    print(f"  Raw polynomial block (Config E / homotopy target){'':<4} {raw_tot:>3}")
    print(f"  Fix 2 block (adds ReLU surrogate + row division){'':<5} {f2lo:>3}-{f2hi}")

    print("\n" + "=" * 78)
    print("C. HEADLINE COMPARISON (per block, and full 6-block forward pass)")
    print("=" * 78)
    eplo, ephi = total(ep_lo), total(ep_hi)
    rows = [
        ("Model-preserving (N-PRE, POLARIS recipe)", npre_tot, npre_tot),
        ("Fix 2 (N-MOD, superseded)", f2lo, f2hi),
        ("EvenPow_RowSum (N-MOD, Gate 1 winner)", eplo, ephi),
        ("Raw polynomial (N-MOD, homotopy target)", raw_tot, raw_tot),
    ]
    print(f"  {'Configuration':<42}{'levels/blk':>12}{'total':>9}{'bootstraps':>12}")
    print(f"  {'-'*42}{'-'*12}{'-'*9}{'-'*12}")
    results = {}
    for name, lo, hi in rows:
        bt_lo, tot_lo = bootstraps(lo, B, usable)
        bt_hi, tot_hi = bootstraps(hi, B, usable)
        lvl = f"{lo}" if lo == hi else f"{lo}-{hi}"
        tot = f"{tot_lo}" if lo == hi else f"{tot_lo}-{tot_hi}"
        bt = f"{bt_lo}" if lo == hi else f"{bt_lo}-{bt_hi}"
        print(f"  {name:<42}{lvl:>12}{tot:>9}{bt:>12}")
        results[name] = {"levels_per_block": [lo, hi],
                         "total_levels": [tot_lo, tot_hi],
                         "bootstraps": [bt_lo, bt_hi]}

    ratio_lo = npre_tot / f2hi
    ratio_hi = npre_tot / f2lo
    ep_ratio_lo = npre_tot / ephi
    ep_ratio_hi = npre_tot / eplo
    print(f"\n  GATE 1 RESULT (2026-08-27): EvenPow_RowSum supersedes Fix 2. It matches")
    print(f"  Fix 2's accuracy (94.92+/-0.28 vs 94.80+/-0.46, p=0.627, 5 seeds) with the")
    print(f"  minimax-ReLU removed entirely, so the headline ratio is now")
    print(f"  {ep_ratio_lo:.1f}-{ep_ratio_hi:.1f}x rather than {ratio_lo:.1f}-{ratio_hi:.1f}x.")
    print(f"\n  Model preservation costs {ratio_lo:.1f}-{ratio_hi:.1f}x more "
          f"multiplicative depth than Fix 2,")
    print(f"  and {npre_tot / raw_tot:.1f}x more than a raw polynomial block.")

    bt_npre = bootstraps(npre_tot, B, usable)[0]
    bt_f2lo, bt_f2hi = bootstraps(f2lo, B, usable)[0], bootstraps(f2hi, B, usable)[0]
    bt_raw = bootstraps(raw_tot, B, usable)[0]

    print("\n" + "=" * 78)
    print("D. THE UNCOMFORTABLE READING")
    print("=" * 78)
    print(f"""  Fix 2 is NOT dramatically cheaper than model preservation. It costs
  {f2lo}-{f2hi} levels/block against N-PRE's {npre_tot} -- a factor of only
  {ratio_lo:.1f}-{ratio_hi:.1f}x, and {bt_f2lo}-{bt_f2hi} bootstraps against {bt_npre}.

  That is the finding this analysis actually produces, and it cuts against
  the current narrative. The motivation for leaving model-preservation is
  circuit cost. Fix 2 gives most of that advantage back: it pays for a
  comparison (minimax sign, 13-14 levels) AND a division (Goldschmidt, 6-10),
  which together are ~2/3 of its per-block budget. For a ~2x depth saving you
  accept retraining, a Major accuracy-loss grade under the SoK rubric, and
  the interaction-collapse fragility this paper documents.

  The raw polynomial block is the configuration where the argument works:
  {raw_tot} levels/block, {bt_raw} bootstraps, {npre_tot / raw_tot:.1f}x cheaper than N-PRE. But raw
  is exactly the configuration that COLLAPSES in training -- which is why the
  homotopy repair (fix4) and the even-power ablation (fix5, Gate 1) matter far
  more than they appeared to. If Gate 1's EvenPow_FixedDenom arm trains, it
  lands near the raw budget while remaining trainable, and the depth argument
  becomes decisive instead of marginal.

  WHAT THIS DOES NOT ARGUE: that our approach beats POLARIS. POLARIS pays the
  full {npre_tot} levels and gets <=0.5% accuracy loss with NO retraining, on real
  hardware, with open-source code and measured end-to-end latency. We have
  never run a single encrypted transformer block. On the SoK's own axes
  POLARIS is ahead on everything except depth.""")

    # ---- E. Sensitivity to the Chebyshev depth rule ----
    print("\n" + "=" * 78)
    print("E. SENSITIVITY (does the conclusion survive the +/-1 level uncertainty?)")
    print("=" * 78)
    sens = {}
    print(f"  {'Chebyshev rule':<20}{'N-PRE/blk':>11}{'softmax':>9}"
          f"{'vs Fix 2':>12}{'vs raw':>9}{'bootstraps':>12}")
    print(f"  {'-'*20}{'-'*11}{'-'*9}{'-'*12}{'-'*9}{'-'*12}")
    for off, label in [(0, "ceil(log2 d)"), (1, "ceil(log2 d)+1"),
                       (2, "ceil(log2 d)+2")]:
        c = POLARIS["cheb"]
        ln_ = 1 + (math.ceil(math.log2(c["inv_sqrt_layernorm"])) + off) + 1
        sm_ = ((math.ceil(math.log2(c["exp"])) + off)
               + sum((math.ceil(math.log2(c[k])) + off) + 2 for k in
                     ["inv_sqrt_softmax_iter1", "inv_sqrt_softmax_iter2",
                      "inv_sqrt_softmax_iter3"]))
        gl_ = math.ceil(math.log2(c["gelu"])) + off
        n_ = 6 + 2 * ln_ + sm_ + gl_
        bt_ = bootstraps(n_, B, usable)[0]
        tag = label + (" *" if off == 1 else "")
        print(f"  {tag:<20}{n_:>11}{sm_:>9}"
              f"{f'{n_/f2hi:.1f}-{n_/f2lo:.1f}x':>12}{f'{n_/raw_tot:.1f}x':>9}{bt_:>12}")
        sens[label] = {"npre_per_block": n_, "softmax": sm_, "bootstraps": bt_}
    print("  * = rule used above. The qualitative conclusion is unchanged across")
    print("    the whole band: N-PRE is ~2x Fix 2 and ~6x a raw polynomial block.")

    # ---- F. Cross-check the model against POLARIS's own measured latency ----
    print("\n" + "=" * 78)
    print("F. CROSS-CHECK: apply this cost model to POLARIS's OWN model")
    print("=" * 78)
    polaris_layers = 4
    polaris_total = npre_tot * polaris_layers
    polaris_bt = bootstraps(npre_tot, polaris_layers, usable)[0]
    meas = POLARIS["measured_latency_s"]["BERT-Mini"]
    per_bt = meas / polaris_bt if polaris_bt else float("nan")
    print(f"  BERT-Mini, {polaris_layers} layers x {npre_tot} levels = {polaris_total} levels")
    print(f"  -> {polaris_bt} bootstraps for their measured {meas:.0f}s "
          f"({POLARIS['seq_len']} tokens, A100)")
    print(f"  -> implies ~{per_bt:.2f} s per bootstrap-equivalent")
    print(f"  A published A100 CKKS bootstrap is order seconds, so the model is")
    print(f"  not wildly wrong. This is a plausibility check, NOT a validation.")
    tok_scale = OURS["tokens"] / POLARIS["seq_len"]
    est = bt_npre * per_bt * tok_scale
    print(f"\n  Extrapolated to our {B}-block, {OURS['tokens']}-token ViT:")
    print(f"    {bt_npre} bootstraps x {per_bt:.2f}s x {tok_scale:.2f} (token scaling)")
    print(f"    ~= {est:.0f} s per image, model-preserving, on an A100.")
    print(f"  CRUDE: ignores d=192 vs 256, attention's n^2 term, and packing.")
    print(f"  Quote only as an order of magnitude.")

    payload = {
        "sensitivity": sens,
        "polaris_crosscheck": {
            "polaris_total_levels": polaris_total,
            "polaris_bootstraps": polaris_bt,
            "polaris_measured_s": meas,
            "implied_s_per_bootstrap": per_bt,
            "our_extrapolated_s_per_image": est,
            "caveat": "order-of-magnitude only; ignores d, n^2 attention, packing",
        },
        "assumptions": {
            "cheb_depth_rule": "ceil(log2(degree)) + 1, +/-1",
            "polaris_source": "SoK ePrint 2026/935 Table 1 p.12",
            "relu_goldschmidt": "literature estimates, unmeasured",
            "levels_between_bootstraps": usable,
        },
        "architecture": OURS,
        "npre_block_items": [{"op": n, "levels": v} for n, v in npre],
        "npre_total_per_block": npre_tot,
        "npre_softmax_levels": sm,
        "npre_layernorm_levels": ln,
        "npre_gelu_levels": gl,
        "raw_total_per_block": raw_tot,
        "fix2_total_per_block": [f2lo, f2hi],
        "comparison": results,
        "depth_ratio_npre_over_fix2": [ratio_lo, ratio_hi],
        "depth_ratio_npre_over_evenpow_rowsum": [ep_ratio_lo, ep_ratio_hi],
        "evenpow_rowsum_per_block": [eplo, ephi],
        "gate1": {
            "date": "2026-08-27",
            "winner": "EvenPow_RowSum",
            "source": "results/ablations/fix5_evenpower_ablation.json",
            "finding": "ReLU is dead weight, row division is load-bearing",
            "evenpow_rowsum_acc": "94.92 +/- 0.28",
            "fix2_acc": "94.80 +/- 0.46",
            "p_vs_fix2": 0.6269,
        },
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nJSON written to {args.json}")


if __name__ == "__main__":
    main()
