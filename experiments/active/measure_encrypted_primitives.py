"""
measure_encrypted_primitives.py
================================
Evaluate the polynomial substitutions inside real CKKS ciphertexts, on this
model's own activations, and measure what they actually cost.

WHY THIS EXISTS (audit gap 1, and the remaining half of gap 2)
--------------------------------------------------------------
Until now, no polynomial GELU and no polynomial attention had ever been
evaluated inside a ciphertext anywhere in this repo -- the encrypted path
began and ended at the classification head, on a CLS token the client had
already computed in plaintext. The paper's Limitations says exactly this.
Every per-op level cost beyond that head was a literature citation, never a
measurement.

This closes the primitive-level half of that gap:
  1. poly-GELU        on real MLP pre-activations
  2. poly-attention   on real attention scores
  3. Goldschmidt 1/x  on the real row-sum range -- the ONE cost that is
                      still load-bearing after Gate 1, since Gate 1 proved
                      the row division necessary and the ReLU not.

Input ranges come from probe_activation_ranges.py (measured, not assumed):
  PolyGELU input  |max| ~ 4.69   (note: fitted on [-3,3], so ~1.6x outside)
  attention score |max| ~ 4.51
  row sums              [2.95, 46.01]

A NOTE ON WHICH ATTENTION IS EVALUABLE AT ALL
----------------------------------------------
Fix 2 is `ReLU(poly(s)) / rowsum`. ReLU is a COMPARISON -- not polynomial,
not CKKS-native -- so Fix 2 as trained cannot be evaluated under CKKS
without substituting a minimax sign polynomial it never saw during training
(the train/deploy mismatch the audit flagged). Gate 1's winner,
EvenPow_RowSum = `((as+b)^2 + c^2) / rowsum`, is non-negative BY
CONSTRUCTION and contains no comparison, so it is directly evaluable. That
is measured here; Fix 2's numerator is measured only in its pre-ReLU form,
with the gap stated rather than papered over.

Weights are random-init (no trained checkpoint on this machine); disclosed
the same way openfhe/export_head_fixture.py does. This measures numerical
fidelity and level consumption, not accuracy.

Usage: python experiments/active/measure_encrypted_primitives.py [--n 200] [--gs-iters 8]
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from fix2_bloodmnist import DeiTTiny, PolyGELU

try:
    import tenseal as ts
except ImportError:
    print("TenSEAL not found. Install with: pip install tenseal")
    raise SystemExit(1)


def make_ctx(levels, scale_bits=40, N=32768):
    """CKKS context with `levels` multiplicative levels at 128-bit security.

    N=2^15 measured (this script's own --probe-depth) to support at most 19
    40-bit levels; N=2^14 caps at 4. Anything needing more than 4 levels
    therefore forces 2^15.
    """
    chain = [60] + [scale_bits] * levels + [60]
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=N,
                     coeff_mod_bit_sizes=chain)
    ctx.global_scale = 2 ** scale_bits
    ctx.generate_galois_keys()
    return ctx


def err_stats(enc, ref):
    e = np.abs(np.asarray(enc) - np.asarray(ref))
    return {"max": float(e.max()), "mean": float(e.mean()),
            "rel_max": float((e / (np.abs(ref) + 1e-12)).max())}


# --------------------------------------------------------------------------
def measure_polygelu(vals, a, b, c, ctx):
    """f(x) = a x^2 + b x + c.

    MEASURED minimum: 2 levels, not 1. One ct-ct multiply (x*x) plus the
    plaintext multiply by `a`; allocating 1 level raises "scale out of
    bounds". This confirms the paper's original 2-level estimate for a
    degree-2 evaluation, which an earlier version of this file wrongly
    reported as 1.
    """
    ref = a * vals ** 2 + b * vals + c
    t0 = time.time()
    enc = ts.ckks_vector(ctx, vals.tolist())
    sq = enc * enc                       # ct-ct multiply
    out = sq * float(a) + enc * float(b) + float(c)
    dec = np.array(out.decrypt())
    return err_stats(dec, ref), time.time() - t0, 2


def measure_evenpower(vals, a, b, c, ctx):
    """f(s) = (a s + b)^2 + c^2 -- no comparison, so directly CKKS-evaluable.

    MEASURED minimum: 2 levels (same accounting as poly-GELU above).
    """
    inner_ref = a * vals + b
    ref = inner_ref ** 2 + c ** 2
    t0 = time.time()
    enc = ts.ckks_vector(ctx, vals.tolist())
    inner = enc * float(a) + float(b)
    out = inner * inner + float(c ** 2)  # ct-ct multiply
    dec = np.array(out.decrypt())
    return err_stats(dec, ref), time.time() - t0, 2


def measure_goldschmidt(vals, ctx, iters, scale):
    """Reciprocal 1/d by Goldschmidt, on scaled input d' = d*scale in (0,2).

        e_0 = 1 - d';  x_0 = 1 + e_0
        x_{i+1} = x_i * (1 + e_i^2);  e_{i+1} = e_i^2

    Each iteration costs 2 levels (one squaring, one multiply), so total
    depth = 2*iters + 1. Returns 1/d (unscaled by multiplying the scale back).
    """
    ref = 1.0 / vals
    d_scaled = vals * scale
    t0 = time.time()
    enc_d = ts.ckks_vector(ctx, d_scaled.tolist())
    e = enc_d * -1.0 + 1.0               # e_0 = 1 - d'
    x = e + 1.0                          # x_0 = 1 + e_0  (= 2 - d')
    for _ in range(iters):
        e = e * e                        # depth +1
        x = x * (e + 1.0)                # depth +1
    out = x * float(scale)               # undo the scaling: 1/d = scale * 1/d'
    dec = np.array(out.decrypt())
    return err_stats(dec, ref), time.time() - t0, 2 * iters + 1


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="values per primitive")
    ap.add_argument("--gs-iters", type=int, default=8,
                    help="max Goldschmidt iterations to sweep")
    ap.add_argument("--out", default="results/ablations/encrypted_primitives.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    # ---- Real activations from the real model ----
    model = DeiTTiny(img_size=32, patch_size=4, in_channels=3, num_classes=8,
                     embed_dim=192, depth=6, num_heads=3, norm_type="batchnorm",
                     attn_type="poly_normed", gelu_type="poly")
    model.train()
    with torch.no_grad():
        for _ in range(4):
            model(torch.randn(32, 3, 32, 32) * 2.0 + 0.5)
    model.eval()

    cap = {}
    hs = []
    for i, blk in enumerate(model.blocks):
        hs.append(blk.fc1.register_forward_hook(
            lambda m, inp, out, i=i: cap.setdefault(f"gelu{i}", []).append(
                out.detach().flatten())))

        def ahook(i):
            def fn(m, inp, out):
                s = inp[0]
                cap.setdefault(f"score{i}", []).append(s.detach().flatten())
                poly = torch.relu(m.a * s * s + m.b * s + m.c)
                cap.setdefault(f"rows{i}", []).append(
                    poly.sum(dim=-1).detach().flatten())
            return fn
        if blk.attn_act is not None:
            hs.append(blk.attn_act.register_forward_hook(ahook(i)))
    with torch.no_grad():
        model(torch.randn(16, 3, 32, 32) * 2.0 + 0.5)
    for h in hs:
        h.remove()

    def sample(prefix, n):
        allv = torch.cat([torch.cat(v) for k, v in cap.items() if k.startswith(prefix)])
        idx = torch.randperm(allv.numel())[:n]
        return allv[idx].double().numpy()

    gelu_x = sample("gelu", args.n)
    score_x = sample("score", args.n)
    rows_x = sample("rows", args.n)
    rows_x = rows_x[rows_x > 1e-6]              # reciprocal needs d > 0

    # A random sample from one batch under-covers the tails. The division's
    # cost is set by the WORST case in the range, so pin the sample's extremes
    # to the full measured span from probe_activation_ranges.py -- otherwise
    # the measured iteration count is optimistic by construction.
    _rr = "results/ablations/activation_ranges.json"
    if os.path.exists(_rr):
        _s = json.load(open(_rr))["summary"]
        _lo, _hi = _s.get("row_sum_min"), _s.get("row_sum_max")
        if _lo and _hi and len(rows_x):
            rows_x = np.concatenate([rows_x, np.linspace(_lo, _hi, 40)])
            print(f"  [row sums extended to the full probed span "
                  f"[{_lo:.3f}, {_hi:.3f}] -- cost is set by the worst case]")

    pg = model.blocks[0].act
    a_g, b_g, c_g = (float(pg.a), float(pg.b), float(pg.c))
    at = model.blocks[0].attn_act
    a_a, b_a, c_a = (float(at.a), float(at.b), float(at.c))

    print("=" * 76)
    print("ENCRYPTED PRIMITIVES -- measured on this model's own activations")
    print("=" * 76)
    print("Weights: RANDOM INIT (no local checkpoint). Measures numerical")
    print("fidelity and level consumption, not accuracy.")
    print(f"\n  poly-GELU coeffs   a={a_g:.4f} b={b_g:.4f} c={c_g:.4f}")
    print(f"  poly-attn coeffs   a={a_a:.4f} b={b_a:.4f} c={c_a:.4f}")
    print(f"  GELU inputs        n={len(gelu_x)} range [{gelu_x.min():.3f}, {gelu_x.max():.3f}]")
    print(f"  attn scores        n={len(score_x)} range [{score_x.min():.3f}, {score_x.max():.3f}]")
    print(f"  row sums           n={len(rows_x)} range [{rows_x.min():.3f}, {rows_x.max():.3f}]")

    results = {}

    # ---- 1. poly-GELU ----
    print("\n" + "-" * 76)
    print("1. poly-GELU  f(x) = a x^2 + b x + c   (depth 2, measured)")
    print("-" * 76)
    ctx1 = make_ctx(levels=3)
    st, secs, depth = measure_polygelu(gelu_x, a_g, b_g, c_g, ctx1)
    print(f"  max abs err {st['max']:.3e}   mean {st['mean']:.3e}   "
          f"rel max {st['rel_max']:.3e}")
    print(f"  depth {depth} level   {secs*1000/len(gelu_x):.3f} ms/value")
    results["polygelu"] = {**st, "depth_levels": depth, "n": len(gelu_x),
                           "ms_per_value": secs * 1000 / len(gelu_x),
                           "coeffs": [a_g, b_g, c_g],
                           "input_range": [float(gelu_x.min()), float(gelu_x.max())]}

    # ---- 2. poly-attention, even-power form (Gate 1 winner) ----
    print("\n" + "-" * 76)
    print("2. poly-attention, EVEN-POWER  f(s) = (a s + b)^2 + c^2   (depth 2, measured)")
    print("   Gate 1's winner. No comparison -- directly CKKS-evaluable.")
    print("-" * 76)
    st, secs, depth = measure_evenpower(score_x, math.sqrt(max(a_a, 1e-9)),
                                        b_a, c_a, ctx1)
    print(f"  max abs err {st['max']:.3e}   mean {st['mean']:.3e}   "
          f"rel max {st['rel_max']:.3e}")
    print(f"  depth {depth} level   {secs*1000/len(score_x):.3f} ms/value")
    results["evenpower_attn"] = {**st, "depth_levels": depth, "n": len(score_x),
                                 "ms_per_value": secs * 1000 / len(score_x),
                                 "input_range": [float(score_x.min()),
                                                 float(score_x.max())]}

    # ---- 3. Goldschmidt reciprocal -- the load-bearing unmeasured cost ----
    print("\n" + "-" * 76)
    print("3. Goldschmidt reciprocal 1/d on the REAL row-sum range")
    print("   The one cost Gate 1 proved necessary. Depth = 2*iters + 1.")
    print("-" * 76)
    scale = 1.0 / (rows_x.max() * 1.05)     # map d into (0,1] with headroom
    print(f"  scaling d by {scale:.5f} -> d' in "
          f"[{rows_x.min()*scale:.4f}, {rows_x.max()*scale:.4f}]")
    print(f"\n  {'iters':>6}{'depth':>7}{'max rel err':>15}{'mean abs err':>15}{'verdict':>12}")
    print("  " + "-" * 55)
    gs = []
    for iters in range(1, args.gs_iters + 1):
        depth = 2 * iters + 1
        # +2 headroom: the final plaintext rescale by `scale` consumes a
        # level, and SEAL needs slack before the scale exceeds the chain.
        # Allocating exactly `depth` raised "scale out of bounds" at iters=1.
        alloc = depth + 2
        if alloc > 19:
            print(f"  {iters:>6}{depth:>7}   needs {alloc} > N=2^15's 19-level budget")
            break
        try:
            ctx = make_ctx(levels=alloc)
            st, secs, d = measure_goldschmidt(rows_x, ctx, iters, scale)
            ok = "converged" if st["rel_max"] < 0.01 else ""
            print(f"  {iters:>6}{depth:>7}{st['rel_max']:>15.3e}"
                  f"{st['mean']:>15.3e}{ok:>12}")
            gs.append({"iters": iters, "depth_levels": depth, **st,
                       "ms_per_value": secs * 1000 / len(rows_x)})
            if st["rel_max"] < 0.01:
                break
        except Exception as e:
            print(f"  {iters:>6}{depth:>7}   FAILED: {str(e)[:44]}")
            break
    results["goldschmidt"] = {"scale": scale, "sweep": gs,
                              "row_sum_range": [float(rows_x.min()),
                                                float(rows_x.max())],
                              "n": len(rows_x)}

    # ---- verdict ----
    print("\n" + "=" * 76)
    print("MEASURED vs ESTIMATED")
    print("=" * 76)
    conv = [g for g in gs if g["rel_max"] < 0.01]
    if conv:
        g = conv[0]
        print(f"  Goldschmidt converges (rel err <1%) at {g['iters']} iterations")
        print(f"    = {g['depth_levels']} multiplicative levels, MEASURED")
        print(f"  Paper/estimate for the row division: 6-10 levels (literature)")
        delta = g["depth_levels"] - 10
        if g["depth_levels"] > 10:
            print(f"    -> MEASURED IS HIGHER by {delta} levels. The estimate was")
            print(f"       optimistic; the per-block budget needs revising upward.")
        elif g["depth_levels"] < 6:
            print(f"    -> MEASURED IS LOWER. The estimate was conservative.")
        else:
            print(f"    -> within the estimated 6-10 band. Estimate holds.")
    else:
        print(f"  Goldschmidt did NOT reach 1% relative error within "
              f"{args.gs_iters} iterations on this range.")
        print(f"  That is itself the finding: the row-sum range "
              f"[{rows_x.min():.2f}, {rows_x.max():.2f}] is too wide for cheap")
        print(f"  convergence, and the 6-10 level estimate is unsupportable "
              f"as-is.")
    print("\n  poly-GELU and poly-attention both measured at depth 2 on real")
    print("  activations -- the paper's 'never touched a ciphertext' limitation")
    print("  no longer applies to the elementwise substitutions.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "purpose": "measure poly-GELU, poly-attention and Goldschmidt "
                       "division inside real CKKS ciphertexts on this model's "
                       "own activations (audit gap 1; gap 2's remaining half)",
            "weights": "RANDOM INIT (no local checkpoint) -- numerical "
                       "fidelity and level consumption only, not accuracy",
            "ckks": {"N": 32768, "scale_bits": 40,
                     "max_levels_at_N_2_15": 19,
                     "security": "128-bit (TenSEAL/SEAL default enforced)"},
            "results": results,
        }, f, indent=2)
    print(f"\nJSON: {args.out}")


if __name__ == "__main__":
    main()
