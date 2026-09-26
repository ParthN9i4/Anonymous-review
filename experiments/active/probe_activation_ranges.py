"""
probe_activation_ranges.py
==========================
What value ranges do the polynomial substitutions actually see?

WHY THIS EXISTS
---------------
Every depth/precision claim about encrypting poly-GELU and poly-attention
depends on the input range those polynomials are evaluated over. CKKS
precision degrades with magnitude, and a degree-2 polynomial fitted on
[-3,3] behaves very differently if the real activations reach 1e8 (which
Config E's do, per results/ablations/activation_magnitudes.json).

Nothing in the repo had measured these ranges on the ACTUAL model at the
ACTUAL sites, so the encrypted-circuit work (audit gap 1) had no grounded
input distribution to test against. This produces that.

Instruments, per block, on real forward passes:
  - MLP pre-activation  (input to PolyGELU)
  - attention scores    (input to PolyAttn / EvenPowerAttn)
  - attention row sums  (denominator of the row-normalization -- the
                         quantity Goldschmidt division must invert, and the
                         one unmeasured cost in the Gate-1 winning arm)

Weights are random-init unless a checkpoint is supplied: no trained
BloodMNIST checkpoint exists on this machine (the A6000 has one). Ranges
from random init are indicative of SCALE, not of a trained model's exact
distribution -- stated here rather than hidden, same as
openfhe/export_head_fixture.py does.

Usage: python experiments/active/probe_activation_ranges.py [--config fix2|evenpow|raw] [--batches 8]
"""

import argparse
import json
import os

import numpy as np
import torch

from fix2_bloodmnist import DeiTTiny


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fix2",
                    choices=["fix2", "raw"],
                    help="fix2 = PolyAttnNormed (ReLU+rowsum); raw = PolyAttn (unnormalized)")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="results/ablations/activation_ranges.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    attn_type = "poly_normed" if args.config == "fix2" else "poly"

    model = DeiTTiny(img_size=32, patch_size=4, in_channels=3, num_classes=8,
                     embed_dim=192, depth=6, num_heads=3,
                     norm_type="batchnorm", attn_type=attn_type, gelu_type="poly")
    model.train()
    with torch.no_grad():
        for _ in range(4):
            model(torch.randn(32, 3, 32, 32) * 2.0 + 0.5)
    model.eval()

    print("=" * 74)
    print(f"Activation range probe -- config={args.config} (attn_type={attn_type})")
    print("=" * 74)
    print("Weights: RANDOM INIT (no local checkpoint). Ranges indicate SCALE,")
    print("not a trained model's exact distribution.")

    # Capture sites via hooks on the real modules.
    stats = {f"block{i}": {} for i in range(len(model.blocks))}

    def rec(store, key, t):
        a = t.detach().float()
        d = store.setdefault(key, {"min": np.inf, "max": -np.inf,
                                   "absmax": 0.0, "sum": 0.0, "n": 0})
        d["min"] = min(d["min"], a.min().item())
        d["max"] = max(d["max"], a.max().item())
        d["absmax"] = max(d["absmax"], a.abs().max().item())
        d["sum"] += a.abs().sum().item()
        d["n"] += a.numel()

    handles = []
    for i, blk in enumerate(model.blocks):
        # PolyGELU input == fc1 output
        handles.append(blk.fc1.register_forward_hook(
            lambda m, inp, out, i=i: rec(stats[f"block{i}"], "polygelu_input", out)))
        # attention scores == input to attn_act.
        # NOTE: a forward hook that RETURNS a value replaces the module's
        # output. Returning a tuple here silently corrupted the forward pass
        # ("unsupported operand @ for tuple"). The hook must return None.
        def attn_hook(i):
            def fn(m, inp, out):
                rec(stats[f"block{i}"], "attn_scores_input", inp[0])
                rec(stats[f"block{i}"], "attn_output", out)
            return fn
        if blk.attn_act is not None:
            handles.append(blk.attn_act.register_forward_hook(attn_hook(i)))

    # Row sums need computing explicitly (they're internal to PolyAttnNormed).
    rowsum_stats = {f"block{i}": {} for i in range(len(model.blocks))}

    def rowsum_hook(i):
        def fn(m, inp, out):
            s = inp[0]
            poly = m.a * s * s + m.b * s + m.c
            if attn_type == "poly_normed":
                poly = torch.relu(poly)
            rec(rowsum_stats[f"block{i}"], "row_sum", poly.sum(dim=-1))
        return fn

    for i, blk in enumerate(model.blocks):
        if blk.attn_act is not None:
            handles.append(blk.attn_act.register_forward_hook(rowsum_hook(i)))

    with torch.no_grad():
        for _ in range(args.batches):
            model(torch.randn(args.batch_size, 3, 32, 32) * 2.0 + 0.5)
    for h in handles:
        h.remove()

    def fin(d):
        return {"min": d["min"], "max": d["max"], "absmax": d["absmax"],
                "mean_abs": d["sum"] / d["n"]}

    out = {}
    print(f"\n{'site':<22}{'min':>13}{'max':>13}{'|max|':>13}{'mean|.|':>12}")
    print("-" * 74)
    for i in range(len(model.blocks)):
        b = f"block{i}"
        out[b] = {}
        for key in ["polygelu_input", "attn_scores_input", "attn_output"]:
            if key in stats[b]:
                f = fin(stats[b][key])
                out[b][key] = f
                print(f"{b}.{key:<14}{f['min']:>13.4g}{f['max']:>13.4g}"
                      f"{f['absmax']:>13.4g}{f['mean_abs']:>12.4g}")
        if "row_sum" in rowsum_stats[b]:
            f = fin(rowsum_stats[b]["row_sum"])
            out[b]["row_sum"] = f
            print(f"{b}.{'row_sum':<14}{f['min']:>13.4g}{f['max']:>13.4g}"
                  f"{f['absmax']:>13.4g}{f['mean_abs']:>12.4g}")

    # The two numbers that decide the encrypted design
    gelu_absmax = max(out[b]["polygelu_input"]["absmax"] for b in out)
    score_absmax = max(out[b]["attn_scores_input"]["absmax"] for b in out
                       if "attn_scores_input" in out[b])
    rs = [out[b]["row_sum"] for b in out if "row_sum" in out[b]]
    rs_min = min(r["min"] for r in rs) if rs else None
    rs_max = max(r["max"] for r in rs) if rs else None

    print("\n" + "=" * 74)
    print("WHAT THIS DETERMINES FOR THE ENCRYPTED CIRCUIT")
    print("=" * 74)
    print(f"  PolyGELU input |max|      : {gelu_absmax:.4g}")
    print(f"    -> fitted on [-3,3]; inputs {'EXCEED' if gelu_absmax > 3 else 'stay within'} "
          f"the fit range by {gelu_absmax/3:.1f}x" if gelu_absmax > 3 else
          f"    -> within the [-3,3] fit range")
    print(f"  Attention score |max|     : {score_absmax:.4g}")
    if rs_min is not None:
        print(f"  Row-sum range             : [{rs_min:.4g}, {rs_max:.4g}]")
        print(f"    -> Goldschmidt must invert over this range. Convergence needs")
        print(f"       the input scaled into (0,2); required scaling factor ~1/{rs_max:.3g}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": args.config, "attn_type": attn_type,
            "weights": "RANDOM INIT (no local checkpoint) -- ranges indicate scale, "
                       "not a trained model's exact distribution",
            "batches": args.batches, "batch_size": args.batch_size,
            "per_block": out,
            "summary": {"polygelu_input_absmax": gelu_absmax,
                        "attn_score_absmax": score_absmax,
                        "row_sum_min": rs_min, "row_sum_max": rs_max},
        }, f, indent=2)
    print(f"\nJSON: {args.out}")


if __name__ == "__main__":
    main()
