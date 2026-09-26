"""
verify_bn_fold.py
=================
Does the "BatchNorm folds into the next linear layer at zero multiplicative
depth" claim actually hold for THIS model?

WHY THIS EXISTS
---------------
`fix2_bloodmnist.py:171` (TokenBatchNorm) and the paper both assert that at
inference BatchNorm becomes a per-feature affine that folds into the adjacent
linear layer's weights, costing zero CKKS levels. Until now that was an
assertion with no code behind it anywhere in the repo: the encrypted path
starts at the classification head, which sits *after* the final norm, so no
BatchNorm has ever been inside a ciphertext (audit finding F04).

This script closes the cheap half of that gap. It verifies, in plaintext and
exactly, that:

  1. TokenBatchNorm in eval mode IS a per-feature affine  y = s*x + t,  with
         s = gamma / sqrt(running_var + eps)
         t = beta - gamma*running_mean / sqrt(running_var + eps)
  2. The composition  Linear(BatchNorm(x))  equals a single Linear with
         W' = W * s        (scale each input column)
         b' = W @ t + b
     i.e. the norm disappears into weights the server already holds.
  3. The fold is exact to floating-point tolerance on real tensors, at each
     of the three sites where this pattern occurs in the architecture:
     norm1 -> qkv, norm2 -> fc1, and final norm -> head.

WHAT THIS DOES *NOT* SHOW
-------------------------
That the folded layer behaves correctly under CKKS. The fold changes W and b,
which changes their magnitudes and therefore the scale/precision budget of the
encrypted multiply. That is a separate measurement and still outstanding. Zero
*multiplicative depth* is proven here; zero *cost* is not.

Runs on CPU in a few seconds. No training, no data download.

Usage:  python experiments/active/verify_bn_fold.py
"""

import sys

import torch
import torch.nn as nn

from fix2_bloodmnist import DeiTTiny, TokenBatchNorm

TOL = 1e-5


def affine_from_bn(bn: nn.BatchNorm1d):
    """Extract (s, t) such that bn(x) == s*x + t in eval mode."""
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    t = bn.bias - bn.weight * bn.running_mean / torch.sqrt(bn.running_var + bn.eps)
    return s, t


def fold_into_linear(lin: nn.Linear, s: torch.Tensor, t: torch.Tensor):
    """Return (W', b') for Linear(affine(x)) collapsed to a single Linear."""
    W = lin.weight                      # [out, in]
    b = lin.bias if lin.bias is not None else torch.zeros(W.shape[0])
    W_folded = W * s.unsqueeze(0)       # scale each input column
    b_folded = W @ t + b
    return W_folded, b_folded


def main():
    torch.manual_seed(0)
    checks = 0

    def check(label, cond, detail=""):
        nonlocal checks
        if not cond:
            print(f"  FAIL  {label}  {detail}")
            sys.exit(1)
        checks += 1
        print(f"  ok    {label}  {detail}")

    print("=" * 72)
    print("BatchNorm-fold verification (plaintext, exact)")
    print("=" * 72)

    # Build the real model and give the BN layers non-trivial running stats,
    # as a trained checkpoint would have. Random init leaves mean=0, var=1,
    # which would make the fold trivially true and prove nothing.
    model = DeiTTiny(img_size=32, patch_size=4, in_channels=3, num_classes=8,
                     embed_dim=192, depth=6, num_heads=3,
                     norm_type="batchnorm", attn_type="poly_normed",
                     gelu_type="poly")
    model.train()
    with torch.no_grad():
        for _ in range(5):
            model(torch.randn(16, 3, 32, 32) * 2.0 + 0.5)
    model.eval()

    bn0 = model.blocks[0].norm1.bn
    check("running stats are non-trivial after warm-up",
          not torch.allclose(bn0.running_mean, torch.zeros_like(bn0.running_mean),
                             atol=1e-3),
          f"mean|.|={bn0.running_mean.abs().mean():.4f} "
          f"var={bn0.running_var.mean():.4f}")

    # ---- 1. TokenBatchNorm in eval mode is exactly a per-feature affine ----
    x = torch.randn(4, 65, 192) * 1.7 - 0.3
    tbn = model.blocks[0].norm1
    with torch.no_grad():
        y_ref = tbn(x)
        s, t = affine_from_bn(tbn.bn)
        y_affine = x * s + t
    err = (y_ref - y_affine).abs().max().item()
    check("TokenBatchNorm(eval) == s*x + t", err < TOL, f"max err {err:.2e}")

    # ---- 2. Fold at each of the three sites in the architecture ----
    sites = [
        ("norm1 -> qkv", model.blocks[0].norm1, model.blocks[0].qkv),
        ("norm2 -> fc1", model.blocks[0].norm2, model.blocks[0].fc1),
        ("final norm -> head", model.norm, model.head),
    ]
    for label, norm, lin in sites:
        with torch.no_grad():
            s, t = affine_from_bn(norm.bn)
            ref = lin(norm(x))
            Wf, bf = fold_into_linear(lin, s, t)
            folded = torch.nn.functional.linear(x, Wf, bf)
        err = (ref - folded).abs().max().item()
        rel = err / max(ref.abs().max().item(), 1e-12)
        check(f"fold exact at {label}", err < TOL,
              f"max abs {err:.2e}, rel {rel:.2e}")

    # ---- 3. The fold costs no multiplication: it only rewrites W and b ----
    with torch.no_grad():
        s, t = affine_from_bn(model.norm.bn)
        Wf, bf = fold_into_linear(model.head, s, t)
    check("folded weight has the same shape as the original",
          Wf.shape == model.head.weight.shape, f"{tuple(Wf.shape)}")
    check("folded bias has the same shape as the original",
          bf.shape == model.head.bias.shape, f"{tuple(bf.shape)}")

    # ---- 4. Honest scale reporting: the fold changes magnitudes ----
    w_ratio = (Wf.abs().max() / model.head.weight.abs().max()).item()
    b_ratio = (bf.abs().max() / model.head.bias.abs().max().clamp(min=1e-12)).item()
    print()
    print("  Scale impact of folding (matters for CKKS precision, NOT depth):")
    print(f"    max|W| changes by  {w_ratio:8.4f}x")
    print(f"    max|b| changes by  {b_ratio:8.4f}x")
    print("    -> depth is unchanged (still one ct-pt multiply), but the")
    print("       encrypted operand magnitudes change. Whether the existing")
    print("       scale 2^40 / [60,40,60] chain still gives the same precision")
    print("       after folding is UNMEASURED and remains open.")

    print()
    print(f"{checks} assertions passed.")
    print()
    print("CONCLUSION: the zero-multiplicative-depth fold is algebraically exact")
    print("for this architecture at all three norm->linear sites. It has still")
    print("never been executed inside a ciphertext. Cite this as a plaintext")
    print("verification, not as an encrypted measurement.")


if __name__ == "__main__":
    main()
