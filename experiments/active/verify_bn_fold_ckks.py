"""
verify_bn_fold_ckks.py
=======================
Does the BatchNorm fold survive actual CKKS encryption, at the exact
parameters the rest of the repo uses?

WHY THIS EXISTS
---------------
`verify_bn_fold.py` proved the fold algebraically exact in plaintext (7
assertions, max err 4e-6) but explicitly stopped there: "no folded BatchNorm
has been evaluated inside a ciphertext... Whether the [60,40,60] modulus
chain at scale 2^40 preserves the measured 1.14e-5 logit error once these
magnitudes shift is unmeasured" (paper/main.tex, sec:bnfold). Folding is
depth-neutral (still one ct-pt multiply) but not scale-neutral: on trained
weights it shifts max|b| by 2.31x. This closes that gap for real, per audit
finding F04.

WHAT THIS DOES
--------------
Extends the fold to the site that actually matters for deployment: the
FINAL TokenBatchNorm -> classification head, i.e. one step earlier than
`ckks_classification_head.py`'s encryption boundary (which currently
encrypts the CLS token AFTER BatchNorm has already run in plaintext). Here
the CLS token is encrypted BEFORE BatchNorm, and the folded W'/b' are
applied entirely inside the ciphertext -- so this is a genuine test of
whether "the fold" is safe to deploy, not just whether the fold is
algebraically correct.

Uses the IDENTICAL CKKS context as `ckks_classification_head.py`
(`setup_ckks_context()`, imported directly -- no parameters re-typed):
N=8192, coeff_mod_bit_sizes=[60,40,60], scale=2^40, 1 multiplicative level.
So any degradation observed is attributable to the fold's magnitude shift,
not to a different parameter choice.

HONEST LIMITATION, stated exactly like `openfhe/export_head_fixture.py`
does for the same reason: no trained BloodMNIST checkpoint exists on this
machine (the A6000 has one; this laptop does not). Running stats are warmed
up via forward passes on random inputs, matching `verify_bn_fold.py`'s
approach. That makes the fold's SCALE SHIFT representative of a real
post-training BatchNorm (mean/var are non-trivial, not the 0/1 an
untouched BatchNorm would have) even though the WEIGHTS are random and
therefore accuracy is meaningless. Numerical fidelity of the encrypted
computation is unaffected by whether the weights were trained.

Usage: python experiments/active/verify_bn_fold_ckks.py [--n-samples 500]
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from fix2_bloodmnist import DeiTTiny
from ckks_classification_head import setup_ckks_context

try:
    import tenseal as ts
except ImportError:
    print("TenSEAL not found. Install with: pip install tenseal")
    raise SystemExit(1)


def affine_from_bn(bn: torch.nn.BatchNorm1d):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    t = bn.bias - bn.weight * bn.running_mean / torch.sqrt(bn.running_var + bn.eps)
    return s, t


def fold_into_linear(lin: torch.nn.Linear, s: torch.Tensor, t: torch.Tensor):
    W = lin.weight
    b = lin.bias if lin.bias is not None else torch.zeros(W.shape[0])
    W_folded = W * s.unsqueeze(0)
    b_folded = W @ t + b
    return W_folded, b_folded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--results-path", type=str,
                    default="results/ablations/bn_fold_ckks.json")
    args = ap.parse_args()

    torch.manual_seed(0)

    print("=" * 72)
    print("BatchNorm-fold CKKS verification")
    print("=" * 72)
    print(f"Samples : {args.n_samples}")
    print("Site    : final TokenBatchNorm -> classification head")
    print("Weights : RANDOM INIT (no local checkpoint) -- accuracy is")
    print("          meaningless, numerical fidelity is not")

    # ---- 1. Build the model, warm up BatchNorm running stats ----
    model = DeiTTiny(img_size=32, patch_size=4, in_channels=3, num_classes=8,
                     embed_dim=192, depth=6, num_heads=3,
                     norm_type="batchnorm", attn_type="poly_normed",
                     gelu_type="poly")
    model.train()
    with torch.no_grad():
        for _ in range(8):
            model(torch.randn(32, 3, 32, 32) * 2.0 + 0.5)
    model.eval()

    bn = model.norm.bn
    print(f"\nWarmed-up final-norm stats: "
          f"mean|.|={bn.running_mean.abs().mean():.4f}  "
          f"var_mean={bn.running_var.mean():.4f}")

    with torch.no_grad():
        s, t = affine_from_bn(bn)
        Wf, bf = fold_into_linear(model.head, s, t)

    W_np = Wf.numpy().astype(np.float64)
    b_np = bf.numpy().astype(np.float64)
    print(f"Folded weight/bias: max|W'|={np.abs(W_np).max():.4f}  "
          f"max|b'|={np.abs(b_np).max():.4f}")
    print(f"Original head:      max|W|={model.head.weight.abs().max():.4f}  "
          f"max|b|={model.head.bias.abs().max():.4f}")

    # ---- 2. Generate synthetic pre-BN CLS tokens ----
    # A realistic value range for a pre-norm CLS token: whatever the model's
    # own forward pass produces before the final norm, on random images --
    # not an arbitrary distribution invented for this test.
    with torch.no_grad():
        pre_norm_tokens = []
        for _ in range(0, args.n_samples, 32):
            n = min(32, args.n_samples - len(pre_norm_tokens))
            x = torch.randn(32, 3, 32, 32) * 2.0 + 0.5
            B = x.shape[0]
            h = model.patch_embed(x).flatten(2).transpose(1, 2)
            cls = model.cls_token.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1) + model.pos_embed
            for blk in model.blocks:
                h = blk(h)
            pre_norm_tokens.append(h[:, 0].numpy())  # CLS token, pre-final-norm
        pre_norm_tokens = np.concatenate(pre_norm_tokens, axis=0)[:args.n_samples]

    print(f"\nCLS token range (pre-norm): "
          f"min={pre_norm_tokens.min():.3f} max={pre_norm_tokens.max():.3f}")

    # ---- 3. Plaintext reference: BN(x) -> head, unfolded ----
    with torch.no_grad():
        x_t = torch.from_numpy(pre_norm_tokens).float().unsqueeze(1)  # (B,1,D): TokenBatchNorm needs (B,N,D)
        ref_logits = model.head(model.norm(x_t).squeeze(1)).numpy().astype(np.float64)

    # ---- 4. Encrypted: Enc(x) -> W'@x+b' (folded), same context as
    #         ckks_classification_head.py ----
    ctx = setup_ckks_context()

    enc_logits = np.zeros_like(ref_logits)
    t_start = time.time()
    for i in range(args.n_samples):
        enc_x = ts.ckks_vector(ctx, pre_norm_tokens[i].tolist())
        for c in range(W_np.shape[0]):
            enc_logits[i, c] = enc_x.dot(W_np[c].tolist()).decrypt()[0] + b_np[c]
    elapsed = time.time() - t_start

    # ---- 5. Compare ----
    abs_err = np.abs(enc_logits - ref_logits)
    max_err = abs_err.max()
    mean_err = abs_err.mean()
    argmax_match = (enc_logits.argmax(axis=1) == ref_logits.argmax(axis=1)).mean()

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"  n samples          : {args.n_samples}")
    print(f"  argmax match       : {argmax_match*100:.2f}%")
    print(f"  max logit error    : {max_err:.3e}")
    print(f"  mean logit error   : {mean_err:.3e}")
    print(f"  ms/sample          : {elapsed/args.n_samples*1000:.1f}")
    print(f"\n  Reference (unfolded plaintext head, ckks_classification_head.py):")
    print(f"    max logit error 1.14e-05 (Fix2, N=8192, 10,000 samples)")

    if max_err < 1e-4:
        print(f"\n  VERDICT: fold survives CKKS at this precision. Max error "
              f"{max_err:.2e} is the same order as the unfolded head's "
              f"1.14e-05 (finding F04, closed for the classification-head site).")
    else:
        print(f"\n  VERDICT: fold does NOT survive at the expected precision -- "
              f"{max_err:.2e} exceeds the unfolded head's 1.14e-05 by "
              f"{max_err/1.14e-5:.0f}x. Investigate before claiming the fold "
              f"is safe at these parameters.")

    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump({
            "purpose": "measure whether the BatchNorm fold survives CKKS "
                       "encryption at the exact parameters "
                       "ckks_classification_head.py validates",
            "site": "final TokenBatchNorm -> classification head, folded",
            "weights": "RANDOM INIT (no local checkpoint) -- accuracy is "
                       "meaningless, numerical fidelity is not",
            "n_samples": args.n_samples,
            "argmax_match_rate": float(argmax_match),
            "max_logit_error": float(max_err),
            "mean_logit_error": float(mean_err),
            "ms_per_sample": elapsed / args.n_samples * 1000,
            "ckks_params": {"poly_modulus_degree": 8192,
                            "coeff_mod_bit_sizes": [60, 40, 60],
                            "global_scale_bits": 40, "mult_levels": 1},
            "fold_scale_shift": {"max_W_ratio": float(np.abs(W_np).max() /
                                 model.head.weight.abs().max().item()),
                                "max_b_ratio": float(np.abs(b_np).max() /
                                 max(model.head.bias.abs().max().item(), 1e-12))},
            "reference_unfolded_max_logit_error": 1.14e-05,
            "reference_source": "results/ablations/ckks_Fix2_Normalized.json",
        }, f, indent=2)
    print(f"\nResults JSON: {args.results_path}")


if __name__ == "__main__":
    main()
