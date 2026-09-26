"""
measure_encrypted_attention.py
===============================
The two ciphertext-ciphertext matmuls of a real attention head, under CKKS.

WHY THIS EXISTS (audit gap 1, second half)
-------------------------------------------
measure_encrypted_primitives.py closed the ELEMENTWISE half: poly-GELU and
poly-attention evaluated inside real ciphertexts at depth 2 each. What it
did not touch is the part that makes attention expensive and that no script
in this repo had ever attempted:

    scores = Q @ K^T      both operands ENCRYPTED
    out    = attn @ V     both operands ENCRYPTED

Every other linear op in the block is ciphertext-times-PLAINTEXT (the
weights are public), which is cheap and already measured. These two are
ct-ct, and they are the reason a transformer block is harder to encrypt
than a classification head.

WHAT IS MEASURED
----------------
One attention head of the real model (d_head = 192/3 = 64, n_tokens = 65),
on real Q/K/V captured from a real forward pass:
  1. Q @ K^T          -> 65x65 scores, 65*65 = 4225 ct-ct dot products
  2. even-power attn  -> the Gate-1 winner, elementwise, depth 2
  3. attn @ V         -> 65x64 output, 65*64 = 4160 ct-ct dot products
against the plaintext reference, reporting error and measured level use.

The three stages are measured independently.  Each stage's output is decrypted
and the next stage is encrypted again at a fresh modulus.  Therefore this script
does NOT measure a continuously encrypted attention head: 1 + 2 + 1 is a sum of
independent stage depths, not a measured four-level circuit, and the final error
does not include noise accumulated through all three stages.

HONEST SCOPE
------------
This is the STRAIGHTFORWARD packing: one ciphertext per row, one ct-ct dot
per output element. It is deliberately not the efficient construction --
production frameworks use diagonal/BSGS packing to amortise rotations, and
the SoK's L-COL/L-CYC classes exist precisely for that. Measuring the naive
form first gives an honest upper bound on cost and a correctness baseline
that a faster implementation must match. Reported as such, not as a
competitive latency number.

Row normalisation is EXCLUDED here and measured separately
(measure_encrypted_primitives.py: Goldschmidt, 13 levels) because at
65 tokens it dominates wall-clock and its cost is already known.

Weights random-init (no local checkpoint), same disclosure as elsewhere:
this measures numerical fidelity and level consumption, not accuracy.

Usage: python experiments/active/measure_encrypted_attention.py [--tokens 16] [--full]
"""

import argparse
import math
import time

import numpy as np
import torch

from fix2_bloodmnist import DeiTTiny
from tools.result_contracts import atomic_write_json

try:
    import tenseal as ts
except ImportError:
    print("TenSEAL not found. Install with: pip install tenseal")
    raise SystemExit(1)


def make_ctx(levels, scale_bits=40, N=32768):
    chain = [60] + [scale_bits] * levels + [60]
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=N,
                     coeff_mod_bit_sizes=chain)
    ctx.global_scale = 2 ** scale_bits
    ctx.generate_galois_keys()
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16,
                    help="tokens to encrypt (65 = full sequence; smaller is a "
                         "cost-model probe, since cost is O(n^2) dot products)")
    ap.add_argument("--full", action="store_true",
                    help="use all 65 tokens (slow: ~8400 ct-ct dots)")
    ap.add_argument("--out", default="results/ablations/encrypted_attention.json")
    args = ap.parse_args()
    n_tok = 65 if args.full else args.tokens

    torch.manual_seed(0)

    # ---- real Q, K, V from a real forward pass ----
    model = DeiTTiny(img_size=32, patch_size=4, in_channels=3, num_classes=8,
                     embed_dim=192, depth=6, num_heads=3, norm_type="batchnorm",
                     attn_type="poly_normed", gelu_type="poly")
    model.train()
    with torch.no_grad():
        for _ in range(3):
            model(torch.randn(16, 3, 32, 32) * 2.0 + 0.5)
    model.eval()

    blk = model.blocks[0]
    with torch.no_grad():
        x = torch.randn(1, 3, 32, 32) * 2.0 + 0.5
        h = model.patch_embed(x).flatten(2).transpose(1, 2)
        cls = model.cls_token.expand(1, -1, -1)
        h = torch.cat([cls, h], dim=1) + model.pos_embed
        hn = blk.norm1(h)
        B, N, D = hn.shape
        qkv = blk.qkv(hn).reshape(B, N, 3, blk.num_heads, blk.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0, 0, 0], qkv[1, 0, 0], qkv[2, 0, 0]   # head 0

    Q = q[:n_tok].double().numpy()
    K = k[:n_tok].double().numpy()
    V = v[:n_tok].double().numpy()
    scale = blk.scale
    d_head = Q.shape[1]

    print("=" * 76)
    print("ENCRYPTED ATTENTION -- the two ct-ct matmuls of a real head")
    print("=" * 76)
    print(f"  tokens {n_tok}  d_head {d_head}  (full sequence is 65)")
    print(f"  Q range [{Q.min():.3f}, {Q.max():.3f}]   V range [{V.min():.3f}, {V.max():.3f}]")
    print(f"  ct-ct dot products: {n_tok*n_tok} (QK^T) + {n_tok*d_head} (attn@V)"
          f" = {n_tok*n_tok + n_tok*d_head}")
    print("  Weights RANDOM INIT -- numerical fidelity and levels, not accuracy.")

    at = blk.attn_act
    a_a, b_a, c_a = float(at.a), float(at.b), float(at.c)
    a_sq = math.sqrt(max(a_a, 1e-9))

    # ---- plaintext reference: the exact circuit we will encrypt ----
    scores_ref = (Q @ K.T) * scale
    inner = a_sq * scores_ref + b_a
    attn_ref = inner ** 2 + c_a ** 2          # even-power, no ReLU, depth 2
    out_ref = attn_ref @ V

    # Allocate enough room for any one independently encrypted stage plus
    # headroom. The degree-2 stage needs 2 levels, not 1 -- allocating 1 raises
    # "scale out of bounds". This allocation is not a continuous-depth claim.
    ctx = make_ctx(levels=5)

    # ---- 1. Q @ K^T, ct-ct ----
    print("\n" + "-" * 76)
    print("1. Q @ K^T   (ciphertext x ciphertext)")
    print("-" * 76)
    t0 = time.time()
    encQ = [ts.ckks_vector(ctx, Q[i].tolist()) for i in range(n_tok)]
    encK = [ts.ckks_vector(ctx, K[j].tolist()) for j in range(n_tok)]
    t_enc = time.time() - t0

    t0 = time.time()
    scores_enc = np.zeros((n_tok, n_tok))
    for i in range(n_tok):
        for j in range(n_tok):
            scores_enc[i, j] = encQ[i].dot(encK[j]).decrypt()[0] * scale
    t_qk = time.time() - t0
    e_qk = np.abs(scores_enc - scores_ref)
    print(f"  max abs err {e_qk.max():.3e}   mean {e_qk.mean():.3e}")
    print(f"  {t_qk:.1f}s for {n_tok*n_tok} dots "
          f"({t_qk/(n_tok*n_tok)*1000:.2f} ms/dot)   depth 1")

    # ---- 2. even-power attention, elementwise ----
    print("\n" + "-" * 76)
    print("2. even-power attention  (a s + b)^2 + c^2   -- Gate 1 winner (2 levels)")
    print("-" * 76)
    t0 = time.time()
    enc_rows = [ts.ckks_vector(ctx, scores_enc[i].tolist()) for i in range(n_tok)]
    attn_enc = np.zeros((n_tok, n_tok))
    for i in range(n_tok):
        inner_c = enc_rows[i] * a_sq + b_a
        attn_enc[i] = np.array((inner_c * inner_c + c_a ** 2).decrypt())
    t_act = time.time() - t0
    e_act = np.abs(attn_enc - attn_ref)
    print(f"  max abs err {e_act.max():.3e}   mean {e_act.mean():.3e}")
    print(f"  {t_act:.1f}s   depth 2 (measured minimum)")

    # ---- 3. attn @ V, ct-ct ----
    print("\n" + "-" * 76)
    print("3. attn @ V   (ciphertext x ciphertext)")
    print("-" * 76)
    t0 = time.time()
    encA = [ts.ckks_vector(ctx, attn_enc[i].tolist()) for i in range(n_tok)]
    Vt = V.T.copy()
    encVt = [ts.ckks_vector(ctx, Vt[c].tolist()) for c in range(d_head)]
    out_enc = np.zeros((n_tok, d_head))
    for i in range(n_tok):
        for c in range(d_head):
            out_enc[i, c] = encA[i].dot(encVt[c]).decrypt()[0]
    t_av = time.time() - t0
    e_av = np.abs(out_enc - out_ref)
    rel = e_av / (np.abs(out_ref) + 1e-12)
    print(f"  max abs err {e_av.max():.3e}   mean {e_av.mean():.3e}")
    print(f"  max rel err {rel.max():.3e}")
    print(f"  {t_av:.1f}s for {n_tok*d_head} dots "
          f"({t_av/(n_tok*d_head)*1000:.2f} ms/dot)   depth 1")

    total_s = t_enc + t_qk + t_act + t_av
    print("\n" + "=" * 76)
    print("RESULT")
    print("=" * 76)
    print(f"  final independently measured stage error: max {e_av.max():.3e}, "
          f"max relative {rel.max():.3e}")
    print("  independent stage depths: QK^T 1, activation 2, attn@V 1")
    print("  continuous encrypted depth: NOT MEASURED (decrypt/re-encrypt boundaries)")
    print(f"  wall clock: {total_s:.1f}s for ONE head at {n_tok} tokens")
    if n_tok < 65:
        # cost is O(n^2) in the dot count
        f = (65 * 65 + 65 * d_head) / (n_tok * n_tok + n_tok * d_head)
        print(f"  extrapolated to 65 tokens: ~{total_s*f:.0f}s per head "
              f"({total_s*f*3/60:.1f} min for 3 heads, "
              f"{total_s*f*3*6/3600:.1f} h for all 6 blocks)")
        print("  -- naive packing, one ct-ct dot per output element. An efficient")
        print("     implementation would amortise this; treat as an upper bound.")

    atomic_write_json(args.out, {
        "purpose": "measure the two ciphertext-ciphertext matmuls of a real "
                   "attention head under CKKS (audit gap 1, second half)",
        "scope": {
            "continuous_ciphertext": False,
            "decrypt_reencrypt_between_stages": True,
            "claim": "component-wise numerical fidelity and cost baseline; "
                     "not end-to-end encrypted attention",
        },
        "packing": "naive: one ciphertext per row, one ct-ct dot per output "
                   "element. Deliberately not the efficient construction; "
                   "an upper bound and a correctness baseline.",
        "weights": "RANDOM INIT (no local checkpoint) -- numerical fidelity "
                   "and level consumption only, not accuracy",
        "excluded": "row normalisation (Goldschmidt, 13 levels) measured "
                    "separately in encrypted_primitives.json",
        "tokens": n_tok, "d_head": d_head, "heads_in_model": 3,
        "blocks_in_model": 6,
        "ckks": {"N": 32768, "scale_bits": 40, "levels_allocated": 5,
                 "independent_stage_levels": {"qk": 1, "activation": 2,
                                              "attn_v": 1},
                 "independent_stage_level_sum": 4,
                 "continuous_levels_used": None,
                 "security": "128-bit"},
        "qk": {"max_abs_err": float(e_qk.max()), "mean_abs_err": float(e_qk.mean()),
               "seconds": t_qk, "n_dots": n_tok * n_tok, "depth": 1},
        "activation": {"max_abs_err": float(e_act.max()),
                       "mean_abs_err": float(e_act.mean()),
                       "seconds": t_act, "depth": 2},
        "av": {"max_abs_err": float(e_av.max()), "mean_abs_err": float(e_av.mean()),
               "max_rel_err": float(rel.max()), "seconds": t_av,
               "n_dots": n_tok * d_head, "depth": 1},
        "sum_of_independent_stage_seconds": total_s,
    })
    print(f"\nJSON: {args.out}")


if __name__ == "__main__":
    main()
