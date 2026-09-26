"""
Stage 2 of 2 — CKKS classification head under OpenFHE.
======================================================================

Runs in WSL (Ubuntu 24.04, CPython 3.12). Depends on `openfhe` and the
standard library ONLY -- no torch, no numpy, no tenseal. Consumes the JSON
fixture written by export_head_fixture.py.

Circuit (identical to the TenSEAL path in ../ckks_classification_head.py):

    logit_i = < Enc(cls), W[i] > + b[i]      for i in 0..9

  Enc(cls)               one ciphertext, 192 real slots, zero-padded to 256
  < . , W[i] >           EvalInnerProduct(ct, pt, batchSize)
                           = one ciphertext x plaintext multiply  (1 level)
                             followed by EvalSum's log2(256) = 8 rotations
  + b[i]                 EvalAdd(ct, scalar) -- free, no level consumed

So the whole head is depth 1. That is why SetMultiplicativeDepth(1) suffices
and mirrors TenSEAL's coeff_mod_bit_sizes=[60, 40, 60].

What OpenFHE makes explicit that TenSEAL hid:
  - Enable(...) capability layers, chosen per operation
  - EvalMultKeyGen  -> the relinearization key
  - EvalSumKeyGen   -> the Galois/rotation keys EvalSum needs
  - the ring dimension is CHOSEN BY THE LIBRARY from the security level,
    rather than asserted by hand. We report what it picked.

Usage (from Windows):
  wsl -d Ubuntu-24.04 -- python3 openfhe_head.py --fixture head_fixture.json
"""

import argparse
import json
import time
from pathlib import Path

from openfhe import (CCParamsCKKSRNS, GenCryptoContext, SecurityLevel,
                     PKESchemeFeature, ScalingTechnique)

# Scaling technique drives the ring dimension, and therefore the runtime.
# Measured on this circuit (depth 1, first=60, scale=40, 128-bit classical):
#
#   FLEXIBLEAUTOEXT  (OpenFHE's DEFAULT)  -> N = 16384    ~1550 ms/sample
#   FLEXIBLEAUTO / FIXEDAUTO / FIXEDMANUAL -> N =  8192
#
# FLEXIBLEAUTOEXT appends an extra modulus to the chain, which pushes the
# total past what N=8192 supports at 128-bit -- OpenFHE then refuses 8192
# outright ("does not comply with HE standards recommendation (16384)").
# That is a property of the default, NOT evidence that N=8192 is insecure:
# with FIXEDAUTO, OpenFHE itself accepts 8192, agreeing with the SEAL/TenSEAL
# configuration coeff_mod_bit_sizes=[60, 40, 60].
#
# FIXEDAUTO is the closest analogue to SEAL/TenSEAL's fixed-scale RNS, so it
# is the default here for an apples-to-apples comparison.
SCALING = {
    "fixedauto": ScalingTechnique.FIXEDAUTO,
    "flexibleauto": ScalingTechnique.FLEXIBLEAUTO,
    "flexibleautoext": ScalingTechnique.FLEXIBLEAUTOEXT,
}


def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def build_context(depth, scale_bits, first_mod_bits, batch_size, ring_dim,
                  scaling="fixedauto"):
    """CKKS context mirroring the TenSEAL reference, but security-driven."""
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(depth)
    params.SetScalingModSize(scale_bits)
    params.SetFirstModSize(first_mod_bits)
    params.SetBatchSize(batch_size)
    params.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
    params.SetScalingTechnique(SCALING[scaling])
    if ring_dim:
        params.SetRingDim(ring_dim)

    cc = GenCryptoContext(params)
    # Capability layers: PKE = encrypt/decrypt, KEYSWITCH + LEVELEDSHE =
    # relinearization and rescaling, ADVANCEDSHE = EvalSum / EvalInnerProduct.
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)
    return cc


def describe(cc, depth, scale_bits, first_mod_bits, batch_size, scaling):
    """The CKKS parameter table. Doubles as viva material for atoms PP5.5/PP5.11."""
    def safe(fn, *variants):
        """Try each argument tuple; pybind overloads differ across builds."""
        for args in (variants or [()]):
            try:
                return fn(*args)
            except Exception:
                continue
        return "(unavailable)"

    n = cc.GetRingDimension()
    tech = str(safe(cc.GetScalingTechnique)).split(".")[-1]
    rows = [
        ("multiplicative depth (requested)", depth),
        ("scaling mod size   (log2 delta)", "%d bits" % scale_bits),
        ("first mod size     (log2 q0)", "%d bits" % first_mod_bits),
        ("batch size (slots summed by EvalSum)", batch_size),
        ("security level", "HEStd_128_classic"),
        ("scaling technique (requested)", scaling),
        ("scaling technique (effective)", tech),
        ("ring dimension N (chosen by OpenFHE)", n),
        ("available slots (N/2)", n // 2),
        ("scaling factor delta", safe(cc.GetScalingFactorReal, (), (0,))),
        ("ciphertext modulus q", safe(cc.GetModulus)),
    ]
    w = 42
    print("+" + "-" * (w + 26) + "+")
    print("| CKKS PARAMETERS" + " " * (w + 10) + "|")
    print("+" + "-" * (w + 26) + "+")
    for k, v in rows:
        print("|  %-*s %-22s|" % (w, k, str(v)[:22]))
    print("+" + "-" * (w + 26) + "+")
    return {k: str(v) for k, v in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="head_fixture.json")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all in fixture")
    ap.add_argument("--depth", type=int, default=0,
                    help="0 = auto: 1 for perrow, 2 for packed (EvalSumCols "
                         "spends a level on its internal row mask)")
    ap.add_argument("--scale-bits", type=int, default=40)
    ap.add_argument("--first-mod-bits", type=int, default=60)
    ap.add_argument("--ring-dim", type=int, default=0,
                    help="0 = let OpenFHE choose from the security level; "
                         "8192 forces parity with the TenSEAL reference")
    ap.add_argument("--scaling", default="fixedauto", choices=sorted(SCALING),
                    help="fixedauto (default, closest to SEAL/TenSEAL); "
                         "flexibleautoext is OpenFHE's own default and forces "
                         "N=16384 on this circuit")
    ap.add_argument("--method", default="packed", choices=["perrow", "packed"],
                    help="perrow: 10 x EvalInnerProduct (mirrors TenSEAL .dot); "
                         "packed: 1 x EvalMult + 1 x EvalSumCols (default)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.out:
        args.out = "openfhe_head_results_%s.json" % args.method
    if not args.depth:
        # EvalSumCols multiplies by a mask internally to stop rows bleeding into
        # each other, so it consumes a multiplicative level on top of the
        # ct*pt product. EvalSum/EvalInnerProduct are rotation-only and don't.
        # Verified: at depth 1 the packed path decrypts to "approximation error
        # is too high"; at depth 2 it is exact. Depth 2 forces N=16384.
        args.depth = 2 if args.method == "packed" else 1

    here = Path(__file__).resolve().parent
    fx = json.loads((here / args.fixture).read_text())
    meta = fx["meta"]
    W, b = fx["W"], fx["b"]
    cls_all, plain_all = fx["cls"], fx["plain_logits"]
    ts_all = fx.get("tenseal_logits")

    n = len(cls_all)
    if args.max_samples:
        n = min(n, args.max_samples)
    dim, n_classes = meta["embed_dim"], meta["num_classes"]

    # perrow: one row per ciphertext, batch = padded feature dim.
    # packed: the whole 10 x 256 matrix in ONE ciphertext, so the batch has to
    #         cover n_classes * row_size (2560 -> next power of two = 4096).
    row_size = next_pow2(dim)
    batch = row_size if args.method == "perrow" \
        else next_pow2(row_size * n_classes)

    print("fixture      : %s" % args.fixture)
    print("  weights    : %s" % meta["weights"])
    print("  cls source : %s" % meta["cls_source"])
    print("  method     : %s" % args.method)
    print("  samples    : %d of %d   dim=%d  classes=%d  row_size=%d  batch=%d"
          % (n, len(cls_all), dim, n_classes, row_size, batch))
    print()

    cc = build_context(args.depth, args.scale_bits, args.first_mod_bits,
                       batch, args.ring_dim, args.scaling)
    params_table = describe(cc, args.depth, args.scale_bits,
                            args.first_mod_bits, batch, args.scaling)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)   # relinearization key

    # Weight rows are public in this threat model (server holds the model,
    # client's CLS token is the secret), so they stay plaintext.
    col_keys = None
    if args.method == "perrow":
        cc.EvalSumKeyGen(keys.secretKey)    # Galois keys for EvalSum's rotations
        print("\nkeys generated (public/secret, relin, sum-rotation)\n")
        row_pad = [0.0] * (row_size - dim)
        W_pt = [cc.MakeCKKSPackedPlaintext(list(r) + row_pad) for r in W]
        bias_pt = None
    else:
        # EvalSumCols collapses each row of a row-major packed matrix. Despite
        # the name it sums ACROSS columns, i.e. within a row -- verified
        # empirically on a 2x4 matrix before this was written.
        # EvalSumRows is the wrong call here: it needs numRows to be a power of
        # two, and 10 classes is not.
        col_keys = cc.EvalSumColsKeyGen(keys.secretKey)
        print("\nkeys generated (public/secret, relin, sum-cols)\n")

        # The bias is folded into the weight matrix rather than added after the
        # sum. row_size (256) exceeds dim (192), so slot `dim` of every row is
        # free: put b[i] there in the weights and 1.0 there in the replicated
        # CLS token, and the dot product carries the bias itself.
        #
        # This is not just tidier. EvalAdd(ct_sum, bias_plaintext) fails here --
        # ct_sum is at level 1 after the multiply while a fresh plaintext is at
        # level 0, and reconciling them needs a second multiplicative level
        # ("The current multiplicative depth [1] is insufficient"). Folding the
        # bias in keeps the whole head at depth 1.
        flat = []
        for r in W:
            row = list(r) + [0.0] * (row_size - dim)
            row[dim] = float(b[len(flat) // row_size])
            flat.extend(row)
        flat.extend([0.0] * (batch - len(flat)))
        W_pt = cc.MakeCKKSPackedPlaintext(flat)
        bias_pt = None

    match_plain = 0
    match_ts = 0
    max_err = 0.0
    sum_err = 0.0
    errs_ts = []
    times = []

    for s in range(n):
        cls = list(cls_all[s])
        t0 = time.perf_counter()

        if args.method == "perrow":
            # 10 x (ct*pt multiply + EvalSum's 8 rotations) + 10 decryptions.
            ct = cc.Encrypt(keys.publicKey,
                            cc.MakeCKKSPackedPlaintext(cls + [0.0] * (row_size - dim)))
            logits = []
            for i in range(n_classes):
                ct_dot = cc.EvalInnerProduct(ct, W_pt[i], batch)
                ct_logit = cc.EvalAdd(ct_dot, float(b[i]))
                out = cc.Decrypt(keys.secretKey, ct_logit)
                out.SetLength(1)
                logits.append(out.GetRealPackedValue()[0])
        else:
            # 1 multiply + 8 rotations + 1 decryption, whatever n_classes is.
            # The CLS token is replicated once per class so that each row of the
            # packed matrix meets its own copy of the feature vector; slot `dim`
            # holds 1.0 so the folded bias term is picked up by the same sum.
            block = cls + [0.0] * (row_size - dim)
            block[dim] = 1.0
            rep = block * n_classes
            rep.extend([0.0] * (batch - len(rep)))
            ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(rep))

            ct_prod = cc.EvalMult(ct, W_pt)
            ct_sum = cc.EvalSumCols(ct_prod, row_size, col_keys)

            out = cc.Decrypt(keys.secretKey, ct_sum)
            out.SetLength(batch)
            slots = out.GetRealPackedValue()
            logits = [slots[i * row_size] for i in range(n_classes)]

        times.append(time.perf_counter() - t0)

        ref = plain_all[s]
        errs = [abs(logits[i] - ref[i]) for i in range(n_classes)]
        max_err = max(max_err, max(errs))
        sum_err += max(errs)

        if max(range(n_classes), key=lambda i: logits[i]) == \
           max(range(n_classes), key=lambda i: ref[i]):
            match_plain += 1

        if ts_all is not None:
            tref = ts_all[s]
            errs_ts.append(max(abs(logits[i] - tref[i]) for i in range(n_classes)))
            if max(range(n_classes), key=lambda i: logits[i]) == \
               max(range(n_classes), key=lambda i: tref[i]):
                match_ts += 1

        if s == 0 or (s + 1) % 10 == 0 or s == n - 1:
            print("  sample %4d/%d   argmax match %d/%d   max|err| %.3e   %.0f ms"
                  % (s + 1, n, match_plain, s + 1, max(errs), times[-1] * 1000))

    mean_ms = sum(times) / len(times) * 1000.0

    print("\n" + "=" * 64)
    print("RESULT -- OpenFHE CKKS vs plaintext")
    print("=" * 64)
    print("  samples                 : %d" % n)
    print("  argmax match            : %d/%d (%.2f%%)"
          % (match_plain, n, 100.0 * match_plain / n))
    print("  max  |logit error|      : %.3e" % max_err)
    print("  mean |logit error|      : %.3e" % (sum_err / n))
    print("  mean time per sample    : %.1f ms" % mean_ms)

    if ts_all is not None:
        print("\n  --- vs TenSEAL on the same CLS tokens ---")
        print("  argmax match            : %d/%d (%.2f%%)"
              % (match_ts, n, 100.0 * match_ts / n))
        print("  max |OpenFHE - TenSEAL| : %.3e" % max(errs_ts))
        if "tenseal_mean_ms" in meta:
            print("  TenSEAL mean time       : %.1f ms  (OpenFHE %.1f ms)"
                  % (meta["tenseal_mean_ms"], mean_ms))

    ref = meta.get("tenseal_reference", {})
    print("\n  reference run (10k samples, ../experiment_results/ckks_*.json):")
    print("    100%% match, max err 1.14e-05, 175.3 ms/sample, N=%s, scale 2^%s"
          % (ref.get("poly_modulus_degree"), ref.get("scale_bits")))

    results = {
        "backend": "openfhe-python",
        "method": args.method,
        "row_size": row_size,
        "batch": batch,
        "n_samples": n,
        "argmax_match_vs_plaintext": match_plain,
        "match_rate_percent": round(100.0 * match_plain / n, 4),
        "max_logit_error": float("%.3e" % max_err),
        "mean_logit_error": float("%.3e" % (sum_err / n)),
        "mean_time_ms": round(mean_ms, 2),
        "ckks_params": params_table,
        "fixture_meta": meta,
    }
    if ts_all is not None:
        results["argmax_match_vs_tenseal"] = match_ts
        results["max_abs_diff_vs_tenseal"] = float("%.3e" % max(errs_ts))

    (here / args.out).write_text(json.dumps(results, indent=2))
    print("\nwrote %s" % (here / args.out))


if __name__ == "__main__":
    main()
