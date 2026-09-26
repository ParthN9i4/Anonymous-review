# OpenFHE port — CKKS classification head

Makes true the claim in `CompPrep\admin\HANDOFF-2026-08-08.md:11` that the encrypted-
inference thread is "implemented in OpenFHE." Before this, the only encryption imports in
this repo were TenSEAL (`core/ckks_classification_head.py:44`, `archive/family_a_step_pipeline/step6_encrypted_inference.py:48`).

Built 2026-08-09. Circuit is identical to the TenSEAL path: `logit_i = <Enc(cls), W[i]> + b[i]`,
192-dim CLS token, 10 classes, multiplicative depth 1.

## Why two files

`openfhe-python` ships a Linux CPython-3.12 binary (`openfhe.cpython-312-x86_64-linux-gnu.so`)
inside a wheel mislabelled `py3-none-any`. pip therefore installs it happily on Windows
Python 3.13 and leaves an import that can never work:

```
ModuleNotFoundError: No module named 'openfhe.openfhe'
```

torch/tenseal live on the Windows side; openfhe works in WSL. So the work splits at a JSON
boundary — which is the right architecture anyway, since the crypto script then carries no
ML dependencies at all.

| | Windows py3.13 | WSL Ubuntu-24.04 py3.12 |
|---|---|---|
| `torch`, `tenseal`, `numpy` | yes | no |
| `openfhe` | broken | **yes** |

## Run

```bash
python openfhe\export_head_fixture.py --max-samples 50 --with-tenseal
```

```bash
wsl -d Ubuntu-24.04 -- python3 /mnt/c/Users/YOUR_USERNAME/Documents/Documents/PPML/Private-ViT-FHE/openfhe/openfhe_head.py --fixture head_fixture.json --max-samples 10
```

Stage 1 needs no OpenFHE; stage 2 needs nothing but `openfhe` and the stdlib.

## What is verified

Correctness — **100% argmax match in every configuration tested**, against both the
plaintext head and TenSEAL on identical CLS tokens. Full run: all 50 fixture samples,
FIXEDAUTO, N=8192 (`openfhe_head_results.json`, 2m27s wall):

| Check (n=50) | `perrow` | `packed` |
|---|---|---|
| OpenFHE vs plaintext (argmax) | **50/50 (100%)** | **50/50 (100%)** |
| OpenFHE vs TenSEAL (argmax) | **50/50 (100%)** | **50/50 (100%)** |
| max abs diff vs TenSEAL | 1.357e-06 | 1.986e-06 |
| max abs logit error | 8.757e-07 | 7.388e-07 |
| mean abs logit error | 1.388e-07 | 2.832e-07 |
| ms/sample | 2810 | **619** |
| wall clock, 50 samples | 2m27s | **32s** |

Also verified at smaller n: **8/8** at N=8192 under FIXEDAUTO
(`res_fixedauto.json`) and **8/8** at N=16384 under FLEXIBLEAUTOEXT
(`res_flexibleautoext.json`) — the N each technique selects on its own, not a forced
cross-pairing. No configuration produced a single argmax disagreement.

> Corrected 2026-08-31 (audit finding, `docs/design/P1B_REPRO_NOVELTY.md` #27). This previously
> claimed "10/10 and 8/8... at both N=8192 and N=16384, under FIXEDAUTO and
> FLEXIBLEAUTOEXT" — no n=10 result exists anywhere in this directory, and no run pairs
> FIXEDAUTO with N=16384 or FLEXIBLEAUTOEXT with N=8192 (each technique deterministically
> selects its own N; forcing the other pairing was never attempted).

The OpenFHE-vs-TenSEAL number is the clean one: both paths consumed the *same* fixture,
so it is a true apples-to-apples measurement of the two libraries against each other.

## The parameter finding (viva material — atoms PP5.5, PP5.11)

OpenFHE initially **refused to build the context** at the ring dimension TenSEAL was using:

```
RuntimeError: The specified ring dimension (8192) does not comply with
HE standards recommendation (16384).
```

Cause, established by sweeping the parameters:

| Configuration | N chosen |
|---|---|
| HYBRID key-switch + **FLEXIBLEAUTOEXT** (OpenFHE's defaults) | 16384 |
| HYBRID + FIXEDAUTO / FLEXIBLEAUTO / FIXEDMANUAL | 8192 |
| BV key-switch + FLEXIBLEAUTOEXT | 8192 |
| BV + FIXEDMANUAL | 4096 |

It is **FLEXIBLEAUTOEXT**, the default *scaling technique*, not the key-switching method.
FLEXIBLEAUTOEXT appends an extra modulus to the chain, pushing the total past what N=8192
supports at 128-bit classical security. Key-switching is a secondary term (BV carries no
auxiliary modulus P, hence the drop to 4096).

**This does not indicate a security problem in the TenSEAL configuration.** With FIXEDAUTO,
OpenFHE independently selects N=8192 at `HEStd_128_classic` — agreeing with SEAL/TenSEAL's
`coeff_mod_bit_sizes=[60, 40, 60]`. The measured scaling factor is exactly 2^40, matching
`global_scale`. The N=16384 demand was a property of one library's default, nothing more.

The transferable lesson, and the answer to "justify your CKKS parameters": **the ring
dimension is not a free choice — it is derived from the security level together with the
scaling technique and key-switching method.** TenSEAL lets you *assert* N; OpenFHE *derives*
it and refuses non-compliant values. Naming FLEXIBLEAUTOEXT vs FIXEDAUTO is the level of
detail that question is really probing.

`--scaling` defaults to `fixedauto` here precisely because it is the closest analogue to
SEAL/TenSEAL's fixed-scale RNS, which is what makes the comparison fair.

## The packed matrix-vector method (`--method packed`, default)

`perrow` mirrors TenSEAL's `.dot()`: one `EvalInnerProduct` per class, so 10 ct-pt
multiplies, 10 x log2(256) = 80 rotations, and 10 decryptions per sample.

`packed` puts the whole 10 x 256 weight matrix in **one** plaintext (row-major), replicates
the CLS token once per class, and does **one** multiply, **one** row-collapse, **one**
decryption — 8 rotations total, independent of the number of classes.

Three things had to be got right, none of them guessable from the docs:

**1. `EvalSumCols`, not `EvalSumRows`.** The names mislead. Verified on a 2x4 matrix
`[[1,2,3,4],[10,20,30,40]]` before writing anything:

```
EvalSumCols(ct, numCols=4) -> [10,10,10,10, 100,100,100,100]   <- row sums, replicated
EvalSumRows                -> fails in this configuration
```

"Cols" is the axis summed *over*, which collapses each row. `EvalSumRows` also demands
`numRows` be a power of two (8 works; 2, 3, 5, 10 all fail), so with 10 classes it was
never viable. `EvalSumCols` takes only the row width, so 10 rows is fine.

**2. `EvalSumCols` costs a multiplicative level.** It multiplies by an internal mask to stop
rows bleeding into one another. Isolated by decrypting after each step:

```
after Encrypt                     ok
after EvalMult(ct, pt)            ok
after EvalSumCols on the product  DECRYPT FAILED: approximation error too high
EvalSumCols without the multiply  [4.0, ...]  ok
```

So `EvalSumCols` needs depth 2 here, while `EvalSum`/`EvalInnerProduct` are rotation-only
and run at depth 1. **That is the distinction worth remembering:** summing *within blocks*
needs masking; summing the *whole* batch does not. Depth 2 forces N=16384.

**3. The bias is folded into the weight matrix.** `row_size` is 256 and `dim` is 192, so
slot 192 of each row is free: it holds `b[i]` in the weights and `1.0` in the replicated
CLS token, and the dot product carries the bias. This is not cosmetic —
`EvalAdd(ct_sum, bias_plaintext)` fails outright, because `ct_sum` is a level below a fresh
plaintext and reconciling them wants yet another level.

Net effect: **4.5x faster** (619 vs 2810 ms/sample), identical argmax agreement, error of
the same order. The win holds *despite* N doubling to 16384 — cutting 10 multiplies to 1
and 80 rotations to 8 more than pays for the costlier ring. The advantage should widen with
more classes, since the packed cost is flat in `n_classes` while `perrow` is linear.

## Timing — measured, but do not quote it yet

| Method | Scaling | depth | N | ms/sample | n |
|---|---|---|---|---|---|
| perrow | FLEXIBLEAUTOEXT | 1 | 16384 | 1741 | 8 (paired) |
| perrow | FIXEDAUTO | 1 | 8192 | 2567 | 8 (paired) |
| perrow | FIXEDAUTO | 1 | 8192 | 2810 | 50 (full) |
| **packed** | FIXEDAUTO | 2 | 16384 | **619** | **50 (full)** |

The smaller ring is *slower*, and that ordering reproduces; the n=50 figure (2810 ms)
confirms the n=8 measurement was not an artifact. [Likely] FIXEDAUTO rescales after every
multiply and must match scales explicitly, so it does more level work per EvalSum rotation,
while FLEXIBLEAUTOEXT buys cheaper level handling with its extra modulus. Not profiled —
treat as a hypothesis.

Per-sample variance is large and is VM contention, not signal: within the n=50 run,
neighbouring samples ranged from ~2400 ms to 4596 ms. Any published latency number needs
warm-up discarded and the spread reported, not just a mean.

**The TenSEAL comparison (99.9 ms/sample) is not valid and is not claimed here.** TenSEAL
ran natively on Windows, OpenFHE in a WSL VM. Different environments; the numbers are not
comparable. A real benchmark needs one host, warm-up discarded, and hundreds of samples.

## Honest limitations

- **No trained checkpoint exists** (`ckks_models/` is absent, no `.pth` anywhere in the
  repo). Weights are random-init, so *accuracy* here is meaningless — the fixture says so
  in its own metadata. Numerical fidelity is unaffected: `W @ cls + b` behaves identically
  whether or not W was trained. The 10k-sample, 100%-match accuracy claim continues to rest
  on the earlier TenSEAL run in `results/ablations/ckks_*.json`.
- The full run is 50 samples, not 10,000. Enough to verify the cryptographic path, not to
  restate an accuracy result. At the packed method's 619 ms/sample a 10k run is ~1.7 h.
- Neither method is tuned beyond what is described above. `perrow` is kept deliberately
  unoptimized because its job is to mirror TenSEAL's `.dot()` for comparability.
- Classification head only. The full-backbone depth budget remains the open research debt
  (`HANDOFF` contribution #3).

## Next steps

1. ~~Packed matrix-vector product~~ — **done**, see above. `--method packed` is now the
   default: 4.5x faster, same correctness. Covers drill atom **PP5.8** (SIMD packing,
   matrix-vector products).
2. Re-run against a trained checkpoint once one exists, to reproduce the accuracy line
   under OpenFHE rather than inheriting it from TenSEAL. At 619 ms/sample a 10k-sample
   run is ~1.7 h, which is now actually practical (it was ~39 h under `perrow`).
3. Proper single-host benchmark before any latency number goes in a paper.
4. `FIXEDMANUAL` + explicit `Rescale`/`Relinearize` calls — that is the tier2 C++ exercise
   in miniature, and the version where nothing is hidden.

## Files

- `export_head_fixture.py` — stage 1 (Windows, torch). Reuses `DeiTTiny` from
  `core/ckks_classification_head.py` rather than redeclaring the architecture.
- `openfhe_head.py` — stage 2 (WSL, `openfhe` + stdlib only).
- `head_fixture.json` — the handoff artifact.
- `openfhe_head_results.json`, `res_*.json` — recorded runs.
