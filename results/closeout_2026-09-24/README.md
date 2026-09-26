# Closeout return, 24 September 2026 (block-0 refit and degree-31 CKKS probe)

Unpacked unchanged from `pvit_closeout_return_20260924_v2.tgz`
(SHA-256 `860a141cd064dde3fde44ac85284c5906ca155d630c6dd035431d31c1d1b8d1c`), produced on the H200 by
the scripts in [the closeout scripts](../../experiments/active/closeout_2026-09-24/)
(bundle `private_vit_closeout_scripts_20260924_v2.zip`, SHA-256
`9f61cef92670c7172d2d4a174f19188a13a9d20c4735cde67104c80f8b99b4d9`; the earlier bundle
`..._20260924.zip`, SHA-256 `5c8315d101516a75701484aed620c33dcf39d998494e03e7d4b929e16eb3579e`, differs
only in `degree31_ckks_probe.py`, which the v2 bundle made independent of PyTorch).

| Files | What |
|---|---|
| `block0_s4{2,3,4}.json` | Degree-7 block-0 GELU refit on 200,000 sampled fit-set activations vs. the deployed Chebyshev-node fit; errors on sampled gate activations; gate accuracy with only block 0 converted. Gate = 2,048 training images, disjoint from the fit set. No test images. |
| `degree31_s4{2,3,4}_block0.json` | Degree-31 block-0 GELU refit evaluated under TenSEAL CKKS with the original low-cost chain (35-bit scale), two trials per seed, on the four-image fixture. |
| `*.out`, `*.err`, `slurm_accounting.txt` | Job logs; all six jobs COMPLETED. The `.err` files hold a NumPy poor-conditioning warning for the refit. |

Known issue: in `block0_s*.json`, `empirical_polynomial.grid_max_abs_error` and `grid_min_output` are copied
from the deployed fit (`{**domain, ...}` in `block0_diagnosis.py`), so they do not describe the refit. The paper
does not use them.

The paper's copies are in `paper/revised/data/closeout_2026-09-24/` and feed `scripts/make_tables.py`.
