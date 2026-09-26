# Activation-density GELU refit: gate and test evaluation (25 September 2026)

Unpacked from `block0_refit_eval_return_20260925.tgz`
(SHA-256 `d408f5e43fa21d1aba985efcf35fba269ca1bd64b83640cda74b4b18a435b6ce`). The file hashes match
`file_hashes.sha256`, in which the home path is anonymized. The script that ran is `block0_refit_eval.py`
(SHA-256 `a46a822c…`). It also replaces the earlier draft at
`experiments/active/closeout_2026-09-24/block0_refit_eval.py`.

## What each configuration means
| JSON key | Configuration |
|---|---|
| `reference_*` | Native model |
| `frozen_*` | Deployed low-cost GELU-only conversion (all six GELU sites), recalibrated with the campaign's `calibrate()`. The test accuracies reproduce the recorded 26.93/25.55/22.72% exactly |
| `block0_*` | The frozen conversion with **block 0's GELU replaced by the refit**; blocks 1–5 keep the deployed fits |
| `all_blocks_*` | All six GELU sites replaced by their refits |

- **How the refit is computed:** degree 7, on the deployed interval of each site, by least squares in the
  Chebyshev basis on GELU inputs captured from the 2,048 fitting images.
- **Data splits:** the gate is 2,048 disjoint training images (`fit_gate_overlap` = 0). The test is the full
  10,000-image CIFAR-10 test set, which is historically exposed. It is evaluated after the refit is fixed and
  is not used to fit.

## Results (mean ± SD over seeds 42–44)
| Configuration | Gate (%) | Test (%) |
|---|---|---|
| Native | 98.62 ± 0.44 | 86.27 ± 0.09 |
| Frozen conversion | 24.69 ± 2.51 | 25.07 ± 2.15 |
| Block 0 refitted | 86.69 ± 0.80 | 78.67 ± 0.46 |
| All six sites refitted | 95.33 ± 0.25 | 83.48 ± 0.56 |

No invalid outputs occurred in any configuration.

## Caveats
- `weights_preserved: true` is written by the script, not measured. The weights cannot change here (inference
  only, no gradients), but the field is not an independent check.
- This is not the earlier `block0_s*.json` diagnostic in `results/closeout_2026-09-24`, which converted
  *only* block 0 (gate 88.61%).
- The refit coefficients are not saved in the JSON.
- The refit has not been evaluated under CKKS.
