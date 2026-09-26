# Review of F3 and F4 graphics, 20 September 2026

The separate manuscript branch commit `8014cdba8c6f2cae8d2bf087f46fd5fc7b9f9db3` introduced `paper/make_figures.py` and two figures. The file and its source CSVs were inspected. The figures are not part of this evidence snapshot and should be regenerated after the fixes below.

## F3: accuracy against extreme finite loss

The figure juxtaposes three `regularized_quadratic_bn` validation measurements (accuracy 83.64, 84.02, 84.76%, maximum finite per-image NLL 3.25e7, 4.50e6, 2.51e8) with paired `regularized_quadratic_ln` validation measurements (accuracy 86.90, 85.86, 85.96%, maxima 5.35, 5.53, 5.25). The contrast is real but the annotation “same accuracy” is too strong: paired accuracy differs by −3.26, −1.84 and −1.20 percentage points.

**Resolve a split mismatch:** the highlighted six points come from `recipe_quadratic_tail.csv` using `validation_acc` (5,000 validation images). Fifteen gray points come from `endpoint_tail_metrics.csv` with `split == clean` (10,000 clean test images). Mixing these on a single unqualified accuracy/tail axis can mislead. Use one split and a matched selection protocol for every plotted arm; if validation per-image outputs are unavailable for the gray arms, omit the gray arms and show the six paired points with seed identifiers.

The x≥80% shading and NLL=100 line illustrate a possible accuracy-only screen and a recorded finite-tail threshold, not a calibrated deployment acceptance rule. Mark them “illustrative” or remove the acceptance shading. Say that nonfinite predictions are counted as accuracy failures but excluded from the finite-loss maximum; give invalid counts beside the figure. At 3.45-inch column width, check the real paper PDF: the legend masks an older point, and the orange matched-control label/arrow crosses low-lying points in the currently shared preview.

## F4: approximation against CKKS arithmetic error

The paired bars report independent medians of `approx_relative_l2` and `ckks_relative_l2` from `ckks_primitive_ledger.csv`, filtered to `status == MEASURED`, separately by operation and profile. Labels are ratios of those two medians. Clarify the exact reference values and denominator floor in the figure caption and give the number of measured trials and excluded statuses per group. A ratio of independent medians need not equal a median paired-trial ratio.

These are **separate encrypted primitive circuits on captured trained values**, not a chained transformer block, backbone, inference latency, or bootstrap. For inverse square root, an 18× or 65× difference means CKKS error is approximately 5.6% or 1.5% of the approximation error on the plotted median measure; “co-dominant” should be removed. For softmax and GELU, approximation dominates under the measured parameter/profile choices, which does not establish a universal HE bottleneck. The latency and multiplicative depth table needs to accompany F4 for the efficiency claim.

## Release rule

Regenerate all figures from committed generating code and source tables. Verify number of samples, shared image IDs, train/validation/test split, status filtering, ratios, caption limits and collisions in the rendered IEEE two-column PDF. Preserve the old figure inputs and scripts under their own dated provenance; do not silently restyle older graphics without checking their data lineage.
