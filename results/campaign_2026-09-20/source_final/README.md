# Final frozen-conversion assessment campaign

This package contains all runner/model code. It needs only the already-trained checkpoint bytes and CIFAR-10 cache on H200, plus installed GPU PyTorch/torchvision. It does not import earlier delivered packages or modify the project repository. Checkpoint paths and hashes are supplied in checkpoint_inventory.json; the preparer verifies bytes before submission. Python 3.13 on H200 is supported by the pinned FHE dependencies.

## What runs tonight

| Work | Jobs | Purpose |
|---|---:|---|
| Frozen standard reference → polynomial arithmetic | 24 | 3 seeds × 2 degree budgets × GELU/LN/attention/all |
| Calibration-size sensitivity | 2 | Seed42/all, 128 rather than 2048 fit images, same 2048-image gate |
| Existing P/Q and B/E/F | 15 | 3 seeds × 5 checkpoints, identical clean and stress cohorts |
| Actual leveled CKKS | 12 | First/last block fixtures from each main all-operator conversion |
| Structural smoke | 2 | GPU/model/checkpoint check and actual encrypted arithmetic check |
| Report/archive | 1 | All results, exceptions, costs and figures |

No new training. The completed six GELU×normalization training runs and 72 prior CKKS reports are audited and copied, not rerun. Each FHE job tests both source-path and converted-path values, three operations and two trials; invalid inputs/targets are retained as explicit outcomes instead of filtered away.

## Run from Windows and H200

Download pvit_final_assessment.zip to C:\Users\YOUR_USERNAME\Downloads. In Windows PowerShell:

```powershell
scp C:\Users\YOUR_USERNAME\Downloads\pvit_final_assessment.zip YOUR_USERNAME@YOUR_H200_HOST:/home/anonuser/
ssh YOUR_USERNAME@YOUR_H200_HOST
```

Then on H200:

```bash
module load anaconda-2025.12-2
PVIT_STAGE="$(mktemp -d "$HOME/pvit_final_assessment_XXXXXX")"
unzip -n "$HOME/pvit_final_assessment.zip" -d "$PVIT_STAGE"
cd "$PVIT_STAGE/pvit_final_assessment"
bash launch.sh
```

launch.sh pins NumPy 2.5.3 and TenSEAL 0.3.18 in a separate CPU environment. It does not replace your GPU PyTorch. Package setup may need internet; GPU jobs never download CIFAR. Each launch creates an isolated source snapshot and records job IDs immediately. Do not launch a second copy while a first campaign is pending/running.

Check from any directory:

```bash
PVIT_RUN="$(cat "$HOME/pvit_results/latest_final_assessment.txt")"
bash "$PVIT_RUN/code/status.sh"
```

If the normal report job finishes, return_archive.txt and return_sha256.txt identify the archive. For a failed smoke or interrupted campaign, collect available evidence without rerunning anything:

```bash
PVIT_RUN="$(cat "$HOME/pvit_results/latest_final_assessment.txt")"
bash "$PVIT_RUN/code/collect.sh"
```

From Windows, copy the exact archive path printed by the collector. To avoid manually entering the campaign suffix, use PowerShell:

```powershell
$archive = (ssh YOUR_USERNAME@YOUR_H200_HOST 'cat "$(cat "$HOME/pvit_results/latest_final_assessment.txt")/return_archive.txt"').Trim()
scp "YOUR_USERNAME@YOUR_H200_HOST:$archive" C:\Users\YOUR_USERNAME\Downloads\
```

`afterok` structural checks intentionally block dependent jobs on failure. Numeric failures in full experiments remain scientific results. Do not replace a failed gate with an unconditional success or repeatedly resubmit. Inspect logs and collect the archive.

## Frozen protocol

The original fixed CIFAR split is 45,000 train / 5,000 validation. This campaign fits arithmetic on the first 2048 examples of the clean training split, then fixes a configuration-level decision on the next disjoint 2048 training examples. Both groups participated in original model training: they are not independent of model fitting.

A frozen.json is written and hashed before this campaign opens the 10,000-image CIFAR test set. Fixed brightness (×0.7) and noise (σ0.03 in pixel units) stress cohorts use its first 1000 images. Those cohorts are paired transformations, not independent datasets. The project has previously inspected test results; disclose that historical exposure. No result in this campaign is a pristine external confirmation.

The model weights are exactly copied and frozen. Only operator arithmetic changes. `budget` uses GELU7, inverse-sqrt7, squared exp-root4 and two reciprocal refinement iterations. `accurate` uses 15/15/8 and four iterations. The name `accurate` is an identifier, not a guarantee. Neither is claimed best, minimax, or a faithful reproduction of ATLAS/PowerSoftmax. The purpose is to expose a controlled accuracy/cost/domain tradeoff.

Softmax scores are row-mean-centered. A public shift is included in fitting exp(z/2), whose polynomial is squared and given a 1e-12 floor. The positive numerator is normalized using a fixed Goldschmidt reciprocal. No data-dependent clamp, actual division, maximum, oracle fallback or refitting is inserted into deployment arithmetic. Exact row mean/variance and fixed affine parameters are retained in LN; only inverse square root changes. Invalid outputs and domain violations are preserved.

## Outcome and assessments

A conversion failure requires a finite reference and at least one of: nonfinite converted logits, changed argmax, or maximum absolute logit error >0.1. This is a declared engineering tolerance, not a universal safety standard. Flips, correct→wrong changes and invalidity are also reported separately so the logit tolerance cannot hide prediction outcomes.

Four fixed assessments are compared:

1. Configuration-only: review every input for a conversion if >1% of the calibration-gate examples fail.
2. Finite-output screen.
3. Per-input domain screen on actual converted-path operator inputs, plus finite-output check.
4. Domain screen plus a second FP64 converted forward: review for nonfiniteness, self-prediction mismatch, or FP32/FP64 maximum discrepancy > one quarter of the FP64 top-two margin (floor1e-8).

Reference held-out logits are used to score these screens, never as inputs to the warning rule. FP64 is another floating-point computation, not exact arithmetic; both precisions can agree and be wrong. A configuration rejected for every image achieves zero accepted coverage, not proven useful reliability. These are offline predeployment assessments, not cheap checks inside encrypted inference.

## Metrics and scope

JSON and per-input NPZ include source/converted/FP64 logits, image IDs, invalid counts, accuracy counting invalids as failures, finite-only NLL and tails, ECE, Brier score, prediction flips, correct→wrong transitions, domain exceedance, warning decisions, false positives/negatives, recall, precision, coverage and failure rate among accepted inputs. Show denominators. Do not treat finite-only metrics as full-cohort reliability.

Timing measures warmed synchronized reference/P32/P32+range/P64 forwards on a fixed gate batch. Instrumented complete evaluation time is stored separately. Peak GPU allocated/reserved memory and process RSS are included, with process boundary disclosed. Additional FP64 assessment costs another complete forward; deployment arithmetic latency and assessment latency must not be conflated.

FHE records CKKS parameters, level consumption, public context size, serialized input/output bytes, encryption/server-evaluation/serialization/decryption times, numerical error against exact and deployed targets, exceptions, output scale, and whole-process RSS. Actual physical multiply/rescale/rotation instrumentation is unavailable in this wrapper: those counters are null, not estimated scalar MACs. There are no rotations in the explicit column-packed softmax computation. No energy or network timing is fabricated.

## What is encrypted

For each selected trained fixture: (a) a complete attention-weight row from raw scores, including encrypted mean centering, numerator polynomial, denominator, reciprocal and normalization; (b) GELU polynomial on class-token features; (c) inverse square root on the class-token LN variance. Source and actual converted-path fixtures are both included. The server context contains no secret key, and only final outputs are decrypted.

This is leveled CKKS with ring degree32768, 18 intermediate primes, scale35 or40, SEAL default tc128 parameter validation, no Galois keys and no bootstrap. Attention head0/query0 is fixed before outcomes. Four images occupy SIMD lanes; 65 ciphertexts represent a row. Low slot utilization and large communication are disclosed. This is a correctness/measurement implementation, not optimized packing.

It does NOT encrypt QK, AV, the complete LN layer, a transformer block or the backbone. The previous campaign's encrypted MLP branch remains separate evidence. Full-backbone accuracy and bootstrapping still have not been measured. Compare our operation boundaries honestly with published full-model systems.

## Files

- prepare.py / protocol.py: verified inputs, isolated snapshot, frozen jobs and thresholds.
- polynomial.py / polynomial_numpy.py: fitting, floating-point deployment arithmetic and matching NumPy reference.
- worker.py / data_eval.py: frozen conversion, clean/stress cohorts, held-out replay and timing.
- assessment.py / metrics.py: explicit failure definitions and decision-quality measurements.
- fhe.py: actual public-context CKKS circuits and ciphertext measurements.
- native_reference.py: model classes copied verbatim from native_source.py, omitting unused training imports; native P/Q parity is tested.
- pvit_lab/: bundled checkpoint/model/data code; no external previous-package import.
- audit_previous.py: reads completed prior72 reports, including tolerance failures, without rerunning them.
- figures.py / report.py: risk/coverage, paired added-check benefit, range/error plots, CSVs, archive.
- launch.sh / run_*.sbatch / status.sh / collect.sh: scheduling and evidence retrieval.
- test_*.py / VALIDATION_SCOPE.json: local and H200 smoke validation scope.

All plots are descriptive. Inputs/layers/repeated ciphertext trials do not become additional training seeds. Matplotlib is optional for report figures; CSV/JSON survive its absence. Raw return bundles contain hostnames/paths and require an anonymized release copy before linking them in a paper.
