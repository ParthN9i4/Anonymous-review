# Private ViT evidence update (20 September 2026)

This dated directory is additive. The repository's earlier model implementations, logs, paper draft, and historical README remain available. It records the experiments completed on the shared H200 and CPU host since the GitHub main commit `7b3af215110f64f81927c1122199686db59fcce1`. The H200 working copy was reported at `5428ef1fef9f96982d9672cd9da0038dcd283ac6` during job 556 and may include uncommitted differences; **these Git commits have not been reconciled**. The frozen `source_final/` directory is the executed final-assessment code from the supplied result bundle, isolated from the root trainer.

## Experimental families (do not pool incompatible protocols)

| Experiment | Inputs and unit | Headline evidence | Reference |
|---|---|---|---|
| Legacy eight-arm factorial | CIFAR-10, five paired seeds 42–46; validation-selected; unchanged-KD arm | 0: 77.536%, B: 69.956%, C: 81.540%, E: 28.600%, F: 42.384%, G: 79.960% mean validation accuracy. E/F are seed-sensitive and have invalid outputs; G has rare enormous finite loss. | `results/factorial_summary.csv` and earlier 67-replay audit |
| Strong reference | Three seeds, CE only, validation-selected; legacy and regularized recipes | Legacy LN 76.800%, BN 81.413%; regularized LN 86.707%, BN 87.553% validation. Changes between recipes are confounded. | `results/baseline_summary.csv` |
| GELU × norm control | Three regularized seeds; controls plus six new 100-epoch runs | GELU effect LN +0.86/−1.08/−1.18pp; BN −4.00/−3.20/−3.04pp. Interaction mean −2.95pp. | `results/regularized_gelu_interaction.csv` |
| Smooth transition | 14 full 30-epoch continuation runs: development 42/43 and confirmation 44–46 | Simultaneous smooth E conversion ends near 10% on all five seeds; abrupt E succeeds on only one at ~39% test. Tests a specific schedule/destination, not cold vs warm KD generally. | `results/smooth_schedule_summary.csv` |
| Custom positive P/Q | Five saved checkpoint seeds; attention normalized and nonnegative | Existing custom attention trains, but is not a faithful PowerSoftmax or Powerformer implementation. Endpoint scores must retain their campaign and split labels. | `results/power_summary.csv` |
| Frozen conversion | 3 seeds × 2 degree profiles × 4 masks plus 2 calibration-size controls = 26; 2048 fit + disjoint 2048 training gate; 10,000 clean test per configuration; two fixed 1,000-image transformed cohorts | Budget/full mean clean accuracy 10.21%; higher degree/full 69.54%; current reference mean 86.27%. All 26 static gates reject. Seed43 budget GELU falls 86.33% → 25.55% with 0/10,000 out-of-domain flags and no invalid logits. | `results/conversion_verified.csv` and `results/detectors.csv` |
| Endpoints | 15 saved models (P/Q, legacy B/E/F on seeds42–44), three cohorts | These are replay diagnostics of retrained model modifications. Do not subtract their scores from frozen reference as a pure polynomial/CKKS error. | `results/endpoints.csv` |
| Final CKKS | 12 reports (3 seeds × 2 profiles × blocks 0/5), 126 measured trials + 9 invalid input/target cases | Captured trained primitives only. The maximum saved CKKS-to-plaintext arithmetic error is below 1e−3 for measured cases. The earlier 72-report campaign had 459/1116 nonwarmup trials beyond its own recorded tolerance, under different circuits and chosen stress cases. | `results/ckks.csv`, `results/completion.json`, `results/prior_ckks_audit.json` |

**Disjointness:** 2048 fitting and 2048 screening images are disjoint *from one another*, but both are drawn from training data used to fit the model. The 10,000 clean images are CIFAR-10 test images repeatedly exposed earlier in this project. Noise and brightness cohorts transform some of the same test images; they are not external datasets. Repeat trials, image rows and model layers are not independent model seeds. Validation accuracy must not be compared directly with differently run test accuracy.

## Arithmetic and scientific interpretation

The frozen arithmetic uses Chebyshev-node least-squares polynomial fits; it does **not** provide uniform minimax error bounds. Low/high-degree labels are profiles (GELU and inverse-root degrees 7/15; exponential-root 4/8; reciprocal iterations 2/4), not guarantees of accuracy. The evaluation failure condition is finite reference and any converted invalidity, argmax flip, or raw logit error over 0.1; a uniform shift to all logits can satisfy the last condition without changing probabilities. Inspect actual label flips and probability divergence separately.

The last CKKS campaign evaluates selected attention rows, GELU features and an inverse-square-root normalization primitive from trained captures. It does not encrypt full QK and AV products, full LayerNorm, a residual-connected block, the six-block backbone or a classifier head in a single ciphertext chain. The 65-ciphertext attention packing puts only four useful rows into 16,384 slots; reported 50.86/95.43-second median attention primitive evaluation is not per-image encrypted inference. Backend physical multiplication counters are unavailable; null is not zero. Earlier CKKS failures remain evidence under their own tolerance and circuit definitions.

## Data and provenance

The supplied original final archive is named `campaign_23crox26_return(1).tgz`, SHA256 `748c8c9e0fa43a27e149cb6f1d78ae61d811396cbbb5cd89254cc869a708ca29` (approx. 85 MiB). It contains raw NPZ predictions and decrypted arrays, `source_hashes.json`, `checkpoint_provenance.json`, the complete executed source and report outputs. Keep it and the untouched H200 checkpoints outside Git. The compact repo tables are derived views, not replacements for raw evidence.

The saved-output audit recomputed reported classification and detector metrics from 123 saved prediction files (78 conversion + 45 endpoint) and checked 126 decrypted trial errors against frozen arithmetic without numerical discrepancies. This **did not** rerun trained checkpoint inference, verify missing checkpoint bytes or rerun CKKS on ciphertexts. `results/verification_scope.json` records the bound. Source snapshot file hashes are checked against `source_hashes.json` from the archived run, available in `results/source_hashes.json`.

Historical `verification_summary.json`, `verification_curves.json`, early test-selected CIFAR-10 substitutions and `activation_magnitudes.json` are tainted or superseded; preserve them as historical records only. Earlier BloodMNIST and CIFAR-100 experiments have different recipes/splits and are not external confirmation of this frozen conversion procedure.

## Next decisions

1. Faithfully reproduce a published approximation on matching trained ViT inputs and cost/error definitions; add a viable candidate so the gate can have positive coverage.
2. Freeze acceptance criterion, then validate accuracy of accepted candidates and missed failures on a genuinely new dataset/protocol. Report calibration and independent confirmation separately.
3. Measure packing occupancy and costs, then implement one continuous trained encrypted block, two blocks and the complete backbone. A bootstrap experiment must use a backend that supports it; no full-model result exists today.
4. Measure client-side inference latency, memory, energy and offline/device constraints before claiming remote encrypted inference serves users better than local cleartext inference.
5. Archive code+report hashes for each future campaign and append dated decisions. Cold/warm KD requires initialization × objective and learning-rate controls with matched update budgets; the failed smooth experiment alone cannot settle it.

Do not modify the existing running H200 checkout to review this branch while jobs are queued. Use `git worktree` or a separate clone and compare remote/local commits before copying training changes. No branch or PR here constitutes a SaTML anonymous artifact.
