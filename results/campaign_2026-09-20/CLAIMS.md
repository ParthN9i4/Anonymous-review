# Claim ledger

| ID | Claim | Evidence and status | Boundary / forbidden extrapolation |
|---|---|---|---|
| C01 | Attention×BN interaction in the legacy ViT | Five-seed clean factorial, validation replay | Recipe-specific, not universal polynomial instability |
| C02 | G has extreme finite loss tails | Saved logits independently checked in earlier67-replay audit | Same development image across seeds; not five independent inputs |
| C03 | BN advantage shrinks under regularization | Three-seed four-cell CE-only reference experiment | Multiple recipe ingredients change together |
| C04 | Tested gradual conversion fails |14full runs and eligible-checkpoint verification | Does not establish cold>warm KD or refute all smooth-transition methods |
| C05 | Regularized GELU×norm interaction averages−2.95pp | Six new runs plus controls, validation | Three seeds, no causal mechanism isolated |
| C06 | Some in-domain finite conversions distort predictions severely | Latest raw predictions,123files checked overall | Chosen weak profiles, not a faithful modern comparator |
| C07 | Calibration screen rejects all26 configurations | Frozen gate files and confusion counts | Zero accepted coverage, no proven useful detector |
| C08 | FP64 self-consistency misses approximation bias | Latest detector reports independently recalculated | FP64 is not exact reference arithmetic |
| C09 | Measured CKKS errors below1e−3 on captured primitive values |126trials,63finite cases,2repeats | No full encrypted backbone/long-chain guarantee |
| C10 | Encrypted primitive costs measured |CPU,4threads,ring32768,declared packing/precision | Not per-image full-model latency or SOTA ranking |
| C11 | Faithful published comparator | Missing | Custom P/Q is not PowerSoftmax/Powerformer reproduction |
| C12 | Independent external confirmation | Missing | CIFAR shifts and new seeds do not create pristine external data |
| C13 | Full encrypted backbone and bootstrap | Missing | Primitive chaining not implemented in current evidence |
| C14 | Model-specific privacy-preserving conversion advisor | Proposed future direction | Generic automated approximation search is already prior art |
| C15 | Remote service practical advantage over local inference | Unverified application assumption | Compact-model resource constraints must be measured |

| C16 | Earlier CKKS branches exceeded numerical tolerance in 459 of 1,116 trials | Archived 72-report audit: reciprocal 36/36, MLP 411/540, quadratic 12/540 | Stress-selected trials; different circuit and parameters from latest primitives; not a population failure probability |

Forbidden shortcuts: scheduler COMPLETED=scientific success; finite=accurate; matching NaNs=valid export; in-range=low approximation error; low ECE=useful classifier; new seeds=independent test set; bootstrap=repair of a bad polynomial;126CKKStrials=126models; three baseline reproductions=three new independent seeds.
