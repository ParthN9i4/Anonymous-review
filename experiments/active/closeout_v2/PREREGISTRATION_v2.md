# closeout_v2 pre-registration (written 2026-09-23, before any v2 outcome; amended the same day, still before any run)

`prepare_v2.py` records the SHA-256 of this file and of `protocol_v2.py` in every run manifest.
Any change after launch starts a new campaign version.

## Questions and hypotheses

| ID | Question | Hypothesis (stated before running) | What either outcome licenses |
|---|---|---|---|
| E1 | Does polynomial-attention damage come from the fitted exponential numerator, the fixed Goldschmidt reciprocal, or neither alone? | The reciprocal dominates at the budget profile. Its analytic worst-case relative error on the fitted denominator domain is ≈0.89 (budget, 2 steps) and ≈0.65 (4 steps). | Reciprocal dominates: "attention collapse under this conversion is a reciprocal-depth problem", backed by the F0 cost of fixing it. Numerator dominates, or both: the claim is re-worded accordingly. `exact_both` must reproduce the source; if it does not, E1 is void. |
| E2 | Is the failure only a budget choice, and which pre-deployment metric predicts the model-level prediction-change rate? | Accuracy recovers with degree and reciprocal steps. The fitted-interval sup-error and domain coverage rank configurations worse (lower Spearman ρ with the dev change rate) than activation-weighted error on gate inputs or the gate change rate. | Hypothesis holds: primitive sup-error and coverage checks are the wrong acceptance statistics, and a named alternative is supported. Hypothesis fails: report that sup-error suffices on this model, which weakens the "silent" framing; state it. |
| E3 | Which blocks carry the damage? | Descriptive; no directional hypothesis. | Localization only. |
| E1-med, E2-med | Do the E1 attribution and E2 dose-response patterns hold on BloodMNIST, PathMNIST and DermaMNIST (arm-000 checkpoints, float64, official validation split)? | Same directions as CIFAR-10. PathMNIST budget GELU, the known exception, recovers with degree. | Holds: operator-level findings generalize across the four datasets of this project. Fails: dataset dependence is reported per dataset. |
| E4 | Can a pre-deployment screen accept useful configurations with no unacceptable ones? (**developmental**: thresholds from seed 42, scored on held-out checkpoints 43/44 and a historically exposed test split; not independent confirmation) | At least one rule reaches nonzero coverage on held-out seeds with no false accepts. `v1_static` and `domain_coverage` are expected to fail (zero coverage, or false accepts). | Positive: a screen with measured coverage and risk. Negative: "no tested screen is reliable", reported with the table. |
| F0 | What does each primitive cost in CKKS levels? | Linear ops 1 level. Goldschmidt 1 + 2k levels. An accurate reciprocal on the v1 domain does not fit a block without bootstrapping. | Measured ledger; the block composition is labelled arithmetic. |
| F1 | Does encrypting a trained MLP sub-block of the failing conversion change predictions relative to plaintext P? | CKKS minus plaintext stays below 1e-3; zero prediction changes C vs P, while P vs O changes a large fraction. | Links the encrypted evidence to the headline failure. Any C-vs-P change is reported as found. |

**Overlap with evidence already in hand (recorded 2026-09-24, before launch).** The H200 return of 24 September already contains 27 medical validation-intervention jobs on the same arm-000 checkpoints: attention P/P, P/E and E/E, and all-operation P/P, on fixed validation subsets of at most 1,024 images (the medical validation-intervention results in the paper and in `docs/submission/external_review_2026-09-24/`). E1-med therefore adds three things only: the full official validation split, the offset-invariant metrics (centered-logit error, probability total variation, the margin sufficient condition) and wrong-to-correct counts. Its accuracy changes are a re-measurement, not new evidence. If an E1-med accuracy change differs in sign from the corresponding 24 September value, both are reported, and neither replaces the other.

## Fixed decisions
- **Data splits:** fit = train[0:2048], gate = train[2048:4096], dev = the 5,000-image validation split. Test is opened only by confirm jobs, which run after `screen_frozen.json` exists.
- **Acceptability of a configuration:** prediction-change rate ≤ 1% on the target split, with invalid converted outputs counted as changes. A 5% level is also reported but never used for selection.
- **Screen thresholds:** chosen per operator family on dev seed 42 only, as the largest observed value admitting no unacceptable development configuration. Frozen once, then applied unchanged to seeds 43/44 dev and to the test confirm pass.
- **Checkpoints:** the three regularized LayerNorm references (seeds 42–44), verified by SHA-256. Medical: arm-000 checkpoints of `medical_replication_v1` (seeds 42–44), verified against their training `status.json`. A checkpoint whose training did not complete is excluded and recorded, never imputed. No training.
- **Medical protocol:** fit/gate indices `default_rng(20260921).permutation(n_train)[:2048]` and `[2048:4096]` as in the original medical conversion; float64 evaluation; dev = official validation split. The medical test split is not opened.
- **Additional metrics** (reported, not used for selection): probability total variation; centered-logit error, which is invariant to a shared logit offset; the margin sufficient condition 2·(centered error) < reference top-2 margin; wrong→correct counts.
- **CKKS for F1:** TenSEAL, N=16384, modulus bits [60, 40×7, 60] (400 ≤ 438, SEAL tc128), scale 2^40, no bootstrapping, Galois keys. Public server context (`make_context_public`); only ciphertexts are serialized.

## Stopping rules
- Arrays start only after the synthetic and structural smoke jobs pass, and after a **real-checkpoint pilot gate** passes (`afterok`). The pilot requires: a no-op conversion reproducing the source logits bit-for-bit; exact-numerator/exact-division attention reproducing the source predictions (max logit difference ≤ 1e-4); and, for medical data, the recalibrated fits equal to the stored medical fits (relative difference ≤ 1e-9).
- F1: if the smoke run exceeds 60 s per token-chain, reduce to 8 images. The circuit is never changed.
- Failed jobs are recorded (`error.json`) and reported. They are resubmitted only for infrastructure failures (node loss or preemption), never for unfavourable numbers.
- No threshold, grid or split changes after any dev outcome is observed. Test results cause no re-tuning.

## Known limitations (stated in advance)
- Configurations share checkpoints and fits, so configuration counts are not independent trials.
- The CIFAR-10 test set has historical exposure in this project. The test pass is a locked confirmation, not a pristine one.
- F1 is hybrid replay (one encrypted sub-block). It is not full encrypted inference.
- The F0 block composition sums separately measured primitive depths. It is not a measured block.
