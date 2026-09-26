# Research decisions after the completed next-phase campaign

The supplied H200 readout establishes completion of six training/assessment jobs and 72 CKKS report files. It does not establish 72 successful numerical executions. audit_previous.py reads those existing detailed reports on the server and preserves their evidence without rerunning them.

## What the new training results say

| Seed | Quadratic effect with LN, validation pp | With BN | Difference of effects |
|---|---:|---:|---:|
|42|+0.86|−4.00|−4.86|
|43|−1.08|−3.20|−2.12|
|44|−1.18|−3.04|−1.86|

All three interaction contrasts are negative; their descriptive mean is −2.95pp. Quadratic GELU is more detrimental with BN under this regularized recipe. This is a useful replication under a stronger training reference. It contradicts any universal statement that GELU is inert or automatically rescues composition. Three existing seeds do not support a general law, and these validation differences cannot be combined with earlier test means.

Why is not established by this table. Plausible explanations include changed input distributions, residual amplification and loss of tokenwise normalization. Those require trajectory measurements and controlled interventions, not explanation by intuition alone. BN's earlier accuracy advantage is recipe-dependent evidence; a normalization that improves classification can still permit rare extreme or nonfinite predictions in another composition.

Smooth-transition failures concern the tested schedules, destination operator, initialization and budget. They do not show that warm KD, all gradual replacement, or published approximation-aware training is ineffective. Do not add a new KD grid to this overnight queue. It would dilute the main question before the conversion assessment is complete.

## Central question and contribution

Can calibration-time acceptance criteria identify conversions that remain numerically and predictively reliable on other inputs? How much does an additional per-input assessment improve detection, and at what computational cost?

The contribution must be the answer, including counterexamples, not a new name for polynomial activation. A useful result could be that calibration acceptance misses harmful tail events, range checks catch some but overreject, and a precision check has a measurable incremental benefit or demonstrably fails to detect shared approximation bias. These are hypotheses until the new reports exist. Do not promise those findings in the abstract.

Three distinct failures must be separated:

1. The trained substituted architecture itself has poor or invalid predictions.
2. Frozen conversion arithmetic differs from the original model.
3. CKKS adds error to the exact same deployment arithmetic.

P/Q and B/E/F supply model-modification context; the frozen reference supplies the controlled conversion comparison. Never compare differently trained weights and call the gap ciphertext error.

## A focused paper structure

1. Problem and deployment boundary: private outsourced inference on resource-limited clients. State that local inference is a meaningful baseline. Connectivity, encryption costs and device constraints need measurement before any remote-area deployment claim.
2. Threat model and reliability outcomes: honest-but-curious server, privacy distinct from numerical/predictive correctness, no malicious-server integrity guarantee.
3. Experimental design and provenance: actual custom 6-block/2.69M ViT, datasets, recipe controls, historical test exposure, checkpoint selection, independent units and tolerances.
4. Composition evidence: factorial results, stronger regularized GELU×normalization interaction, per-seed collapse/invalidity and counterevidence.
5. Frozen conversion and assessment: calibration split, fixed degree/cost budgets, accepted coverage, misses/false alarms, added-check cost, stress cohorts.
6. Encrypted arithmetic validation: source→deployment→CKKS error decomposition, trained captures, secret-key boundary, levels/communication/latency/memory, all failures and unsupported boundaries.
7. Limitations and implications: simple polynomial baseline, no claim of best approximation, historically exposed test, single architecture, no independent dataset confirmation, no full-backbone encrypted accuracy or bootstrap.
8. Open Science and LLM usage disclosures, with required human verification of results, code and references.

A source-code defect found during auditing is not by itself a broad contribution. It becomes meaningful when its prevalence and impact are measured, competing explanations are tested, and a practical assessment is evaluated against both successful and failed conversions.

## Decision after tonight

- If every configuration fails calibration: report it, but do not present a detector as useful merely because it rejects everything. A competitive conversion baseline and a wider mix of viable cases are then essential before a strong submission.
- If range checking flags nearly all clean inputs: quantify zero/low accepted coverage; do not hide it behind recall.
- If FP64 brings no extra useful detections: retain this negative result. More precision cannot fix approximation bias.
- If the approximation passes but CKKS fails: inspect scale/depth/cancellation and input magnitudes. Do not retune parameters on held-out outcomes and reuse the same set as independent confirmation.
- If assessments work on current cohorts: freeze them and confirm on a genuinely new dataset or prospectively reserved model seeds before broad predictive claims. CIFAR stress copies are not a substitute for that.
- If only source-path encrypted fixtures work: report the deployment-path failures. Selecting only valid source captures would overstate reliability.

Keep any final additional confirmation small and justified by an actual uncertainty in these results. Full-model FHE is not a submission dependency. Its implementation requires packed QK/AV and linear layers, depth scheduling, key/communication planning and possibly bootstrapping. This package does not provide those missing pieces under an end-to-end label.

## Comparison discipline

Published system comparisons need architecture, input/token shape, task, weights/training recipe, cryptographic threat model, security parameters, precision, bootstrap count, machine/threads, packing, batch, client/server boundary, communication and model accuracy. Otherwise use a feature/scope table, not a speedup ranking. Our 65-ciphertext row implementation is a transparent correctness baseline with expensive communication; it is not an optimized system competitor.

Official technical references used for the implementation: [TenSEAL](https://github.com/OpenMined/TenSEAL), [Microsoft SEAL CKKS example](https://github.com/microsoft/SEAL/blob/main/native/examples/5_ckks_basics.cpp), and [Slurm sbatch dependencies](https://slurm.schedmd.com/sbatch.html). This campaign is not a faithful reproduction of ATLAS or PowerSoftmax. Existing P/Q should be called custom positive normalized attention unless equation/training fidelity is independently established.

## SaTML timing and trustworthiness

The current [SaTML CFP](https://satml.org/call-for-papers/) lists mandatory abstract registration September22, paper September29, and anonymous artifact updates October2, 2026, all AoE. The September17 update requires fixed authors, affiliations and topics and disallows substantial title/abstract changes after registration. Register the actual measurement/reliability scope, not an aspirational encrypted-backbone contribution.

SaTML explicitly asks empirical papers to identify their trustworthiness question and engage with it throughout. The relevant claim here is dependability of predictions after privacy-preserving conversion, and the effectiveness/limitations of assessment before deployment. Neither use of encryption nor a healthcare motivation alone makes the paper trustworthy-ML research. The review risk remains high if the outcome is merely that a weak approximation fails; controlled comparison, viable conversions and generalization of the assessment are the essential remaining evidence.
