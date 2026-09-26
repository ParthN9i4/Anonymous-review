# Private ViT: focused closeout scripts

These scripts do not edit a manuscript, a checkpoint, or existing results. They
produce new reports only at paths supplied through `--output`.

## 1. Offline evidence audit

Extract `pvit_paper_evidence_core_20260924.tgz` into a fresh directory, then:

```bash
python3 closeout_audit.py /path/to/extracted/root
```

The root must contain `pvit_results` and `pvit_closeout_20260923` directly.
This reproduces aggregate counts, fit/gate overlap, CKKS status counts, and
three manuscript flags. Majority-class accuracies identify candidates only.
To check actual class collapse, optionally supply one `.npz` per candidate,
named `<dataset>_s<seed>_<arm>.npz`, containing arrays `labels` and `logits`:

```bash
python3 closeout_audit.py /path/to/extracted/root --predictions /path/to/raw-medical-predictions
```

Do not treat a candidate as verified until its predicted-class counts are
computed. The supplied core archive lacks those medical raw predictions.

## 2. Block-0 GELU diagnosis on the H200

Copy `block0_diagnosis.py` into the original campaign's `code` directory
beside `worker.py`. Activate that campaign's Python environment, ensure the
manifest's checkpoint and dataset paths exist, and run:

```bash
cd /home/anonuser/pvit_results/baseline_campaigns/YOUR_CAMPAIGN/code
python3 block0_diagnosis.py .. preserve_s42_budget_gelu_n2048 \
  --output /home/anonuser/pvit_results/block0_s42_diagnosis.json
```

The name `YOUR_CAMPAIGN` is a placeholder: use the directory containing the
actual `manifest.json` and `conversion` folder. Run for seeds 43 and 44 by
replacing both the name and output filename. This script takes a deterministic
sample of original block-0 GELU inputs from the original fitting and gate sets.
It fits a new degree-7 polynomial from fit inputs, compares activation errors
on the gate inputs, and evaluates native / original block-0-only conversion /
new block-0-only conversion on the gate. No test images are read. A positive
gate result is not yet a locked-test claim.

## 3. Degree-31 GELU primitive measurement on a CPU node

Copy `degree31_ckks_probe.py` beside the campaign's `fhe.py`, activate the
environment with TenSEAL, and run with at least 4 CPU threads and adequate RAM:

```bash
export SLURM_CPUS_PER_TASK=4
cd /home/anonuser/pvit_results/baseline_campaigns/YOUR_CAMPAIGN/code
python3 degree31_ckks_probe.py \
  ../conversion/preserve_s42_budget_all_n2048 \
  --output /home/anonuser/pvit_results/degree31_s42_block0.json
```

This uses the same degree-32768 CKKS context, 35-bit scale, and modulus chain
as the original budget primitive. It takes the same four preselected fixture
images, refits GELU to degree 31 on the **original frozen interval**, and
reports approximation/CKKS errors, levels, serialized bytes, and time. The
original `fhe_inputs.npz` is required and is absent from the supplied core
archive. A backend exception or insufficient chain is an outcome to record;
do not change parameters silently or report the result as full-model cost.

## Remaining provenance checks

- Inspect the medical arm-111 training traceback to classify its failure.
- Trace the exact invalid-output-to-accuracy rule in the medical evaluator.
- Check fit IDs against *all* validation ID lists; this package only checks
  fit versus the separate calibration gate for the 26 frozen CIFAR jobs.
- Compare `med_e1`'s exact job definitions and metrics with the already
  archived 378 medical validation variant rows before scheduling anything.
- Apply manuscript wording changes only after reconciling the latest draft.
