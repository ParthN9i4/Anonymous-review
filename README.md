# Artifact for “Diagnosing Silent Accuracy Loss in Model-Preserving Conversion of HE-Friendly Vision Transformers”

This repository contains code, configurations, and recorded results for the paper. The study examines three separate settings:

1. **Training with substitutions:** combinations of changes to activation, attention, and normalization operations.
2. **Conversion with fixed learned weights:** replacement of operations in trained models with fitted polynomial approximations.
3. **Evaluation under encryption:** comparison of original operations, plaintext polynomials, and the same polynomials evaluated under CKKS on captured inputs.

These experiments use different checkpoints and do not form a single encrypted inference pipeline.

## Repository contents

- `core/`: model implementations and training code.
- `experiments/active/`: scripts for the reported experiments.
- `experiments/exploratory/`: additional diagnostic experiments.
- `results/`: recorded outcomes, organized by campaign. See [`results/README.md`](results/README.md).
- `paper/`: manuscript-related analysis and figures.
- `infrastructure/`: environment setup and SLURM scripts.

## Scope of the evidence

The experiments under encryption evaluate individual polynomial operations on values captured from trained models. They do not evaluate a complete ViT on encrypted images. The MedMNIST results are benchmark experiments, not clinical validation.

Some older files under `results/ablations/` come from exploratory runs and should not be used in place of the validation-selected factorial results reported in the paper. The later degree-7 GELU refit retains aggregate accuracies and invalid-output counts, but not its refitted coefficients or individual predictions.

Trained checkpoints are not included in this repository. Their recorded paths and SHA-256 hashes are listed in [`checkpoint_inventory.json`](results/campaign_2026-09-20/source_final/checkpoint_inventory.json).

## Getting started

From the repository root:

```bash
conda env create -f environment.yml
conda activate private-vit-fhe
source infrastructure/env.sh
```

The experiment scripts are under `experiments/active/`. The recorded conversion and encrypted-operation campaign is under `results/campaign_2026-09-20/`; its [campaign README](results/campaign_2026-09-20/README.md) and [`infrastructure/RERUN_PLAN.md`](infrastructure/RERUN_PLAN.md) provide further run details.
