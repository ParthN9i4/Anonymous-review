# Harness Code: Baseline Campaigns Analysis

This directory contains the analysis harness for the baseline campaigns, sourced from the verification bundle dated 2026-09-17. It retires the "code was not read" limitation by making the analysis pipeline inspectable.

## Contents

- **`pvit_lab/`**: Package implementing the baseline training and evaluation pipeline
  - `model.py`: DeiT-Tiny variant architecture definition (6 blocks, 2.69M params, 65-token sequence)
  - `train.py`: Knowledge distillation (KD) training loop, with activation substitution variants
  - `campaign.py`: Campaign orchestration (legacy vs. regularized recipes, arm configurations)
  - `measure.py`: Metrics computation (accuracy, NLL, tail statistics, ECE, Brier)
  - `deploy.py`: CKKS plaintext approximation evaluation
  - `data.py`: Dataset loading (CIFAR-10, BloodMNIST)
  - `audit.py`: Checkpoint and training state validation
  - `ckks_primitive.py`: CKKS fixed-point arithmetic and precision handling
  - `diagnose.py`: Activation profiling and failure diagnostics
  - `profile.py`: Performance instrumentation
  - `activation_plot.py`: Activation distribution visualization
  - `figures.py`: Campaign result plotting
  - `final_test.py`: Post-campaign verification

- **`analyze_bundle.py`**: Analysis script for extracting summary statistics from campaign outputs
  - Recomputes descriptive tables and figures
  - No GPU or checkpoints required (metrics-only operation)
  - Outputs CSV tables and comparison figures for paper submission

## Usage

To recompute analysis from baseline campaign outputs:

```bash
python core/harness/analyze_bundle.py --root /path/to/baseline_campaigns/output --out /path/to/paper_analysis
```

Output goes to `paper_analysis/` with:
- `endpoint_tail_metrics.csv`: Tail statistics per arm and seed
- `recipe_quadratic_tail.csv`: Regularized GELU x norm interaction tails
- Figure PDFs and PNGs for paper integration

## Code Review Context

This harness was frozen at the time campaigns completed (2026-09-17). It represents the exact implementation used to produce published numbers in the P1 and baseline sections. Key invariants verified on retrieval:

1. **Substitution arms (A-G in 5-seed CIFAR-10)**: Config E collapse (37.06 ± 18.11% vs. 77.00% teacher) and variance reduction with fixes
2. **Regularized GELU x norm contrast (P1)**: −2.95 pp accuracy, tail stats (max NLL 2.51e8 under BatchNorm vs. 5.2-5.5 under LayerNorm)
3. **Plaintext baseline (pvit_lab)**: Legacy recipe (teacher 76.54 ± 3.86%) vs. regularized recipe (BatchNorm edge shrinks +4.61 pp → +0.85 pp)

**Do not modify the harness code to refit or re-run experiments.** New experiments belong in new scripts in the main repo; the harness is a frozen reference for auditability.

---

*Retrieved from: `/tmp/verify_bundle/pvit_evidence_final_20260917T104757Z/baseline_campaigns_stripped_20260917T104200Z.tgz`*
*Sourced by: an LLM-assisted lookup (unverified), 2026-09-20*
