# Private ViT under CKKS: diagnosing where HE-friendly conversion loses accuracy

Research code and evidence for a measurement study of **HE-friendly Vision Transformer
conversion**. A compact ViT's non-polynomial operations (GELU, softmax attention,
LayerNorm) are replaced by polynomial arithmetic that CKKS can evaluate. The study then
measures where accuracy is lost:

1. **Training with substitutions**: a 2×2×2 factorial over activation, attention and
   normalization replacements (CIFAR-10, five seeds; MedMNIST replication).
2. **Fixed-weight post-training conversion**: learned weights frozen, operators replaced by
   fitted polynomials (budget / accurate profiles, four operator masks, three seeds).
3. **Primitive-level CKKS**: the deployed polynomials evaluated under CKKS (TenSEAL) and
   compared with their plaintext evaluation (O-P-C).

This is a **measurement and failure-diagnosis** project, not a deployable private-inference
system.

## Evidence status: read before citing any number

| Statement | Status |
|---|---|
| Continuously encrypted trained ViT, block or backbone | **Not done.** Encrypted evidence is primitive-level only (`results/campaign_2026-09-20/`). A full context exceeded the 2 GB serialization limit; a reduced configuration exhausted its scale at the first feed-forward projection. |
| `experiments/active/measure_encrypted_attention.py` | **Component-wise.** It decrypts between stages, so it is not an end-to-end circuit. |
| OpenFHE port (`experiments/active/openfhe/`) | Classification head only, random-init weights. |
| Headline conversion result 86.27% → 25.07% | **CIFAR-10, budget profile, GELU-only mask**, three seeds, clean test split (`paper/analysis_20260920/conversion_fidelity.csv`). It is one cell, not the study mean. |
| CIFAR-10 substitution numbers in `results/ablations/substitution_ablation_seeds42_43_44_45_46.json` | Test-set-selected (S1). Use the validation-selected `clean8` / `paper/analysis_20260918/` numbers. |
| Model identity | Custom **six-block** ViT, width 192, 3 heads, patch 4, 32×32 input, 65 tokens, **2.69M parameters**. Not stock DeiT-Tiny. |
| MedMNIST | Benchmark replication, not clinical validation. |

Detailed ground truth, corrections and claim rules are in [`docs/PROJECT_RULES.md`](docs/PROJECT_RULES.md).

## Repository map

| Path | Contents |
|---|---|
| `core/` | Maintained implementation. Thesis model and substitutions (`core/fix2_bloodmnist.py`), factorial trainers (`core/substitution_ablation*.py`), `core/strong_baseline.py`, the TenSEAL head (`core/ckks_classification_head.py`), the `harness/pvit_lab` package and the MedMNIST conversion adapter. |
| `experiments/active/` | Scripts behind current paper claims: fix verification, Gate 1, CKKS primitive probes, BN-fold checks, factorial analysis, smoke tests, OpenFHE head, and the pre-registered `closeout_v2/` campaign (attribution, dose-response, screens, encrypted MLP sub-block). |
| `experiments/exploratory/` | Diagnostics and side studies (collapse investigation, LUT sweep, smooth transition, KD variants). |
| `results/` | Evidence, grouped by campaign. See [`results/README.md`](results/README.md). |
| `paper/` | SaTML 2027 manuscript, analysis CSVs and number verifiers. |
| `infrastructure/` | SLURM scripts, H200 run-books, `env.sh`. |
| `tools/` | Result audit/merge/recovery, status companion, fast checks, path checker. |
| `tests/` | Dependency-free regression tests. |
| `docs/` | Rules, status audits, design notes, operations, talks. Start at [`docs/INDEX.md`](docs/INDEX.md). |
| `archive/` | Frozen history: the Family-A stock-timm step pipeline with its results, logs, handoffs, backups. |

Paths changed on 2026-09-23. [`docs/PATH_MIGRATION.md`](docs/PATH_MIGRATION.md) maps every old path to its new one.

## Reproducing

```bash
conda env create -f environment.yml     # or: pip install -r requirements.txt (Python 3.10)
source infrastructure/env.sh            # puts core/ and the repo root on PYTHONPATH; cd to root
python tools/fast_check.py              # syntax, shell, path/citation checks, unit tests (no ML deps)

# CPU smoke tests before any GPU job
python experiments/active/smoke_test_substitution_ablation.py
python experiments/active/smoke_test_strong_baseline.py

# Paper numbers are re-derived from committed CSVs
python paper/verify_abstract_numbers.py
```

GPU campaigns run under SLURM from the repository root, e.g.
`sbatch infrastructure/slurm/cifar_array.sbatch`. See [`infrastructure/RERUN_PLAN.md`](infrastructure/RERUN_PLAN.md).
The frozen conversion and CKKS campaign has its own launcher and hashes in
`results/campaign_2026-09-20/source_final/`.

Trained checkpoints are not stored in git. Their paths and SHA-256 hashes are listed in
`results/campaign_2026-09-20/source_final/checkpoint_inventory.json`.
