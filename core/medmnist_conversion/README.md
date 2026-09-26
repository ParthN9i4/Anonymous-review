# MedMNIST adapter for the frozen conversion harness

## Why this exists

The frozen-conversion pipeline produces the paper's headline result (86.27% ->
25.07% under a degree-seven GELU fit). It cannot run on any medical dataset as
written. `core/harness/pvit_lab/data.py:28` resolves the dataset through a literal
dict:

```python
cls = {'cifar10': D.CIFAR10, 'cifar100': D.CIFAR100}[name]
...
perm = torch.randperm(50000, generator=...)
```

Both the lookup and the 50,000 are hard-coded. `paper/analysis_20260920/
PROVENANCE.md` declares `core/harness/` a frozen 2026-09-17 reference that must not
be edited for new experiments, so this package replaces the loader by
assignment instead of patching it.

## Use

```bash
# 1. Prove the adapter satisfies the contract before converting anything.
python -m medmnist_conversion.run_conversion --check --dataset bloodmnist

# 2. Convert.
python -m medmnist_conversion.run_conversion \
    --checkpoint /path/to/bloodmnist_seed42.pt \
    --dataset bloodmnist --mask gelu --profile budget \
    --out results/conversion_blood_s42_budget_gelu --data-root ./data
```

`--profile budget` is (degree 7, reciprocal k=2); `accurate` is (15, 4), per
the APPROXIMATIONS note of the H200 submission repository (not mirrored here;
the same profiles are defined in `results/campaign_2026-09-20/source_final/polynomial.py`). The two profiles each bundle polynomial degree AND
iteration count, so a difference between them is not attributable to either.

No checkpoints are stored in this repository; copy one from the training host
first. The runner says so explicitly rather than failing inside `torch.load`.

## What a result from this is, and is not

- **Same procedure as the CIFAR-10 campaign. Not the same split protocol.**
  CIFAR-10 has no official validation split, so the frozen harness carves 5,000
  images from train. MedMNIST ships official train/val/test, and this adapter
  uses them. That is a cleaner separation, not an equivalent one. Report the
  two as separate campaigns.
- **Controlled replication, not independent confirmation.** This repository has
  used MedMNIST since the Family A campaigns. New seeds on previously explored
  datasets do not create untouched inputs, and cannot discharge claim C12.
- Every run writes `adapter_provenance.json` recording both points, so the
  scope travels with the numbers.

## Which datasets are worth running

Measured from this repository: `results/*/results.json` gives a single baseline
run per dataset; the complete BloodMNIST 8x5 factorial took **6.86 GPU-hours**
(40 runs, median 655 s), so a factorial costs roughly 1.5x the baseline run x 40.

| Dataset | Classes | Baseline acc | 1 run | Est. factorial | Verdict |
|---|---|---|---|---|---|
| pathmnist | 9 | 93.87% | 2916 s | ~48 h | **run** - high baseline, 9 classes, histopathology |
| bloodmnist | 8 | 95.32% | 442 s | 6.9 h (done) | done |
| dermamnist | 7 | 76.66% | 264 s | ~4.3 h | **run** - dermatoscopy, different modality |
| pneumoniamnist | 2 | 86.86% | 183 s | ~3.0 h | marginal - binary, collapse less legible |
| retinamnist | 5 | 53.75% | 62 s | ~1.0 h | skip - baseline too weak to read degradation against |
| breastmnist | 2 | 87.18% | 45 s | ~0.7 h | skip - ~546 train images; seed SD swamps the interaction |

Adding breast, retina and pneumonia buys table rows, not evidence.

## Status

**Written without torch, torchvision or medmnist available; syntax-checked
only, never executed.** `--check` is the gate: it asserts tensor shape
(3,32,32), scalar integer labels in range, index identity, that the calibration
split holds at least the 2,048 images `deploy.py` fits domains on, and that the
clean transform is deterministic. Run it before trusting any number.
