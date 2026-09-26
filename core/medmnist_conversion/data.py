"""MedMNIST splits shaped to the contract that pvit_lab.deploy expects.

WHY THIS FILE EXISTS
--------------------
`harness/pvit_lab/data.py:datasets()` resolves its dataset through

    cls = {'cifar10': D.CIFAR10, 'cifar100': D.CIFAR100}[name]

and then carves a validation split with `torch.randperm(50000)`. Both the
lookup and the 50,000 are hard-coded, so the frozen-conversion pipeline -- the
one that produced the 86.27% -> 25.07% result -- cannot run on any medical
dataset as written. `PROVENANCE.md` declares that harness a frozen reference
that must not be edited for new experiments, so this package supplies a
replacement rather than patching it.

THE CONTRACT WE MUST SATISFY
----------------------------
`deploy.run()` does exactly this::

    _, val, clean = datasets(data_root, dataset, synthetic=synthetic)
    ca = loader(clean, workers=workers, limit=2048)   # calibration fitting
    va = loader(val,   workers=workers)               # held-out audit

so the three returned objects are, in order:

    [0] training split WITH augmentation   (unused by deploy, used by trainers)
    [1] held-out audit split, clean transform
    [2] training split, clean transform -- polynomial domains are fitted from
        its first 2,048 images

Each must yield ``(x, y, index)``; `pvit_lab.data.Indexed` provides that and is
reused here unchanged so image IDs stay comparable across runs.

HOW THIS DIFFERS FROM THE CIFAR PATH, DELIBERATELY
--------------------------------------------------
CIFAR-10 has no official validation split, so the frozen harness carves 5,000
images out of train with a fixed permutation. MedMNIST ships official
train/val/test splits, so we use them. That is a *better* separation, not an
equivalent one: do not describe a MedMNIST conversion and a CIFAR-10 conversion
as having had identical split protocols.

Transforms follow `fix2_bloodmnist.py:337` exactly -- resize 28 -> 32, normalise
to mean/std 0.5 -- so accuracies here are comparable with the existing MedMNIST
campaigns in this repository. Do not "improve" them without renaming the
experiment.

NOT VERIFIED BY EXECUTION: this module was written in an environment without
torch, torchvision or medmnist installed. It is syntax-checked only. Run
`python -m medmnist_conversion.data --self-test --dataset bloodmnist` on a
machine with the dependencies before trusting any number produced through it.
"""
from __future__ import annotations

import argparse
import sys

import torch
from torchvision import transforms as T

# Reused unchanged from the frozen harness so indices mean the same thing.
from pvit_lab.data import Indexed


class _Spec:
    """Per-dataset facts that change the correct augmentation or shape."""

    def __init__(self, key, n_classes, channels, vertical_flip, rotation, note):
        self.key = key
        self.n_classes = n_classes
        self.channels = channels
        self.vertical_flip = vertical_flip
        self.rotation = rotation
        self.note = note


# Orientation decisions copied from the reasoning already recorded in
# fix2_bloodmnist.py:337 and fix2_retinamnist.py. They are not cosmetic: a
# vertical flip on a fundus image destroys the macula/optic-disc relationship
# that the label depends on.
SPECS = {
    "bloodmnist": _Spec("bloodmnist", 8, 3, True, 15,
                        "microscopy; no canonical orientation, all flips safe"),
    "pathmnist": _Spec("pathmnist", 9, 3, True, 15,
                       "histopathology tiles; no canonical orientation"),
    "dermamnist": _Spec("dermamnist", 7, 3, True, 15,
                        "dermatoscopy; lesion orientation is not label-bearing"),
    "retinamnist": _Spec("retinamnist", 5, 3, False, 10,
                         "fundus; vertical flip NOT safe, macula anchors 'up'"),
    "pneumoniamnist": _Spec("pneumoniamnist", 2, 1, False, 10,
                            "chest X-ray; grayscale, left/right asymmetry matters"),
    "breastmnist": _Spec("breastmnist", 2, 1, False, 10,
                         "breast ultrasound; grayscale, very small train split"),
}


class _SqueezeLabel(torch.utils.data.Dataset):
    """MedMNIST yields labels of shape (1,); the harness expects a scalar.

    Without this the label reaches the loss as a length-1 array and either
    raises or silently broadcasts, depending on the call site. Squeezing at the
    dataset boundary keeps every downstream consumer identical to the CIFAR
    path.
    """

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, int(y[0]) if hasattr(y, "__len__") else int(y)


def _transforms(spec, train):
    """Resize 28 -> 32 so the 4x4 patch grid still yields 65 tokens."""
    steps = [T.Resize(32)]
    if train:
        steps.append(T.RandomHorizontalFlip(p=0.5))
        if spec.vertical_flip:
            steps.append(T.RandomVerticalFlip(p=0.5))
        steps.append(T.RandomRotation(degrees=spec.rotation))
        if spec.channels == 3:
            steps.append(T.ColorJitter(brightness=0.1, contrast=0.1,
                                       saturation=0.1))
    steps.append(T.ToTensor())
    # Single-channel scans are replicated to three channels: the checkpoint's
    # patch embedding has in_channels=3 and is not re-shaped here.
    if spec.channels == 1:
        steps.append(T.Lambda(lambda t: t.expand(3, -1, -1)))
    steps.append(T.Normalize(mean=[0.5] * 3, std=[0.5] * 3))
    return T.Compose(steps)


def datasets(root, name="bloodmnist", regularized=False, download=False,
             synthetic=False, audit_split="val"):
    """Drop-in replacement for pvit_lab.data.datasets, for MedMNIST.

    `regularized` is accepted for signature compatibility and ignored: the
    MedMNIST augmentation above is already the recipe used by the existing
    campaigns, and silently switching to RandAugment would make these runs
    incomparable with them.

    `audit_split` selects what lands in slot [1]. 'val' is the default and the
    only correct choice while any decision is still open. 'test' is available
    ONLY for a final locked evaluation and will print a warning, because
    reaching for it early is how the CIFAR-10 test set in this project became
    unusable (docs/P1_AUDIT.md).
    """
    if synthetic:
        g = torch.Generator().manual_seed(1234)
        ds = torch.utils.data.TensorDataset(
            torch.randn(32, 3, 32, 32, generator=g),
            torch.randint(0, 8, (32,), generator=g))
        return (Indexed(ds, range(8, 32)), Indexed(ds, range(8)),
                Indexed(ds, range(8, 32)))

    if name not in SPECS:
        raise ValueError(
            f"{name!r} is not a MedMNIST dataset this adapter knows. "
            f"Known: {sorted(SPECS)}. For CIFAR use pvit_lab.data.datasets.")
    spec = SPECS[name]

    import medmnist
    from medmnist import INFO

    cls = getattr(medmnist, INFO[name]["python_class"])

    if audit_split == "test":
        print("WARNING: audit_split='test' opens the held-out test split. Only "
              "do this for a final evaluation after every choice is frozen.",
              file=sys.stderr)

    train_aug = cls(split="train", transform=_transforms(spec, True),
                    download=download, root=root)
    train_clean = cls(split="train", transform=_transforms(spec, False),
                      download=download, root=root)
    audit = cls(split=audit_split, transform=_transforms(spec, False),
                download=download, root=root)

    train_aug, train_clean, audit = (_SqueezeLabel(d)
                                     for d in (train_aug, train_clean, audit))

    # Identity index maps: MedMNIST's on-disk order is already fixed, so an
    # image's index is stable across runs without a stored permutation.
    return (Indexed(train_aug, range(len(train_aug))),
            Indexed(audit, range(len(audit))),
            Indexed(train_clean, range(len(train_clean))))


def _self_test(name, root, download):
    """Prove the contract holds before anyone runs a conversion through it."""
    spec = SPECS[name]
    train, audit, clean = datasets(root, name, download=download)
    print(f"{name}: {spec.note}")
    print(f"  train {len(train)}  audit {len(audit)}  clean {len(clean)}")

    problems = []
    if len(clean) != len(train):
        problems.append("slot [2] must be the training split, clean transform")
    if len(audit) >= len(train):
        problems.append("audit split unexpectedly large -- check split names")

    for label, ds in (("train", train), ("audit", audit), ("clean", clean)):
        x, y, j = ds[0]
        print(f"  {label:5s} x{tuple(x.shape)} dtype={x.dtype} "
              f"y={y} ({type(y).__name__}) index={j} "
              f"range=[{x.min():.3f}, {x.max():.3f}]")
        if tuple(x.shape) != (3, 32, 32):
            problems.append(f"{label}: expected (3,32,32), got {tuple(x.shape)}")
        if not isinstance(y, int):
            problems.append(f"{label}: label must be a python int, got {type(y)}")
        if not 0 <= y < spec.n_classes:
            problems.append(f"{label}: label {y} outside 0..{spec.n_classes - 1}")
        if j != 0:
            problems.append(f"{label}: first index should be 0, got {j}")

    # The calibration set deploy.py actually fits on is clean[:2048].
    if len(clean) < 2048:
        problems.append(f"only {len(clean)} clean images; deploy.py fits domains "
                        f"on 2048 and will silently use fewer")

    # Determinism: the clean transform must not be stochastic, or the fitted
    # domains change between the census pass and evaluation.
    a, b = clean[7][0], clean[7][0]
    if not torch.equal(a, b):
        problems.append("clean transform is non-deterministic -- fitted domains "
                        "would not be reproducible")

    print()
    if problems:
        for p in problems:
            print(f"  FAIL {p}")
        return 1
    print("  contract satisfied")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dataset", default="bloodmnist", choices=sorted(SPECS))
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    if not a.self_test:
        ap.error("nothing to do; pass --self-test")
    sys.exit(_self_test(a.dataset, a.data_root, a.download))
