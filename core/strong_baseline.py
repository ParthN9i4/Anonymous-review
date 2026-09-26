"""
strong_baseline.py
=========================
A genuine plaintext baseline for the thesis ViT. Added 2026-09-10.

WHY THIS EXISTS
---------------
The teacher in substitution_ablation.py / verify_fixes.py reaches only
77.00 +/- 2.86% on CIFAR-10, and docs/PROJECT_RULES.md already concedes this is
undertrained versus ~90%+ for a well-trained DeiT-Tiny. Every "Fix 2
recovers the teacher" claim is therefore anchored to a weak reference,
which is the first thing a reviewer attacks.

DIAGNOSIS (from log_subs.txt, the teacher's own curves) -- the problem is
NOT too few epochs, it is overfitting:

  seed 42:  ep60 loss 0.242 -> ep100 loss 0.018 (-93%), acc 71.04 -> 71.83
  seed 43:  ep60 loss 0.177 -> ep100 loss 0.013 (-92%), acc 76.44 -> 77.15
  seed 44:  ep60 loss 0.122 -> ep100 loss 0.007 (-95%), acc 77.28 -> 78.84
  seed 45:  ep60 loss 0.129 -> ep100 loss 0.007 (-95%), acc 77.79 -> 79.25
  seed 46:  ep60 loss 0.158 -> ep100 loss 0.009 (-94%), acc 75.62 -> 76.99

Training loss falls ~95% over the last 40 epochs while validation accuracy
gains ~1 point: the model memorises the training set. More epochs cannot
help; the missing ingredient is regularisation.

WHAT THE OLD RECIPE HAD          WHAT IT WAS MISSING
  RandomCrop(32, pad=4)            RandAugment
  RandomHorizontalFlip             mixup / CutMix
  AdamW lr=1e-3 wd=0.05            label smoothing
  CosineAnnealingLR                LR warmup (ViTs are very sensitive)
  100 epochs                       stochastic depth (drop-path)
                                   random erasing
                                   longer schedule

This script implements the missing half, following the DeiT recipe
(Touvron et al., 2021) adapted to 32x32. Every component is switchable
from the CLI so each one's contribution can be ablated rather than
asserted.

WHY IT MATTERS BEYOND A NICER NUMBER
------------------------------------
The paper's central claim is about training dynamics under substitution.
A reviewer can currently ask: "does your interaction collapse survive a
properly regularised training regime, or is it an artefact of an
under-regularised one?" That question is unanswered today. This script
makes it answerable, because --poly-gelu/--poly-softmax/--poly-norm let
the SAME strong recipe be applied to any substitution configuration.

Both outcomes are publishable:
  * collapse persists under the strong recipe -> the finding is much
    stronger and immune to the "weak baseline" critique;
  * collapse disappears -> that is itself an important, honest result
    about when the failure mode does and does not occur.

USAGE
-----
  # Strong teacher baseline, single seed, quick look
  python core/strong_baseline.py --seeds 42 --epochs 100

  # Full strong baseline, 5 seeds, 300 epochs
  python core/strong_baseline.py --seeds 42 43 44 45 46 --epochs 300

  # Same strong recipe applied to the collapsing configuration
  python core/strong_baseline.py --seeds 42 43 44 45 46 --epochs 300 \
      --poly-gelu --poly-softmax --poly-norm --tag configE

  # Reproduce the OLD weak recipe for a controlled A/B
  python core/strong_baseline.py --seeds 42 --epochs 100 --recipe weak
"""

import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from substitution_ablation import ConfigurableDeiT, DATASET_INFO, _git_commit

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Data ─────────────────────────────────────────────────────────────

# BloodMNIST is handled separately from DATASET_INFO because it comes from
# medmnist, not torchvision, and its labels ship as shape (N, 1).
# It is the dataset that should ANCHOR the paper: it is the thesis's actual
# domain (medical imaging under CKKS), its teacher already sits at the
# published reference (94.55 +/- 0.27 here vs the ~95% ResNet-18 MedMNIST
# reports, fix2_bloodmnist.py:13), MedBlindTuner evaluates on it too, and the
# metastability signature is far stronger there than on CIFAR-10
# (Config E per-seed 73.31/82.81/18.24/90.18/92.81 -- one seed pinned near the
# 12.5% chance floor; 146x variance ratio vs CIFAR-10's 34x).
BLOODMNIST_INFO = {"num_classes": 8, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}


def dataset_meta(dataset):
    """num_classes / mean / std for any supported dataset."""
    if dataset == "bloodmnist":
        return BLOODMNIST_INFO
    return DATASET_INFO[dataset]


class SqueezeLabels(torch.utils.data.Dataset):
    """medmnist yields labels of shape (1,); the training loop wants scalars."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        if torch.is_tensor(label):
            label = label.reshape(-1)[0].long()
        else:
            label = int(np.asarray(label).reshape(-1)[0])
        return img, label


def build_transforms(dataset, recipe, randaug_n, randaug_m, erasing_p):
    """Train/eval transforms. `recipe` is 'strong' or 'weak'.

    'weak' reproduces exactly what substitution_ablation.py uses today
    (RandomCrop + HFlip only), so the two recipes can be A/B'd under
    otherwise identical conditions.

    For BloodMNIST the geometric augmentation follows fix2_bloodmnist.py's
    reasoning: blood cells in microscopy have no canonical orientation, so
    vertical flips and rotation are safe (unlike fundus images, where the
    macula anchors 'up').
    """
    info = dataset_meta(dataset)
    mean, std = info["mean"], info["std"]

    if dataset == "bloodmnist":
        eval_tf = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        if recipe == "weak":
            # exactly fix2_bloodmnist.py's existing augmentation
            return transforms.Compose([
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]), eval_tf
        ops = [
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandAugment(num_ops=randaug_n, magnitude=randaug_m),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if erasing_p > 0:
            ops.append(transforms.RandomErasing(p=erasing_p))
        return transforms.Compose(ops), eval_tf

    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if recipe == "weak":
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return train_tf, eval_tf

    # strong: DeiT-style, adapted to 32x32
    ops = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=randaug_n, magnitude=randaug_m),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if erasing_p > 0:
        # RandomErasing operates on tensors, so it goes after ToTensor/Normalize.
        ops.append(transforms.RandomErasing(p=erasing_p))
    return transforms.Compose(ops), eval_tf


def get_loaders(dataset, recipe, batch_size, val_size, split_seed,
                randaug_n, randaug_m, erasing_p, num_workers=8):
    """Same 45k/5k train/val protocol as substitution_ablation.py.

    The split seed is independent of the model seed, so every arm and every
    seed sees an identical split, and the test set is touched only at the end.
    """
    train_tf, eval_tf = build_transforms(dataset, recipe, randaug_n, randaug_m, erasing_p)

    if dataset == "bloodmnist":
        # BloodMNIST ships a REAL validation split, so we use it rather than
        # carving one out of train -- this is why docs/PROJECT_RULES.md records BloodMNIST
        # as unaffected by the test-set-selection leakage that hit CIFAR-10.
        import medmnist
        from medmnist import INFO
        DataClass = getattr(medmnist, INFO["bloodmnist"]["python_class"])
        tr = SqueezeLabels(DataClass(split="train", transform=train_tf,
                                     download=True, root="./data"))
        va = SqueezeLabels(DataClass(split="val", transform=eval_tf,
                                     download=True, root="./data"))
        te = SqueezeLabels(DataClass(split="test", transform=eval_tf,
                                     download=True, root="./data"))
        return (
            DataLoader(tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True, drop_last=True),
            DataLoader(va, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True),
            DataLoader(te, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True),
        )

    info = DATASET_INFO[dataset]
    DsCls = info["cls"]
    train_aug = DsCls('./data', train=True, download=True, transform=train_tf)
    train_cln = DsCls('./data', train=True, download=True, transform=eval_tf)
    test = DsCls('./data', train=False, download=True, transform=eval_tf)

    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(train_aug), generator=g).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    return (
        DataLoader(torch.utils.data.Subset(train_aug, train_idx), batch_size=batch_size,
                   shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True),
        DataLoader(torch.utils.data.Subset(train_cln, val_idx), batch_size=batch_size,
                   shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
    )


# ── Schedule ─────────────────────────────────────────────────────────

def lr_at(step, total_steps, warmup_steps, base_lr, min_lr):
    """Linear warmup then cosine decay, computed per-step.

    ViTs trained from scratch are notoriously sensitive to the absence of
    warmup -- the old recipe went straight to lr=1e-3 at step 0 with no
    warmup at all, which is one of the suspected contributors to the weak
    baseline (untested until the --warmup-epochs 0 ablation is run).
    """
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ── Eval ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        correct += model(imgs).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# ── Train ────────────────────────────────────────────────────────────

def train_one(args, seed, loaders, device, num_classes):
    train_loader, val_loader, test_loader = loaders
    set_seed(seed)

    model = ConfigurableDeiT(
        num_classes=num_classes,
        poly_gelu=args.poly_gelu,
        poly_softmax=args.poly_softmax,
        poly_norm=args.poly_norm,
        drop_path_rate=args.drop_path if args.recipe == "strong" else 0.0,
    ).to(device)

    # No weight decay on norms/biases/positional params -- standard for ViTs
    # and worth a few tenths of a point on its own.
    #
    # FIXED 2026-09-15. The predicate was `name.endswith(".cls_token")`, which
    # NEVER MATCHED: ConfigurableDeiT registers these as top-level parameters,
    # so named_parameters() yields the bare names "cls_token" and "pos_embed"
    # with no leading dot. The ndim guard did not catch them either -- both are
    # ndim==3, shapes (1,1,192) and (1,65,192). So both tensors silently
    # received full weight decay, i.e. exactly the opposite of what this block
    # says it does, in every run of this script to date. Verified by
    # instantiating the model and printing named_parameters().
    #
    # The names are matched exactly rather than by suffix: a suffix match on
    # "cls_token" would also catch a hypothetical "blocks.0.cls_token", which
    # would be a different parameter with different treatment warranted.
    NO_DECAY_NAMES = {"cls_token", "pos_embed"}
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name in NO_DECAY_NAMES:
            no_decay.append(p)
        else:
            decay.append(p)
    if not any(p.ndim == 3 for p in no_decay):
        # cls_token and pos_embed are the only ndim==3 parameters in this model.
        # If neither landed in no_decay the predicate has silently broken again.
        raise RuntimeError(
            "weight-decay grouping did not exclude cls_token/pos_embed; "
            f"model parameter names are {[n for n, _ in model.named_parameters()][:4]}")
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.999))

    mixup_fn = None
    if args.recipe == "strong" and (args.mixup > 0 or args.cutmix > 0):
        from timm.data import Mixup
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            label_smoothing=args.smoothing, num_classes=num_classes)

    if mixup_fn is not None:
        from timm.loss import SoftTargetCrossEntropy
        criterion = SoftTargetCrossEntropy()          # mixup emits soft targets
    elif args.recipe == "strong" and args.smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    best_val, best_state, best_epoch = -1.0, None, -1
    hist = []
    step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        running, nb = 0.0, 0
        for imgs, labels in train_loader:
            lr = lr_at(step, total_steps, warmup_steps, args.lr, args.min_lr)
            for gparam in optimizer.param_groups:
                gparam["lr"] = lr

            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if mixup_fn is not None:
                imgs, targets = mixup_fn(imgs, labels)
            else:
                targets = labels

            loss = criterion(model(imgs), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
            optimizer.step()
            running += loss.item()
            nb += 1
            step += 1

        val_acc = evaluate(model, val_loader, device)
        train_loss = running / max(nb, 1)
        hist.append({"epoch": epoch + 1, "train_loss": train_loss,
                     "val_acc": val_acc, "lr": lr})
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % args.print_every == 0 or epoch == 0:
            print(f"    ep {epoch+1:3d}/{args.epochs}  loss={train_loss:.4f}  "
                  f"val={val_acc:.2f}%  best={best_val:.2f}%  lr={lr:.2e}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    test_final = evaluate(model, test_loader, device)
    if best_state is not None:
        model.load_state_dict(best_state)
    test_at_best_val = evaluate(model, test_loader, device)

    # Persist the selected checkpoint. ADDED 2026-09-15: `best_state` was held
    # in memory and never written -- there was no torch.save anywhere in this
    # file -- so every run of this script produced numbers whose model could
    # not afterwards be loaded, evaluated in P-mode, or have its ranges
    # censused. A baseline that cannot be re-examined is not a baseline.
    # Same payload shape as substitution_ablation.py so the downstream
    # checkpoint tooling reads both without a special case.
    checkpoint_written = None
    if args.save_checkpoints and best_state is not None:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        checkpoint_written = os.path.join(
            args.ckpt_dir,
            f"strong_baseline_{args.dataset}_{args.recipe}_seed{seed}.pth")
        tmp_ckpt = checkpoint_written + ".tmp"
        torch.save({
            "state_dict": best_state,
            "label": f"strong_baseline/{args.recipe}",
            "checkpoint_rule": "best VALIDATION accuracy observed during training",
            "seed": seed,
            "git_commit": _git_commit(),
            "metadata": {
                "dataset": args.dataset,
                "recipe": args.recipe,
                "epochs": args.epochs,
                "poly_gelu": args.poly_gelu,
                "poly_softmax": args.poly_softmax,
                "poly_norm": args.poly_norm,
                "best_val": best_val,
                "best_epoch": best_epoch + 1,
                "test_at_best_val": test_at_best_val,
                "test_final": test_final,
            },
        }, tmp_ckpt)
        os.replace(tmp_ckpt, checkpoint_written)

    # The overfitting signature this script exists to fix: if the final
    # train loss is ~0 while the val/test gap stays large, regularisation
    # is still insufficient regardless of how long it ran.
    final_train_loss = hist[-1]["train_loss"]
    print(f"  > test@best-val={test_at_best_val:.2f}%  test@final={test_final:.2f}%  "
          f"best_val={best_val:.2f}% (ep {best_epoch+1})  "
          f"final_train_loss={final_train_loss:.4f}", flush=True)

    return {
        "seed": seed,
        "test_at_best_val": test_at_best_val,
        "test_final": test_final,
        "best_val": best_val,
        "best_epoch": best_epoch + 1,
        "checkpoint_path": checkpoint_written,
        "final_train_loss": final_train_loss,
        # `generalization_gap_pp` was `best_val - 100*exp(-final_train_loss)`.
        # REMOVED 2026-09-15: exp(-CE) is the geometric mean of the predicted
        # probability assigned to the true class, NOT training accuracy, so
        # subtracting it from a percentage compares two different quantities
        # and the result is not a generalization gap in percentage points.
        # The honest version needs measured train accuracy; until this script
        # evaluates the training split, the raw loss is reported on its own and
        # the reader draws their own conclusion.
        "final_train_loss_note": (
            "exp(-loss) is NOT an accuracy surrogate; the former "
            "generalization_gap_pp field derived from it was invalid and is "
            "removed. Measure train accuracy directly if a gap is needed."),
        "history": hist,
        "seconds": time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser(
        description="Strong plaintext baseline for the thesis ViT (CIFAR-10/100).")
    p.add_argument("--dataset", default="cifar10",
                   choices=list(DATASET_INFO.keys()) + ["bloodmnist"],
                   help="bloodmnist is the paper-anchoring dataset; the CIFARs "
                        "are non-medical controls.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--recipe", choices=["strong", "weak"], default="strong",
                   help="'weak' reproduces the current substitution_ablation.py "
                        "recipe exactly, for a controlled A/B.")
    # regularisation knobs -- each individually ablatable
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.8)
    p.add_argument("--cutmix", type=float, default=1.0)
    p.add_argument("--randaug-n", type=int, default=2)
    p.add_argument("--randaug-m", type=int, default=9)
    p.add_argument("--erasing-p", type=float, default=0.25)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--save-checkpoints", action="store_true", default=True,
                   help="Write the best-validation checkpoint (default on). "
                        "Before 2026-09-15 this script saved nothing, so its "
                        "baselines could never be re-evaluated afterwards.")
    p.add_argument("--no-save-checkpoints", dest="save_checkpoints",
                   action="store_false")
    p.add_argument("--ckpt-dir", default="./checkpoints_strong_baseline")
    # substitutions -- lets the SAME strong recipe be applied to any config
    p.add_argument("--poly-gelu", action="store_true")
    p.add_argument("--poly-softmax", action="store_true")
    p.add_argument("--poly-norm", action="store_true")
    p.add_argument("--tag", default="teacher",
                   help="Label for the output filename, e.g. 'teacher' or 'configE'.")
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--split-seed", type=int, default=1234)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--results-dir", default="./results/ablations")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = dataset_meta(args.dataset)["num_classes"]
    subs = [n for n, on in [("GELU", args.poly_gelu), ("softmax", args.poly_softmax),
                            ("norm", args.poly_norm)] if on]

    print(f"Device        : {device}")
    print(f"Dataset       : {args.dataset} ({num_classes} classes)")
    print(f"Recipe        : {args.recipe}")
    print(f"Substitutions : {subs if subs else 'none (standard teacher)'}")
    print(f"Seeds         : {args.seeds}   Epochs: {args.epochs}   Batch: {args.batch_size}")
    if args.recipe == "strong":
        print(f"  warmup={args.warmup_epochs}ep  smoothing={args.smoothing}  "
              f"mixup={args.mixup}  cutmix={args.cutmix}  "
              f"randaug=({args.randaug_n},{args.randaug_m})  "
              f"erasing={args.erasing_p}  drop_path={args.drop_path}")
    t_start = time.time()

    loaders = get_loaders(args.dataset, args.recipe, args.batch_size,
                          args.val_size, args.split_seed,
                          args.randaug_n, args.randaug_m, args.erasing_p,
                          num_workers=args.num_workers)

    per_seed = {}
    for seed in args.seeds:
        print(f"\n{'='*66}\nSEED {seed}\n{'='*66}", flush=True)
        per_seed[seed] = train_one(args, seed, loaders, device, num_classes)
        # Checkpoint after every seed: 300-epoch runs against a SLURM wall,
        # and a timeout must not discard finished seeds.
        try:
            os.makedirs(args.results_dir, exist_ok=True)
            _p = os.path.join(
                args.results_dir,
                f"strong_baseline_{args.dataset}_{args.recipe}_{args.tag}.PARTIAL.json")
            with open(_p + ".tmp", "w") as _f:
                json.dump({"STATUS": "PARTIAL", "seeds_done": sorted(per_seed),
                           "args": vars(args), "per_seed": per_seed},
                          _f, indent=2, default=float)
            os.replace(_p + ".tmp", _p)
        except Exception as _e:
            print(f"  [WARN] partial-save failed (continuing): {_e}", flush=True)

    accs = np.array([per_seed[s]["test_at_best_val"] for s in sorted(per_seed)])
    losses = np.array([per_seed[s]["final_train_loss"] for s in sorted(per_seed)])
    summary = {
        "mean": float(accs.mean()),
        "std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "min": float(accs.min()), "max": float(accs.max()),
        "per_seed": {str(s): per_seed[s]["test_at_best_val"] for s in sorted(per_seed)},
        "final_train_loss_mean": float(losses.mean()),
        "n_seeds": int(len(accs)),
    }

    print(f"\n{'='*66}")
    print(f"RESULT  {args.dataset} / recipe={args.recipe} / tag={args.tag}")
    print(f"{'='*66}")
    print(f"  test acc : {summary['mean']:.2f}% +/- {summary['std']:.2f}%   "
          f"[{summary['min']:.2f}, {summary['max']:.2f}]")
    print(f"  final train loss (mean): {summary['final_train_loss_mean']:.4f}")
    if summary["final_train_loss_mean"] < 0.05:
        print("  [WARN] final train loss ~0 -- still memorising the training set.")
        print("         More epochs will not help; increase regularisation.")
    print(f"  Reference: old weak recipe on CIFAR-10 = 77.00 +/- 2.86% "
          f"(log_subs.txt, 5 seeds, 100 epochs)")

    os.makedirs(args.results_dir, exist_ok=True)
    seed_tag = "_".join(str(s) for s in args.seeds)
    out = os.path.join(
        args.results_dir,
        f"strong_baseline_{args.dataset}_{args.recipe}_{args.tag}_seeds{seed_tag}.json")
    with open(out, "w") as f:
        json.dump({"args": vars(args), "per_seed": per_seed, "summary": summary,
                   "substitutions": subs,
                   "wall_clock_seconds": time.time() - t_start}, f, indent=2, default=float)
    print(f"\nResults JSON: {out}")


if __name__ == "__main__":
    main()
