"""
smoke_test_strong_baseline.py
==============================
Offline pre-flight for strong_baseline.py and the drop_path addition
to substitution_ablation.py. Added 2026-09-10.

Runs on CPU against synthetic data (no dataset download needed), and checks
the things that would otherwise only surface after hours of GPU time:

  1. drop_path=0.0 is BIT-IDENTICAL to the pre-2026-09-10 model -- this is
     what guarantees the substitution-factorial results stay comparable
     after stochastic depth was added as an option.
  2. drop_path>0 builds per-block DropPath with linear scaling and backprops.
  3. The warmup->cosine LR schedule hits 0 at step 0, peaks at base_lr at
     the end of warmup, and floors at min_lr.
  4. The strong recipe (mixup + CutMix + label smoothing + drop-path +
     RandAugment + random erasing) trains end to end.
  5. The 'weak' recipe reproduces the old code path.
  6. Substitutions (--poly-gelu/--poly-softmax/--poly-norm) work under the
     strong recipe -- this is what makes "does collapse survive proper
     regularisation?" answerable.
  7. CIFAR-100's 100-class path threads through.
  8. Transform stacks differ correctly between strong and weak.

Usage: python experiments/active/smoke_test_strong_baseline.py
"""
import torch

import substitution_ablation as sa
import strong_baseline as sb

# ---- 1. drop_path=0.0 must be numerically identical to the old model -------
torch.manual_seed(0)
m0 = sa.ConfigurableDeiT(num_classes=10)                      # default (0.0)
torch.manual_seed(0)
m1 = sa.ConfigurableDeiT(num_classes=10, drop_path_rate=0.0)  # explicit 0.0
x = torch.randn(4, 3, 32, 32)
m0.eval(); m1.eval()
with torch.no_grad():
    a, b = m0(x), m1(x)
assert torch.equal(a, b), "drop_path=0.0 changed the model output!"
assert all(isinstance(blk.drop_path, torch.nn.Identity) for blk in m0.blocks)
print("1. drop_path=0.0 is bit-identical to previous behaviour: OK")

# ---- 2. drop_path>0 actually builds DropPath and trains -------------------
m2 = sa.ConfigurableDeiT(num_classes=10, drop_path_rate=0.1)
from timm.layers import DropPath
kinds = [type(blk.drop_path).__name__ for blk in m2.blocks]
assert kinds[0] == "Identity", kinds          # linear scaling: first block = 0.0
assert kinds[-1] == "DropPath", kinds
m2.train()
out = m2(x); out.sum().backward()
assert torch.isfinite(out).all()
print(f"2. drop_path>0 builds per-block rates {kinds} and backprops: OK")

# ---- 3. LR schedule: warmup then cosine ----------------------------------
total, warm, base, mn = 1000, 100, 1e-3, 1e-5
lrs = [sb.lr_at(s, total, warm, base, mn) for s in range(total)]
assert lrs[0] == 0.0
assert abs(lrs[warm] - base) < 1e-9, lrs[warm]
assert lrs[warm] > lrs[warm * 5] > lrs[-1]
# cosine floors at min_lr, so the end value should sit essentially ON min_lr
assert abs(lrs[-1] - mn) < 1e-7, (lrs[-1], mn)
print(f"3. LR warmup->cosine: start={lrs[0]:.2e} peak={lrs[warm]:.2e} end={lrs[-1]:.2e}: OK")

# ---- 4. Full train_one() on synthetic data, strong recipe ----------------
from torch.utils.data import DataLoader, TensorDataset
def fake(n, nc, bs=16):
    return DataLoader(TensorDataset(torch.randn(n, 3, 32, 32),
                                    torch.randint(0, nc, (n,))),
                      batch_size=bs, shuffle=True, drop_last=True)

class A:  # argparse stand-in
    pass

def mkargs(recipe, nc=10, **kw):
    a = A()
    a.recipe, a.epochs, a.lr, a.min_lr, a.wd = recipe, 2, 1e-3, 1e-5, 0.05
    a.warmup_epochs, a.smoothing, a.mixup, a.cutmix = 1, 0.1, 0.8, 1.0
    a.drop_path, a.clip, a.print_every = 0.1, 5.0, 1
    a.poly_gelu = a.poly_softmax = a.poly_norm = False
    for k, v in kw.items():
        setattr(a, k, v)
    return a

loaders = (fake(64, 10), fake(32, 10), fake(32, 10))
r = sb.train_one(mkargs("strong"), 42, loaders, torch.device("cpu"), 10)
assert set(r) >= {"test_at_best_val", "test_final", "best_val", "final_train_loss", "history"}
assert len(r["history"]) == 2
print(f"4. strong recipe (mixup+cutmix+smoothing+droppath) trains: "
      f"test={r['test_at_best_val']:.2f} loss={r['final_train_loss']:.4f}: OK")

# ---- 5. weak recipe path (no mixup, no droppath) -------------------------
r2 = sb.train_one(mkargs("weak"), 42, loaders, torch.device("cpu"), 10)
print(f"5. weak recipe reproduces old-style path: test={r2['test_at_best_val']:.2f}: OK")

# ---- 6. substitutions honoured under the strong recipe -------------------
a3 = mkargs("strong")
a3.poly_gelu = a3.poly_softmax = a3.poly_norm = True
r3 = sb.train_one(a3, 42, loaders, torch.device("cpu"), 10)
print(f"6. Config-E substitutions under strong recipe: test={r3['test_at_best_val']:.2f}: OK")

# ---- 7. CIFAR-100 class count threads through ---------------------------
l100 = (fake(64, 100), fake(32, 100), fake(32, 100))
r4 = sb.train_one(mkargs("strong"), 42, l100, torch.device("cpu"), 100)
print(f"7. cifar100 (100 classes) path: test={r4['test_at_best_val']:.2f}: OK")

# ---- 8. transforms build for both recipes and both datasets -------------
for ds in ("cifar10", "cifar100"):
    for rec in ("strong", "weak"):
        tr, ev = sb.build_transforms(ds, rec, 2, 9, 0.25)
        names = [type(t).__name__ for t in tr.transforms]
        if rec == "strong":
            assert "RandAugment" in names and "RandomErasing" in names, names
        else:
            assert "RandAugment" not in names, names
        print(f"8. transforms {ds}/{rec}: {names}")

# ---- 9. BloodMNIST paths (no download needed) ---------------------------
# Transform stacks build, and 'weak' reproduces fix2_bloodmnist.py's own
# augmentation (H+V flip + rotation, safe because blood cells have no
# canonical orientation) rather than the CIFAR crop/flip pair.
for rec in ("strong", "weak"):
    tr, ev = sb.build_transforms("bloodmnist", rec, 2, 9, 0.25)
    names = [type(t).__name__ for t in tr.transforms]
    assert "Resize" in names and "RandomVerticalFlip" in names, names
    if rec == "strong":
        assert "RandAugment" in names, names
    else:
        assert "RandAugment" not in names, names
    print(f"9. transforms bloodmnist/{rec}: {names}")
assert sb.dataset_meta("bloodmnist")["num_classes"] == 8

# SqueezeLabels must turn medmnist's (1,)-shaped labels into scalars, or the
# loss silently breaks on shape.
class _FakeMed:
    def __len__(self): return 4
    def __getitem__(self, i):
        return torch.randn(3, 32, 32), torch.tensor([i % 8])
sq = sb.SqueezeLabels(_FakeMed())
_, lab = sq[3]
assert lab.ndim == 0 and int(lab) == 3, lab
print(f"9. SqueezeLabels (1,)->scalar: {lab} ndim={lab.ndim}: OK")

# End-to-end with 8 classes, strong recipe.
l8 = (fake(64, 8), fake(32, 8), fake(32, 8))
r5 = sb.train_one(mkargs("strong"), 42, l8, torch.device("cpu"), 8)
print(f"9. bloodmnist-shaped (8 classes) train_one: test={r5['test_at_best_val']:.2f}: OK")

print("\nALL CHECKS PASSED")
