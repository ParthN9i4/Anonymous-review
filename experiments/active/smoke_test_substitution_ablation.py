"""
smoke_test_substitution_ablation.py
====================================
Offline logic smoke test for substitution_ablation.py. Added 2026-09-09
alongside the F/G configs and CIFAR-100 support, after finding (by static
reading) that the committed leakage fix's JSON payload construction called
float() on a dict and would have crashed at the very last line of a real
run -- after every training run finished.

This does NOT touch the network: torchvision's CIFAR-10/CIFAR-100 downloads
are unreachable from some sandboxed environments (e.g. behind an egress
proxy), so it drives the actual imported model/training code
(ConfigurableDeiT, train_model, and the exact JSON payload construction from
main()) against synthetic in-memory data of the right shape instead. It
proves the code runs and serializes correctly for every CONFIG (including
F/G) at both num_classes=10 and num_classes=100 -- it does NOT validate real
data, CUDA-specific code paths, training dynamics, or timing. Run the real
thing on the A6000 before trusting any accuracy number.

Usage: python experiments/active/smoke_test_substitution_ablation.py
"""
import json
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

import substitution_ablation as sa


def fake_loader(n, num_classes, batch_size=8):
    imgs = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(imgs, labels), batch_size=batch_size, shuffle=True)


def run_for_dataset(dataset_name, num_classes):
    print(f"\n{'='*60}\n{dataset_name} (num_classes={num_classes})\n{'='*60}")
    train_loader = fake_loader(32, num_classes)
    val_loader = fake_loader(16, num_classes)
    test_loader = fake_loader(16, num_classes)
    device = torch.device("cpu")

    teacher = sa.ConfigurableDeiT(num_classes=num_classes)
    teacher_res = sa.train_model(
        teacher, train_loader, val_loader, test_loader, device, epochs=1,
        label=f"Teacher [{dataset_name}]")
    assert set(teacher_res.keys()) == {"test_at_best_val", "test_final", "best_val", "best_epoch"}, teacher_res

    teacher_per_seed = {42: teacher_res}
    kd_per_seed = {name: {} for name, _, _, _ in sa.CONFIGS}

    for name, pg, ps, pn in sa.CONFIGS:
        student = sa.ConfigurableDeiT(num_classes=num_classes, poly_gelu=pg, poly_softmax=ps, poly_norm=pn)
        with torch.no_grad():
            out = student(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, num_classes), f"{name}: bad output shape {out.shape}"
        res = sa.train_model(
            student, train_loader, val_loader, test_loader, device, epochs=1,
            teacher=teacher, label=f"{name} [{dataset_name}]")
        kd_per_seed[name][42] = res
        print(f"  {name:<20} shape OK, result OK: test_at_best_val={res['test_at_best_val']:.2f}")

    # Reproduce the exact payload construction from main() -- this is the
    # code path that had the float(dict) bug (fixed 2026-09-09).
    payload = {
        "teacher_per_seed": {str(k): v for k, v in teacher_per_seed.items()},
        "with_kd_per_seed": {nm: {str(k): v for k, v in d.items()}
                             for nm, d in kd_per_seed.items()},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f, indent=2, default=float)
        out_path = f.name
    with open(out_path) as f:
        reloaded = json.load(f)
    assert reloaded["teacher_per_seed"]["42"]["test_at_best_val"] == teacher_res["test_at_best_val"]
    print(f"  JSON payload construction + round-trip OK ({out_path})")


if __name__ == "__main__":
    run_for_dataset("cifar10", 10)
    run_for_dataset("cifar100", 100)
    print("\nALL SMOKE CHECKS PASSED (synthetic data -- run on the A6000 with "
          "real data before trusting any accuracy number)")
