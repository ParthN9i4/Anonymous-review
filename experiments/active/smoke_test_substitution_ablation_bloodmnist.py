"""
smoke_test_substitution_ablation_bloodmnist.py
================================================
Offline logic smoke test for substitution_ablation_bloodmnist.py. Added
2026-09-09 alongside that script.

MedMNIST's real download is unreachable from some sandboxed environments
(e.g. behind an egress proxy), so this drives the actual imported
fix2_bloodmnist.train_one() (bit-identical to what runs on the A6000) plus
this script's own CONFIGS dict, summarize_results(), print_summary(), and
JSON serialization, against synthetic BloodMNIST-shaped data (labels shaped
(N,1) int, matching medmnist's convention, since train_one does
`labels.squeeze(-1).long()`). It does NOT validate real data, CUDA-specific
code paths, training dynamics, or timing. Run the real thing on the A6000
before trusting any accuracy number.

Usage: python experiments/active/smoke_test_substitution_ablation_bloodmnist.py
"""
import json
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

import substitution_ablation_bloodmnist as sab


def fake_loader(n, num_classes=8, batch_size=8):
    imgs = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, num_classes, (n, 1))  # medmnist label shape
    return DataLoader(TensorDataset(imgs, labels), batch_size=batch_size, shuffle=True)


def main():
    train_loader = fake_loader(32)
    val_loader = fake_loader(16)
    test_loader = fake_loader(16)
    device = "cpu"
    ckpt_dir = tempfile.mkdtemp(prefix="smoke_ckpt_blood_")

    print(f"CONFIGS: {list(sab.CONFIGS.keys())}")
    assert len(sab.CONFIGS) == 8, "expected Teacher + 7 substitution arms"
    assert len(sab.STUDENT_CONFIGS) == 7

    all_results = {"Teacher": []}
    teacher_res = sab.train_one(
        "Teacher", sab.CONFIGS["Teacher"], seed=42,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        teacher_model=None, epochs=1, device=device, ckpt_dir=ckpt_dir, print_every=1)
    all_results["Teacher"].append(teacher_res)

    teacher_model, teacher_obj = sab.load_checkpoint(teacher_res["ckpt_path"], device)
    print(f"load_checkpoint round-trip OK, test_acc={teacher_obj['test_acc']:.4f}")

    for cfg_name in sab.STUDENT_CONFIGS:
        res = sab.train_one(
            cfg_name, sab.CONFIGS[cfg_name], seed=42,
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            teacher_model=teacher_model, epochs=1, device=device, ckpt_dir=ckpt_dir, print_every=1)
        all_results[cfg_name] = [res]
        print(f"  {cfg_name:<20} OK: test_acc={res['test_acc']:.4f} test_bal_acc={res['test_bal_acc']:.4f}")

    # summarize_results/Welch's t-test need >=2 seeds per config; duplicate
    # the single fake run to exercise that aggregation path (values don't
    # matter here, only that the code doesn't crash).
    for name in all_results:
        all_results[name].append(dict(all_results[name][0]))

    summary = sab.summarize_results(all_results)
    sab.print_summary(summary)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"results": all_results, "summary": summary}, f, indent=2, default=float)
        out_path = f.name
    with open(out_path) as f:
        reloaded = json.load(f)
    assert reloaded["summary"]["per_config"]["Teacher"]["n_seeds"] == 2
    print(f"\nJSON round-trip OK ({out_path})")
    print("\nALL SMOKE CHECKS PASSED (synthetic data -- run on the A6000 with "
          "real data before trusting any accuracy number)")


if __name__ == "__main__":
    main()
