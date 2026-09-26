"""
Stage 1 of 2 — export a classification-head fixture for the OpenFHE port.
======================================================================

Runs on WINDOWS (needs torch, optionally tenseal). Writes a dependency-free
JSON fixture that stage 2 (openfhe_head.py, runs in WSL) consumes.

Why two stages: openfhe-python ships a Linux CPython-3.12 binary, so it cannot
import on this Windows Python 3.13 install; torch/tenseal live on the Windows
side. Splitting at a JSON boundary lets each half run where it works, and has
the side benefit that the crypto script carries no ML dependencies at all.

What it exports:
  W (10x192), b (10)      -- the classification head, plaintext
  cls   (n x 192)         -- CLS tokens from a real forward pass
  plain (n x 10)          -- plaintext logits = W @ cls + b   (the ground truth)
  ts    (n x 10)          -- TenSEAL CKKS logits, if --with-tenseal (parity ref)

Usage:
  python export_head_fixture.py --max-samples 50 --with-tenseal
  python export_head_fixture.py --ckpt ../ckks_models/Fix2_Normalized.pth
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Reuse the architecture from the existing TenSEAL script rather than
# re-declaring it -- if the model changes there, this follows automatically.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "core"))
from ckks_classification_head import DeiTTiny  # noqa: E402

# CIFAR-10 lives under SEAFlicker's notebook data dir on this machine.
CIFAR_ROOTS = [
    Path(__file__).resolve().parent.parent / "data",
    Path(__file__).resolve().parent.parent / ".data",
    Path(__file__).resolve().parents[2] / "SEAFlicker-FHE" / "notebooks" / ".data",
]


def find_cifar_root():
    for root in CIFAR_ROOTS:
        if (root / "cifar-10-batches-py").is_dir():
            return root
    return None


def load_cls_tokens(model, n, seed):
    """Return (cls_tokens, labels, source_tag) from a real forward pass."""
    root = find_cifar_root()
    torch.manual_seed(seed)

    if root is not None:
        import torchvision
        import torchvision.transforms as transforms
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        ds = torchvision.datasets.CIFAR10(str(root), train=False,
                                          download=False, transform=tf)
        imgs = torch.stack([ds[i][0] for i in range(n)])
        labels = torch.tensor([ds[i][1] for i in range(n)])
        source = "cifar10-test:" + str(root)
    else:
        # Fall back to noise. The CKKS fidelity claim is about W @ cls + b and
        # does not depend on the images being real -- but say so in the fixture
        # rather than letting a reader assume CIFAR.
        imgs = torch.randn(n, 3, 32, 32)
        labels = torch.full((n,), -1, dtype=torch.long)
        source = "random-normal-images (CIFAR-10 not found)"

    with torch.no_grad():
        cls = model.forward_features(imgs)
    return cls, labels, source


def tenseal_logits(cls_tokens, W, b):
    """Run the existing TenSEAL path on the same CLS tokens, for parity."""
    from ckks_classification_head import (setup_ckks_context,
                                          encrypted_classify_batched)
    ctx = setup_ckks_context()
    out, times = [], []
    for i in range(cls_tokens.shape[0]):
        t0 = time.perf_counter()
        out.append(encrypted_classify_batched(ctx, cls_tokens[i].numpy(), W, b))
        times.append(time.perf_counter() - t0)
    return out, sum(times) / len(times) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=50,
                    help="samples to export (JSON grows ~4KB/sample)")
    ap.add_argument("--ckpt", type=str, default="",
                    help="optional .pth state_dict for the DeiT-Tiny")
    ap.add_argument("--with-tenseal", action="store_true",
                    help="also record TenSEAL CKKS logits for a 3-way check")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="head_fixture.json")
    ap.add_argument("--norm-type", default="batchnorm")
    ap.add_argument("--attn-type", default="poly_normed")
    ap.add_argument("--gelu-type", default="poly")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = DeiTTiny(norm_type=args.norm_type, attn_type=args.attn_type,
                     gelu_type=args.gelu_type)

    trained = False
    if args.ckpt:
        ckpt = Path(args.ckpt)
        if not ckpt.exists():
            raise SystemExit("checkpoint not found: " + str(ckpt))
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        trained = True
    model.eval()

    cls, labels, source = load_cls_tokens(model, args.max_samples, args.seed)

    W = model.head.weight.detach().cpu().numpy()   # (10, 192)
    b = model.head.bias.detach().cpu().numpy()     # (10,)
    with torch.no_grad():
        plain = model.head(cls)

    fixture = {
        "meta": {
            "created_by": "export_head_fixture.py",
            "weights": "trained:" + args.ckpt if trained
                       else "RANDOM INIT (no checkpoint) - accuracy is meaningless, "
                            "numerical fidelity is not",
            "cls_source": source,
            "seed": args.seed,
            "arch": {"norm_type": args.norm_type, "attn_type": args.attn_type,
                     "gelu_type": args.gelu_type},
            "embed_dim": int(W.shape[1]),
            "num_classes": int(W.shape[0]),
            "n_samples": int(cls.shape[0]),
            "tenseal_reference": {
                "poly_modulus_degree": 8192,
                "coeff_mod_bit_sizes": [60, 40, 60],
                "scale_bits": 40,
                "mult_levels": 1,
            },
        },
        "W": W.tolist(),
        "b": b.tolist(),
        "cls": cls.detach().cpu().numpy().tolist(),
        "plain_logits": plain.detach().cpu().numpy().tolist(),
        "labels": labels.tolist(),
    }

    if args.with_tenseal:
        try:
            ts_logits, ts_ms = tenseal_logits(cls, W, b)
            fixture["tenseal_logits"] = ts_logits
            fixture["meta"]["tenseal_mean_ms"] = round(ts_ms, 2)
            print("TenSEAL path recorded: %.1f ms/sample" % ts_ms)
        except Exception as e:
            print("TenSEAL step skipped (%s: %s)" % (type(e).__name__, e))

    out = Path(__file__).resolve().parent / args.out
    out.write_text(json.dumps(fixture))
    size_kb = out.stat().st_size / 1024.0

    print("wrote %s  (%.0f KB)" % (out, size_kb))
    print("  samples      : %d" % cls.shape[0])
    print("  head         : W%s  b%s" % (W.shape, b.shape))
    print("  cls source   : %s" % source)
    print("  weights      : %s" % fixture["meta"]["weights"])
    print()
    print("Next (in WSL):")
    print("  wsl -d Ubuntu-24.04 -- python3 openfhe_head.py --fixture %s" % args.out)


if __name__ == "__main__":
    main()
