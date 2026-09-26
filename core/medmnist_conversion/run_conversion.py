#!/usr/bin/env python3
"""Run the frozen plaintext conversion pipeline on a MedMNIST checkpoint.

    python -m medmnist_conversion.run_conversion \
        --checkpoint checkpoints_substitution_bloodmnist/'C: norm only_seed42.pt' \
        --dataset bloodmnist --mask gelu --degree 7 --iterations 2 \
        --out results/conversion_blood_s42_budget_gelu --data-root ./data

WHAT THIS DOES AND DOES NOT DO
------------------------------
It installs `medmnist_conversion.data.datasets` in place of the CIFAR-only
loader that `pvit_lab.deploy` imports, then calls `deploy.run()` unchanged.
No file under `harness/` is modified; `PROVENANCE.md` declares that tree a
frozen reference.

The substitution is done by assignment on the already-imported module object,
which is the narrowest intervention available. It is asserted before and after
so a future refactor of the frozen harness cannot make this silently no-op --
if `deploy` stops taking its loader from module scope, this script fails loudly
instead of quietly converting CIFAR images.

PROFILES. The campaign's two named profiles are (degree 7, k=2) for 'budget'
and (degree 15, k=4) for 'accurate', per docs/APPROXIMATIONS.md. They are two
points on a cost axis, not a controlled comparison: each bundles polynomial
degree AND reciprocal iteration count. Do not attribute a difference between
them to either factor alone.

SCOPE. A result from this script is a MedMNIST conversion measured under the
same *procedure* as the CIFAR-10 campaign, not under the same split protocol:
MedMNIST uses its official validation split where CIFAR-10 carves one from
train. Report it as a separate campaign. It is also controlled replication
within this project, not independent external confirmation -- this repository
has used these datasets before.

NOT VERIFIED BY EXECUTION: written without torch/medmnist available. Run the
adapter self-test first (see --check).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

PROFILES = {"budget": (7, 2), "accurate": (15, 4)}


def install_loader():
    """Point pvit_lab.deploy at the MedMNIST loader. Verified, not assumed."""
    from pvit_lab import deploy

    from .data import datasets as medmnist_datasets

    original = getattr(deploy, "datasets", None)
    if original is None:
        raise RuntimeError(
            "pvit_lab.deploy has no module-level `datasets` to replace. The "
            "frozen harness has changed shape; re-read deploy.run() before "
            "trusting this wiring.")
    deploy.datasets = medmnist_datasets
    if deploy.datasets is not medmnist_datasets:      # paranoia, cheap
        raise RuntimeError("loader substitution did not take effect")
    return deploy, original


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint")
    ap.add_argument("--out")
    ap.add_argument("--dataset", default="bloodmnist")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--mask", help="gelu | attention | norm | all; "
                                   "omit to convert everything")
    ap.add_argument("--profile", choices=sorted(PROFILES),
                    help="sets --degree and --iterations together")
    ap.add_argument("--degree", type=int)
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--check", action="store_true",
                    help="run the adapter self-test and exit, converting nothing")
    a = ap.parse_args()

    if a.check:
        from .data import _self_test
        return _self_test(a.dataset, a.data_root, download=True)

    for required in ("checkpoint", "out"):
        if not getattr(a, required):
            ap.error(f"--{required} is required unless --check is given")

    if a.profile:
        if a.degree or a.iterations:
            ap.error("--profile sets --degree and --iterations; do not pass both")
        a.degree, a.iterations = PROFILES[a.profile]
    if a.degree is None or a.iterations is None:
        ap.error("give --profile, or both --degree and --iterations")

    ckpt = Path(a.checkpoint)
    if not ckpt.exists():
        # The repository ships no .pt/.pth files; checkpoints live on the
        # training host. Say so rather than failing inside torch.load.
        ap.error(f"checkpoint not found: {ckpt}\n"
                 f"No trained checkpoints are stored in this repository. "
                 f"Copy the MedMNIST checkpoint from the training host first.")

    deploy, original = install_loader()
    print(f"loader: pvit_lab.deploy.datasets <- medmnist_conversion.data "
          f"(was {original.__module__}.{original.__name__})")
    print(f"converting {ckpt.name} | dataset={a.dataset} | mask={a.mask or 'all'} "
          f"| degree={a.degree} | reciprocal k={a.iterations}")

    report = deploy.run(str(ckpt), a.out, a.data_root, a.mask, a.dataset,
                        a.degree, a.iterations, a.workers, a.device)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapter_provenance.json").write_text(json.dumps({
        "loader": "medmnist_conversion.data.datasets",
        "replaced": f"{original.__module__}.{original.__name__}",
        "dataset": a.dataset,
        "mask": a.mask,
        "degree": a.degree,
        "reciprocal_iterations": a.iterations,
        "checkpoint": str(ckpt),
        "split_protocol": "official MedMNIST train/val/test; NOT the CIFAR-10 "
                          "carve-from-train protocol used by the 20 Sep campaign",
        "independent_confirmation": False,
        "note": "Controlled replication within this project. This repository has "
                "previously used MedMNIST, so these are not untouched inputs.",
    }, indent=2))

    print(json.dumps(report, indent=2)[:2000])
    print(f"\nwrote {out}/ (see adapter_provenance.json for the scope record)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
