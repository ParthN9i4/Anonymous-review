#!/usr/bin/env python3
"""Evaluate an activation-distribution refit on the campaign gate and test.

Run from the campaign code directory. The script reconstructs the budget
GELU conversion from the manifest and fit set, verifies the historical frozen
test accuracy, then evaluates a degree-7 refit at block 0 and at all blocks.
The test set is read only after fitting and the expected baseline checks pass.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from data_eval import subsets
from pvit_lab.data import loader
from polynomial import calibrate, convert, PolynomialGELU, Recorder
from worker import load


EXPECTED_REFERENCE = {42: 86.17, 43: 86.33, 44: 86.32}
EXPECTED_FROZEN = {42: 26.93, 43: 25.55, 44: 22.72}


def empirical_gelu_params(model, batches, params, device, seed, cap=200_000):
    """Fit one degree-7 polynomial per GELU using only fitting images."""
    rng = np.random.default_rng(seed)
    values = {f"blocks.{i}.act": [] for i in range(len(model.blocks))}
    handles = []

    def make_hook(name):
        def hook(_module, args):
            x = args[0].detach().reshape(-1).float().cpu().numpy()
            if len(x) <= cap:
                values[name].append(x)
            else:
                idx = rng.choice(len(x), cap, replace=False)
                values[name].append(x[idx])
        return hook

    for i, block in enumerate(model.blocks):
        handles.append(block.act.register_forward_pre_hook(make_hook(f"blocks.{i}.act")))
    try:
        with torch.no_grad():
            for x, _, _ in batches:
                model(x.to(device))
    finally:
        for h in handles:
            h.remove()

    out = {}
    for i in range(len(model.blocks)):
        name = f"blocks.{i}.act"
        x = np.concatenate(values[name])
        old = params[name]["poly"]
        degree = int(old["degree"])
        # Fit on the captured activation density, preserving the frozen domain.
        z = (x - old["center"]) / old["radius"]
        truth = 0.5 * x * (1 + np.fromiter(
            (math.erf(float(v) / math.sqrt(2)) for v in x),
            dtype=float, count=len(x)))
        # Solve in the Chebyshev basis directly; this avoids the poorly
        # conditioned power-basis conversion for wide activation intervals.
        V = np.polynomial.chebyshev.chebvander(z, degree)
        cheb, *_ = np.linalg.lstsq(V, truth, rcond=1e-12)
        co = np.polynomial.chebyshev.cheb2poly(cheb)
        p = dict(old)
        p["coefficients"] = co.tolist()
        p["fit_source"] = "activation-density refit on fitting images"
        p["sample_count"] = int(len(x))
        out[name] = p
    return out


def replace_gelu(model, params, blocks):
    model = copy.deepcopy(model).eval()
    for i in blocks:
        rec = Recorder()
        rec.enabled = False
        model.blocks[i].act = PolynomialGELU(
            params[f"blocks.{i}.act"], f"blocks.{i}.act", rec)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def score(model, batches, device):
    correct = total = invalid = 0
    for x, y, _ in batches:
        z = model(x.to(device))
        valid = torch.isfinite(z).all(dim=1)
        pred = z.argmax(dim=1)
        correct += int(((pred == y.to(device)) & valid).sum())
        invalid += int((~valid).sum())
        total += len(y)
    return dict(n=total, accuracy_percent=100.0 * correct / total,
                invalid_images=invalid)


def ids_from_batches(batches):
    ids = []
    for _, _, batch_ids in batches:
        ids.extend(int(v) for v in batch_ids.tolist())
    return ids


def main(root, configuration, output, device_name, tolerance):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    jobs = [j for j in manifest["conversion"] if j["name"] == configuration]
    if len(jobs) != 1:
        raise ValueError(f"Configuration not uniquely found: {configuration}")
    job = jobs[0]
    seed = int(job["seed"])
    if job["profile"] != "budget" or job["mask"] != "gelu" or int(job["fit_n"]) != 2048:
        raise ValueError("This evaluator requires a budget/GELU/2048 configuration")

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    base = load(job["source"]).to(device).eval()
    fit, gate, sets = subsets(manifest["data_root"], job["fit_n"], evaluation=True)
    fit_batches = list(loader(fit, workers=2 if device.type == "cuda" else 0))
    gate_batches = list(loader(gate, workers=2 if device.type == "cuda" else 0))
    test_batches = list(loader(sets["clean"], workers=2 if device.type == "cuda" else 0))
    fit_ids_exact = ids_from_batches(fit_batches)
    gate_ids_exact = ids_from_batches(gate_batches)

    # Fit and score before reading the test set through the model.
    start = time.time()
    frozen_params, fit_ids = calibrate(base, fit_batches, device, "budget")
    frozen, frozen_rec = convert(base, frozen_params, "gelu")
    # This evaluator measures accuracy only; do not invoke the campaign's
    # per-batch range recorder, which requires an explicit begin() call.
    frozen_rec.enabled = False
    frozen = frozen.to(device).eval()
    ref_gate = score(base, gate_batches, device)
    frozen_gate = score(frozen, gate_batches, device)
    ref_test = score(base, test_batches, device)
    frozen_test = score(frozen, test_batches, device)

    if abs(ref_test["accuracy_percent"] - EXPECTED_REFERENCE[seed]) > tolerance:
        raise RuntimeError(
            f"Reference test accuracy {ref_test['accuracy_percent']:.4f} does not "
            f"reproduce expected {EXPECTED_REFERENCE[seed]:.2f} for seed {seed}")
    if abs(frozen_test["accuracy_percent"] - EXPECTED_FROZEN[seed]) > tolerance:
        raise RuntimeError(
            f"Frozen test accuracy {frozen_test['accuracy_percent']:.4f} does not "
            f"reproduce expected {EXPECTED_FROZEN[seed]:.2f} for seed {seed}")

    refit = empirical_gelu_params(base, fit_batches, frozen_params, device, 20260924 + seed)
    block0 = replace_gelu(frozen, refit, [0])
    all_blocks = replace_gelu(frozen, refit, range(len(base.blocks)))

    results = {
        "configuration": configuration,
        "seed": seed,
        "scope": "post hoc activation-density refit; fit and gate are training images; clean test is locked after baseline checks",
        "fit_n_images": len(fit),
        "gate_n_images": len(gate),
        "test_n_images": len(sets["clean"]),
        "fit_gate_overlap": len(set(fit_ids_exact) & set(gate_ids_exact)),
        "degree": 7,
        "baseline": {"reference_gate": ref_gate, "frozen_gate": frozen_gate,
                     "reference_test": ref_test, "frozen_test": frozen_test},
        "refit": {
            "block0_gate": score(block0, gate_batches, device),
            "block0_test": score(block0, test_batches, device),
            "all_blocks_gate": score(all_blocks, gate_batches, device),
            "all_blocks_test": score(all_blocks, test_batches, device),
        },
        "refit_layers": sorted(refit),
        "elapsed_seconds": time.time() - start,
        "weights_preserved": True,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("configuration")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tolerance", type=float, default=0.25)
    a = p.parse_args()
    main(a.root, a.configuration, a.output, a.device, a.tolerance)
