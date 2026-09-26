#!/usr/bin/env python3
"""CIFAR-10 calibration-only diagnosis of block-0 GELU at equal polynomial degree.

Place this file in campaign/code beside worker.py. Run with campaign root and a
budget conversion name; it reads the frozen checkpoint/fit/gate configuration.
No test images are loaded. No files in the campaign are modified.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data_eval import subsets
from pvit_lab.data import loader
from polynomial import torch_poly
from worker import load


class FittedGELU(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, x):
        return torch_poly(x, self.params)


def sample_inputs(model, batches, cap, seed, device):
    """Uniformly sample activations *within each batch*, then cap total."""
    rng = np.random.default_rng(seed)
    kept = []
    def hook(_module, args):
        x = args[0].detach().reshape(-1)
        take = min(len(x), max(1, cap // max(1, n_batches)))
        idx = rng.choice(len(x), take, replace=False)
        kept.append(x[torch.as_tensor(idx, device=x.device)].cpu().numpy())
    n_batches = len(batches)
    handle = model.blocks[0].act.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for x, _, _ in batches:
                model(x.to(device))
    finally:
        handle.remove()
    return np.concatenate(kept)


def gelu(x):
    # Compare with the original nn.GELU's exact form, not its tanh approximation.
    return .5 * x * (1 + np.fromiter(
        (math.erf(float(v) / math.sqrt(2)) for v in x), float, count=len(x)))


def fit_on_empirical_inputs(x, domain, degree):
    center, radius = domain['center'], domain['radius']
    z = (x - center) / radius
    # Same degree, centered domain and polynomial representation as frozen fit.
    cheb = np.polynomial.chebyshev.chebfit(z, gelu(x), degree)
    co = np.polynomial.chebyshev.cheb2poly(cheb)
    return {**domain, 'coefficients': co.tolist(), 'degree': degree,
            'fit_source': 'sampled fit-set block-0 activations'}


def describe(x, baseline, candidate):
    truth = gelu(x)
    result = {}
    for name, p in [('frozen', baseline), ('empirical', candidate)]:
        y = np.polynomial.polynomial.polyval((x-p['center'])/p['radius'],
                                              p['coefficients'])
        err = np.abs(y-truth)
        result[name] = dict(mean_abs=float(err.mean()),
                            p95_abs=float(np.quantile(err, .95)),
                            max_abs=float(err.max()),
                            central_mean_abs=float(err[np.abs(x) <= 1].mean()))
    result['input_quantiles'] = np.quantile(x, [0, .01, .1, .5, .9, .99, 1]).tolist()
    result['outside_frozen_domain_fraction'] = float(np.mean(
        (x < baseline['lo']) | (x > baseline['hi'])))
    return result


@torch.no_grad()
def accuracy_on_gate(base, p0, p1, batches, device):
    models = [base, copy.deepcopy(base), copy.deepcopy(base)]
    models[1].blocks[0].act = FittedGELU(p0)
    models[2].blocks[0].act = FittedGELU(p1)
    for m in models:
        m.eval()
    counts = [0, 0, 0]
    invalid = [0, 0, 0]
    total = 0
    for x, y, _ in batches:
        x, y = x.to(device), y.to(device)
        total += len(y)
        for i, m in enumerate(models):
            z = m(x)
            valid = torch.isfinite(z).all(dim=-1)
            counts[i] += int(((z.argmax(-1) == y) & valid).sum())
            invalid[i] += int((~valid).sum())
    return dict(n=total, accuracy_percent=[100*c/total for c in counts],
                invalid_images=invalid,
                order=['native', 'block0_frozen_fit', 'block0_empirical_fit'])


def main(root, name, out, sample_cap):
    manifest = json.loads((root/'manifest.json').read_text())
    job = next(j for j in manifest['conversion'] if j['name'] == name)
    if job['profile'] != 'budget' or job['fit_n'] != 2048:
        raise ValueError('Use the original budget/2048 configuration')
    frozen = json.loads((root/'conversion'/name/'frozen.json').read_text())
    p0 = frozen['parameters']['blocks.0.act']['poly']
    model = load(job['source']).to('cuda').eval()
    fit, gate, _ = subsets(manifest['data_root'], job['fit_n'], evaluation=False)
    fit_batches = list(loader(fit, workers=2))
    gate_batches = list(loader(gate, workers=2))
    if sorted(set(frozen['fit_image_ids']) & set(frozen['gate_image_ids'])):
        raise ValueError('Fit and gate overlap')
    x_fit = sample_inputs(model, fit_batches, sample_cap, 20260924, 'cuda')
    x_gate = sample_inputs(model, gate_batches, sample_cap, 20260925, 'cuda')
    p1 = fit_on_empirical_inputs(x_fit, p0, p0['degree'])
    result = dict(configuration=name, scope='fit and disjoint gate only; no test',
                  fit_n=len(x_fit), gate_n=len(x_gate), degree=p0['degree'],
                  fit=describe(x_fit, p0, p1), gate=describe(x_gate, p0, p1),
                  gate_block0=accuracy_on_gate(model, p0, p1, gate_batches, 'cuda'),
                  empirical_polynomial=p1)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('configuration','fit_n','gate_n','degree','fit','gate','gate_block0')},indent=2))


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('campaign_root', type=Path)
    a.add_argument('configuration')
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--sample-cap', type=int, default=200000)
    v = a.parse_args()
    main(v.campaign_root, v.configuration, v.output, v.sample_cap)
