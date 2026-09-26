"""Configurable conversions for closeout_v2, built ON the frozen v1 arithmetic.

`calibrate_v2` delegates to the frozen `polynomial.calibrate` (copied verbatim from
results/campaign_2026-09-20/source_final/) through a named custom profile, so a v2 run
with the v1 settings reproduces the v1 coefficients exactly. The only new arithmetic:

* `domain='central'` refits GELU sites on the [p0.1, p99.9] quantiles of the fit-split
  activations instead of the min-max interval (+5% margins). Inputs outside the fitted
  interval are then evaluated by polynomial extrapolation, and counted.
* Attention oracle variants replace the fitted exponential numerator and/or the fixed
  Goldschmidt reciprocal with exact floating-point operations. They are measurement
  controls for attribution, never deployable arithmetic.
* Block filters convert one block only (`only_block`) or all blocks but one
  (`except_block`).
"""
import copy
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import polynomial as P  # frozen v1 module
from protocol_v2 import CENTRAL_QUANTILES, SAMPLE_PER_SITE, V1


def _profile(cfg):
    base = dict(V1[cfg.get('profile', 'budget')])
    for k in ('gelu_degree', 'norm_degree', 'exp_root_degree', 'reciprocal_steps'):
        if k in cfg:
            base[k] = int(cfg[k])
    return base


def _gelu_np(x):
    return .5 * x * (1 + np.array([math.erf(float(v) / np.sqrt(2)) for v in x]))


@torch.no_grad()
def gelu_samples(base, batches, device, limit=SAMPLE_PER_SITE, seed=0):
    """Uniform random sample of GELU-input activations per site (fit split only)."""
    g = torch.Generator().manual_seed(seed)
    store, handles = {}, []

    def hook(mod, args, name):
        x = args[0].detach().flatten().double().cpu()
        keep = torch.randperm(x.numel(), generator=g)[:max(1, limit // 16)]
        store.setdefault(name, []).append(x[keep])

    for name, mod in base.named_modules():
        if isinstance(mod, nn.GELU):
            handles.append(mod.register_forward_pre_hook(lambda m, a, name=name: hook(m, a, name)))
    try:
        for x, _, _ in batches:
            base(x.to(device))
    finally:
        for h in handles:
            h.remove()
    return {k: torch.cat(v)[:limit].numpy() for k, v in store.items()}


def calibrate_v2(base, batches_factory, device, cfg):
    """Return (params, fit_ids, calibration_meta). `batches_factory()` yields fresh fit batches."""
    prof = _profile(cfg)
    key = 'v2_' + '_'.join(f'{k}{v}' for k, v in sorted(prof.items()))
    P.PROFILES[key] = prof
    try:
        params, ids = P.calibrate(base, batches_factory(), device, key)
    finally:
        P.PROFILES.pop(key, None)
    meta = dict(profile=cfg.get('profile', 'budget'), effective=prof, domain=cfg.get('domain', 'minmax'))
    if cfg.get('domain', 'minmax') == 'central':
        samples = gelu_samples(base, batches_factory(), device)
        lo_q, hi_q = CENTRAL_QUANTILES
        refit = {}
        for name, p in params.items():
            if p['kind'] != 'gelu':
                continue
            s = samples[name]
            lo, hi = float(np.quantile(s, lo_q)), float(np.quantile(s, hi_q))
            p['poly'] = P.fit_poly(lo, hi, prof['gelu_degree'], _gelu_np)
            p['poly']['domain_rule'] = f'central quantiles {CENTRAL_QUANTILES} of fit-split activations'
            refit[name] = dict(lo=lo, hi=hi, fit_samples=int(s.size),
                               fit_fraction_outside=float(((s < lo) | (s > hi)).mean()))
        meta['central_refit'] = refit
    return params, ids, meta


class OracleSoftmax(nn.Module):
    """Frozen polynomial attention with optional exact numerator / exact division."""

    def __init__(self, p, name, rec, variant):
        super().__init__()
        assert variant in ('exact_numerator', 'exact_reciprocal', 'exact_both')
        self.p, self.name, self.rec, self.variant = p, name, rec, variant

    def forward(self, s):
        z = s - s.mean(-1, keepdim=True)
        self.rec.check(self.name, z, self.p['root'])
        if self.variant in ('exact_numerator', 'exact_both'):
            # The fitted root approximates exp((z - shift)/2); its exact square is exp(z - shift).
            u = torch.exp(z - self.p['public_exp_shift']) + self.p['floor']
        else:
            u = P.torch_poly(z, self.p['root']).square() + self.p['floor']
        d = u.sum(-1, keepdim=True)
        self.rec.check(self.name + '.denominator', d, self.p['reciprocal'])
        if self.variant in ('exact_reciprocal', 'exact_both'):
            return u / d
        return u * P.reciprocal(d, self.p['reciprocal'])


def site_block(name):
    parts = name.split('.')
    return int(parts[1]) if parts[0] == 'blocks' else None


def convert_v2(base, params, mask, variant='poly', only_block=None, except_block=None):
    """Like the frozen `polynomial.convert`, plus attention variants and block filters."""
    model = copy.deepcopy(base).eval()
    rec = P.Recorder()
    converted = []
    for name, old in list(model.named_modules()):
        if name not in params:
            continue
        b = site_block(name)
        if only_block is not None and b != only_block:
            continue
        if except_block is not None and b == except_block:
            continue
        parent_name, _, leaf = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        p = params[name]
        if p['kind'] == 'gelu' and mask in ('gelu', 'all'):
            setattr(parent, leaf, P.PolynomialGELU(p['poly'], name, rec)); converted.append(name)
        if p['kind'] == 'norm' and mask in ('norm', 'all'):
            setattr(parent, leaf, P.PolynomialLN(old, p['poly'], name, rec)); converted.append(name)
        if p['kind'] == 'attention' and mask in ('attention', 'all'):
            mod = P.PolynomialSoftmax(p, name, rec) if variant == 'poly' else OracleSoftmax(p, name, rec, variant)
            setattr(parent, leaf, mod); converted.append(name)
    for q in model.parameters():
        q.requires_grad_(False)
    return model, rec, converted
