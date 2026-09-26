#!/usr/bin/env python3
"""One measured TenSEAL CKKS GELU primitive on the original four-image fixture.

Place beside fhe.py in campaign/code; run on CPU node with fixture, frozen JSON,
and metadata from one existing conversion directory. Writes a fresh JSON result.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from fhe import execute
from io_utils import sha


def fit_poly(lo, hi, degree, fn):
    """Fit in the same normalized polynomial basis as campaign/polynomial.py.

    Kept local so the CKKS-only environment does not need PyTorch.
    """
    center = (lo + hi) / 2.0
    radius = (hi - lo) / 2.0
    nodes = np.cos(np.pi * (np.arange(1024) + .5) / 1024)
    x = center + radius * nodes
    cheb = np.polynomial.chebyshev.chebfit(nodes, fn(x), degree)
    co = np.polynomial.chebyshev.cheb2poly(cheb)
    return dict(lo=float(lo), hi=float(hi), center=float(center),
                radius=float(radius), coefficients=co.tolist(), degree=degree)


def main(source, output, block, bits):
    import tenseal as ts
    fixture, frozen_path, metadata_path = (source/'fhe_inputs.npz',
                                           source/'frozen.json',
                                           source/'fhe_inputs_meta.json')
    meta = json.loads(metadata_path.read_text())
    if sha(fixture) != meta['sha256'] or sha(frozen_path) != meta['frozen_sha256']:
        raise ValueError('Fixture/frozen metadata hash mismatch')
    frozen = json.loads(frozen_path.read_text())
    original = frozen['parameters'][f'blocks.{block}.act']['poly']
    if original['degree'] != 7:
        raise ValueError('Use a budget-profile degree-7 reference')
    p = fit_poly(original['lo'], original['hi'], 31,
                 lambda x: .5*x*(1+np.fromiter(
                     (math.erf(float(v)/math.sqrt(2)) for v in x),
                     float, count=len(x))))
    # Same ring/chain/scaling as the existing budget primitive, not a newly
    # chosen security or parameter configuration. Backend validates chain.
    chain = [60] + [bits]*18 + [60]
    if bits != 35 or sum(chain) > 881:
        raise ValueError('This probe is restricted to the original budget chain')
    raw = np.load(fixture)
    values = raw[f'b{block}_gelu'][:,0,:].astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite fixture input')
    x = (values-p['center'])/p['radius']
    plaintext = np.polynomial.polynomial.polyval(x, p['coefficients'])
    exact = .5*values*(1+np.vectorize(math.erf)(values/math.sqrt(2)))
    if not np.isfinite(plaintext).all():
        raise ValueError('Nonfinite degree-31 plaintext result')
    started = time.perf_counter()
    client = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=32768,
                        coeff_mod_bit_sizes=chain,
                        n_threads=int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    client.global_scale = 2**bits
    server = ts.context_from(client.serialize(save_secret_key=False),
                             n_threads=int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    if server.has_secret_key():
        raise RuntimeError('Server unexpectedly has secret key')
    setup_seconds = time.perf_counter()-started
    trials=[]
    for repeat in range(2):
        decrypted, cost = execute(values, p, 'gelu', client, server, ts)
        trials.append(dict(repeat=repeat, cost=cost,
                           ckks_max_abs=float(np.max(np.abs(decrypted-plaintext))),
                           approximation_max_abs=float(np.max(np.abs(plaintext-exact))),
                           nonfinite=int((~np.isfinite(decrypted)).sum())))
    report=dict(scope='Four fixed CIFAR-10 images, block-0 class token GELU only',
                source=str(source), fixture_sha256=meta['sha256'],
                frozen_sha256=meta['frozen_sha256'], block=block, degree=31,
                parameters=dict(ring_degree=32768, modulus_bits=chain,
                                scale_bits=bits, backend='TenSEAL',
                                backend_version=ts.__version__),
                setup_seconds=setup_seconds,
                fit_domain=[p['lo'],p['hi']],
                fixture_outside_fraction=float(np.mean((values<p['lo'])|(values>p['hi']))),
                trials=trials,
                limitation='Primitive only: excludes upstream encrypted layers and model inference')
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    a=argparse.ArgumentParser()
    a.add_argument('conversion_directory', type=Path)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--block', type=int, default=0)
    a.add_argument('--bits', type=int, default=35)
    v=a.parse_args()
    main(v.conversion_directory,v.output,v.block,v.bits)
