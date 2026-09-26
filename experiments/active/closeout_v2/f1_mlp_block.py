"""F1: a continuously encrypted MLP sub-block on the headline failing conversion.

Headline cell: CIFAR-10 checkpoint seed 43, budget profile, GELU-only mask (the 86.33% ->
25.55% seed). For block b the server evaluates, per token and without intermediate
decryption,

    out = x_mid + fc2( p_GELU( fc1( LN2(x_mid) ) ) )

where x_mid is the residual stream after attention, LN2(x_mid) is the (plaintext) LN2
output, and p_GELU is the exact deployed degree-7 polynomial. The decrypted block output is
injected back into the plaintext converted forward (hybrid replay) to measure whether the
CKKS error changes any prediction relative to the plaintext conversion P.

Three stages, because TenSEAL and PyTorch live in different environments on H200:
    capture  (torch)   python f1_mlp_block.py capture <run_root> [--synthetic]
    encrypt  (tenseal) python f1_mlp_block.py encrypt <run_root> <job_index>
    inject   (torch)   python f1_mlp_block.py inject  <run_root>
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from protocol_v2 import F1


def _write(p, d):
    from io_utils import write
    write(p, d)


# ------------------------------------------------------------------ capture (torch)
def capture(root, synthetic=False):
    import torch
    from torch.utils.data import Subset
    from pvit_lab.common import seed_all, device_for
    from pvit_lab.data import loader
    from variants import calibrate_v2, convert_v2
    from worker_v2 import splits, load_model
    root = Path(root); out = root / 'f1'; out.mkdir(parents=True, exist_ok=True)
    m = json.loads((root / 'manifest.json').read_text())
    seed_all(F1['seed']); device = device_for('cpu' if synthetic else 'cuda:0')
    src = None if synthetic else m['sources'][f"regularized_ln_seed{F1['seed']}"]
    base = load_model(src, synthetic).to(device).eval()
    fit, _, dev, _ = splits(m['data_root'], synthetic)
    params, _, meta = calibrate_v2(base, lambda: loader(fit, workers=0 if synthetic else 2), device,
                                   dict(profile=F1['profile']))
    conv, rec, sites = convert_v2(base, params, F1['mask'])
    n = min(F1['images'], len(dev))
    images = Subset(dev, range(n))
    taps = {b: {} for b in F1['blocks']}
    for b in F1['blocks']:
        conv.blocks[b].observer = (lambda name, x, b=b: taps[b].__setitem__(name, x.detach().cpu().double().numpy())
                                   if name in ('norm2_in', 'norm2_out', 'residual_out') else None)
    xs, ys, ids = [], [], []
    for x, y, i in loader(images, workers=0, batch=n):
        xs.append(x); ys.append(y.numpy()); ids.append(i.numpy())
    x = xs[0].to(device)
    with torch.no_grad():
        ref = base(x).cpu().double().numpy()
        rec.begin(len(x), x.device)
        plain = conv(x).cpu().double().numpy()
    for b in F1['blocks']:
        conv.blocks[b].observer = None
    arrays = dict(logits_O=ref, logits_P=plain, labels=ys[0], ids=ids[0])
    record = dict(params_profile=meta, blocks={})
    for b in F1['blocks']:
        blk = conv.blocks[b]
        g = params[f'blocks.{b}.act']['poly']
        arrays[f'b{b}_x_mid'] = taps[b]['norm2_in']; arrays[f'b{b}_h'] = taps[b]['norm2_out']
        arrays[f'b{b}_out_P32'] = taps[b]['residual_out']
        arrays[f'b{b}_fc1_w'] = blk.fc1.weight.detach().cpu().double().numpy()
        arrays[f'b{b}_fc1_b'] = blk.fc1.bias.detach().cpu().double().numpy()
        arrays[f'b{b}_fc2_w'] = blk.fc2.weight.detach().cpu().double().numpy()
        arrays[f'b{b}_fc2_b'] = blk.fc2.bias.detach().cpu().double().numpy()
        record['blocks'][str(b)] = dict(gelu_poly={k: g[k] for k in ('lo', 'hi', 'center', 'radius', 'degree',
                                                                      'coefficients', 'grid_max_abs_error')})
    np.savez_compressed(out / 'fixtures.npz', **arrays)
    from io_utils import sha
    record.update(fixtures_sha256=sha(out / 'fixtures.npz'), images=int(n), seed=F1['seed'], profile=F1['profile'],
                  mask=F1['mask'], split='dev (validation), first images in fixed order',
                  selection='fixed before any encrypted outcome; no filtering of hard or easy images')
    # Optional cross-check against the v1 frozen coefficients for the same cell.
    v1 = m.get('v1_frozen_s43_budget_gelu')
    if v1 and Path(v1).is_file():
        f = json.loads(Path(v1).read_text())['parameters']
        record['v1_coefficients_identical'] = all(
            np.allclose(f[f'blocks.{b}.act']['poly']['coefficients'], params[f'blocks.{b}.act']['poly']['coefficients'],
                        rtol=0, atol=0) for b in F1['blocks'])
    _write(out / 'fixtures_meta.json', record)
    return record


# ------------------------------------------------------------------ encrypt (tenseal)
def encrypt(root, index, smoke=False, threads=4):
    import tenseal as ts
    from fhe_common_v2 import contexts, mlp_weights, plaintext_mlp, encrypted_mlp
    root = Path(root); m = json.loads((root / 'manifest.json').read_text())
    job = m['f1'][index] if not smoke else dict(name='f1_smoke', block=F1['blocks'][0], first=0, count=1)
    out = root / 'f1' / job['name']; out.mkdir(parents=True, exist_ok=smoke)
    try:
        fx = np.load(root / 'f1' / 'fixtures.npz')
        meta = json.loads((root / 'f1' / 'fixtures_meta.json').read_text())
        b = job['block']
        w = mlp_weights(fx[f'b{b}_fc1_w'], fx[f'b{b}_fc1_b'], fx[f'b{b}_fc2_w'], fx[f'b{b}_fc2_b'],
                        meta['blocks'][str(b)]['gelu_poly'])
        t = time.perf_counter()
        client, server = contexts(ts, F1['ring_degree'], F1['modulus_bits'], F1['scale_bits'], threads)
        setup = time.perf_counter() - t
        H, X = fx[f'b{b}_h'], fx[f'b{b}_x_mid']
        first, count = job['first'], job['count']
        tokens = H.shape[1] if not smoke else 1
        C = np.full((count, H.shape[1], H.shape[2]), np.nan); costs = []
        err_cp = []
        for i in range(first, first + count):
            for tkn in range(tokens):
                y, cost = encrypted_mlp(ts, client, server, H[i, tkn], X[i, tkn], w)
                ref = plaintext_mlp(H[i, tkn], X[i, tkn], w)
                C[i - first, tkn] = y
                err_cp.append(float(np.abs(y - ref).max())); costs.append(cost)
        np.savez_compressed(out / 'decrypted.npz', C=C, first=first, count=count, block=b)
        sec = [c['server_seconds'] for c in costs]
        rep = dict(status='measured', job=job, block=b, tokens_per_image=int(tokens),
                   chains=len(costs), ckks_minus_plaintext_max_abs=dict(max=max(err_cp), median=float(np.median(err_cp))),
                   levels_consumed=sorted({c['levels_consumed'] for c in costs}),
                   server_seconds=dict(median=float(np.median(sec)), max=float(max(sec)), total=float(sum(sec))),
                   context_setup_seconds=setup,
                   parameters=dict(scheme='CKKS', backend='TenSEAL', version=ts.__version__,
                                   ring_degree=F1['ring_degree'], modulus_bits=F1['modulus_bits'],
                                   scale_bits=F1['scale_bits'], security='SEAL default tc128 parameter validation',
                                   bootstrapping=False, galois_keys=True, server_has_secret_key=server.has_secret_key()),
                   boundary=('Encrypted: fc1 (input scaling folded), degree-%d GELU polynomial, fc2, biases, residual '
                             'add. Plaintext: LN2 and everything outside block %d MLP. One ciphertext per token; '
                             'hybrid replay afterwards.' % (len(w['coefficients']) - 1, b)))
        _write(out / 'report.json', rep)
        return rep
    except Exception as e:
        _write(out / 'error.json', dict(error=str(e), traceback=traceback.format_exc()))
        raise


# ------------------------------------------------------------------ inject (torch)
def inject(root, synthetic=False):
    import torch
    from torch.utils.data import Subset
    from pvit_lab.common import seed_all, device_for
    from pvit_lab.data import loader
    from variants import calibrate_v2, convert_v2
    from worker_v2 import splits, load_model
    from probes import agreement
    root = Path(root); m = json.loads((root / 'manifest.json').read_text())
    fx = np.load(root / 'f1' / 'fixtures.npz')
    seed_all(F1['seed']); device = device_for('cpu' if synthetic else 'cuda:0')
    src = None if synthetic else m['sources'][f"regularized_ln_seed{F1['seed']}"]
    base = load_model(src, synthetic).to(device).eval()
    fit, _, dev, _ = splits(m['data_root'], synthetic)
    params, _, _ = calibrate_v2(base, lambda: loader(fit, workers=0 if synthetic else 2), device,
                                dict(profile=F1['profile']))
    conv, rec, _ = convert_v2(base, params, F1['mask'])
    n = len(fx['labels'])
    x = next(iter(loader(Subset(dev, range(n)), workers=0, batch=n)))[0].to(device)
    report = dict(status='measured', blocks={})
    for b in F1['blocks']:
        parts = sorted((root / 'f1').glob(f'f1_b{b}_part*/decrypted.npz'))
        C = np.full(fx[f'b{b}_h'].shape, np.nan)
        for p in parts:
            d = np.load(p); C[int(d['first']):int(d['first']) + int(d['count'])] = d['C']
        covered = np.isfinite(C).all((1, 2))
        if not covered.any():
            report['blocks'][str(b)] = dict(status='no_encrypted_outputs'); continue
        from fhe_common_v2 import mlp_weights, plaintext_mlp
        meta = json.loads((root / 'f1' / 'fixtures_meta.json').read_text())
        w = mlp_weights(fx[f'b{b}_fc1_w'], fx[f'b{b}_fc1_b'], fx[f'b{b}_fc2_w'], fx[f'b{b}_fc2_b'],
                        meta['blocks'][str(b)]['gelu_poly'])
        P64 = plaintext_mlp(fx[f'b{b}_h'], fx[f'b{b}_x_mid'], w)

        def run_with(block_out):
            t = torch.as_tensor(block_out, dtype=torch.float32, device=device)
            h = conv.blocks[b].register_forward_hook(lambda mod, a, o: t)
            try:
                with torch.no_grad():
                    rec.begin(len(x), x.device)
                    return conv(x).cpu().double().numpy()
            finally:
                h.remove()
        idx = np.where(covered)[0]
        logits_C = run_with(np.where(covered[:, None, None], C, P64))[idx]
        logits_P64 = run_with(P64)[idx]
        O, P, y = fx['logits_O'][idx], fx['logits_P'][idx], fx['labels'][idx]
        report['blocks'][str(b)] = dict(
            images_with_encrypted_block=int(covered.sum()),
            block_output_ckks_minus_plaintext_max_abs=float(np.abs(C[covered] - P64[covered]).max()),
            injection_sanity_P64_vs_P32=agreement(P, logits_P64, y),
            C_vs_P=agreement(logits_P64, logits_C, y),
            P_vs_O=agreement(O, P, y),
            max_logit_difference_C_vs_P=float(np.abs(logits_C - logits_P64).max()),
            accuracy=dict(O=100 * float((O.argmax(1) == y).mean()), P=100 * float((P.argmax(1) == y).mean()),
                          C=100 * float((logits_C.argmax(1) == y).mean())))
    report['scope'] = ('Hybrid replay: exactly one MLP sub-block is continuously encrypted per run; the rest of '
                       'the network is plaintext converted arithmetic. This is not full encrypted inference.')
    _write(root / 'f1' / 'f1_report.json', report)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['capture', 'encrypt', 'smoke', 'inject'])
    ap.add_argument('root'); ap.add_argument('index', nargs='?', type=int)
    ap.add_argument('--synthetic', action='store_true'); ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    if a.stage == 'capture':
        print(json.dumps(capture(a.root, a.synthetic), indent=1)[:2000])
    elif a.stage == 'encrypt':
        encrypt(a.root, a.index, threads=a.threads)
    elif a.stage == 'smoke':
        r = encrypt(a.root, 0, smoke=True, threads=a.threads)
        print(json.dumps(r, indent=1))
        if r['server_seconds']['max'] > F1['smoke_max_seconds_per_token']:
            print('SLOW: reduce F1 to', F1['reduced_images'], 'images (see protocol_v2.F1)')
    else:
        print(json.dumps(inject(a.root, a.synthetic), indent=1))
