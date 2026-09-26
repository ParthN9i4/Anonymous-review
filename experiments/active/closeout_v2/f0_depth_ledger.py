"""F0: measured CKKS level cost of every primitive a ViT block needs (TenSEAL).

Each primitive runs on fresh random ciphertexts at the stated parameters; the recorded
`levels` is the drop in coefficient-modulus primes. A primitive that exhausts the chain
is recorded as an exception, not dropped. The per-block composition written at the end
is an ARITHMETIC SUM OF MEASURED PRIMITIVE DEPTHS along the critical path. It is not a
measured block, and it says so in the output.

    python f0_depth_ledger.py <run_root> <setting_index>
    python f0_depth_ledger.py --local-test          (small ring, fast)
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from protocol_v2 import F0

D, H, MLP, TOK = 192, 64, 768, 65


def measure(ts, client, server, name, fn, ref=None):
    rng = np.random.default_rng(0)
    t = time.perf_counter()
    try:
        out, start, ref_val = fn(rng)
        end = out.ciphertext()[0].coeff_modulus_size()
        rec = dict(primitive=name, status='measured', levels=start - end, seconds=time.perf_counter() - t,
                   end_primes=end)
        if ref_val is not None:
            dec = np.asarray(ts.ckks_vector_from(client, out.serialize()).decrypt())
            rec['max_abs_error'] = float(np.abs(dec - ref_val).max())
        return rec
    except Exception as e:
        return dict(primitive=name, status='exception', error=str(e), seconds=time.perf_counter() - t)


def ledger(setting, threads=4):
    import tenseal as ts
    from fhe_common_v2 import contexts
    from fhe import crypto_softmax  # frozen v1 attention-row circuit
    client, server = contexts(ts, setting['ring_degree'], setting['modulus_bits'], setting['scale_bits'], threads)

    def enc(v):
        return ts.ckks_vector_from(server, ts.ckks_vector(client, list(map(float, v))).serialize())

    def lvl(v):
        return v.ciphertext()[0].coeff_modulus_size()

    def mm(n_in, n_out):
        def f(rng):
            x = rng.normal(size=n_in) * .5; W = rng.normal(size=(n_in, n_out)) * .05; b = rng.normal(size=n_out) * .1
            e = enc(x); s = lvl(e)
            return e.mm(W.tolist()) + ts.ckks_vector(server, b.tolist()), s, x @ W + b
        return f

    def poly(deg):
        def f(rng):
            x = rng.uniform(-1, 1, size=D); c = rng.normal(size=deg + 1) * .3 ** np.arange(deg + 1)
            e = enc(x); s = lvl(e)
            # v1 circuits apply the (x - centre)/radius scaling as a separate scalar multiply.
            return (e * 1.0).polyval(c.tolist()), s, np.polynomial.polynomial.polyval(x, c)
        return f

    def dot_ctct(rng):
        q = rng.normal(size=H); k = rng.normal(size=H); a, b = enc(q), enc(k); s = lvl(a)
        return a.dot(b), s, np.array([q @ k])

    def mul_ctct(rng):
        a_ = rng.uniform(0, .1, size=TOK); v = rng.normal(size=TOK); a, b = enc(a_), enc(v); s = lvl(a)
        return (a * b).sum(), s, np.array([a_ @ v])

    def reciprocal(steps):
        def f(rng):
            lo, hi = .0117, 1.667                       # the v1 budget denominator domain (Table I)
            d = rng.uniform(lo, hi, size=D); e0 = enc(d); s = lvl(e0); scale = 2 / (lo + hi)
            e = -(e0 * scale) + 1; y = e + 1
            for _ in range(steps):
                e = e.square(); y = y * (e + 1)
            ref = 1 - d * scale; yy = 1 + ref; ee = ref
            for _ in range(steps):
                ee = ee * ee; yy = yy * (1 + ee)
            return y * scale, s, yy * scale
        return f

    def softmax_row(root_degree, steps):
        def f(rng):
            cols = [rng.normal(size=4) * .5 for _ in range(TOK)]
            root = dict(center=0., radius=4., lo=-4., hi=4., coefficients=list(rng.normal(size=root_degree + 1) * .1))
            p = dict(root=root, floor=1e-12, reciprocal=dict(lo=.5, hi=5., steps=steps))
            enc_cols = [enc(c) for c in cols]; s = lvl(enc_cols[0])
            return crypto_softmax(enc_cols, p)[0], s, None
        return f

    rows = [measure(ts, client, server, 'mm 192->576 (QKV)', mm(D, 3 * D)),
            measure(ts, client, server, 'mm 192->192 (proj)', mm(D, D)),
            measure(ts, client, server, 'mm 192->768 (fc1)', mm(D, MLP)),
            measure(ts, client, server, 'mm 768->192 (fc2)', mm(MLP, D)),
            measure(ts, client, server, 'ct.ct dot, 64 (QK^T entry)', dot_ctct),
            measure(ts, client, server, 'ct*ct + sum, 65 (AV entry)', mul_ctct),
            measure(ts, client, server, 'square', lambda r: (lambda e: (e.square(), lvl(e), None))(enc(r.normal(size=D))))]
    for deg in (4, 7, 8, 15):
        rows.append(measure(ts, client, server, f'scale + polyval degree {deg}', poly(deg)))
    for k in (2, 4, 6, 8, 10):
        rows.append(measure(ts, client, server, f'Goldschmidt reciprocal, {k} steps', reciprocal(k)))
    for rd, k in ((4, 2), (8, 4)):
        rows.append(measure(ts, client, server, f'attention row (v1 circuit), root {rd}, {k} steps', softmax_row(rd, k)))
    return dict(parameters=dict(setting, backend='TenSEAL', version=ts.__version__, bootstrapping=False,
                                security='SEAL default tc128 parameter validation',
                                available_levels=len(setting['modulus_bits']) - 2), rows=rows)


def compose(rows, available):
    """Critical-path sum of measured depths for one block. Arithmetic, not a measured block."""
    L = {r['primitive']: r.get('levels') for r in rows if r['status'] == 'measured'}
    get = lambda k: L.get(k)
    out = {}
    for prof, deg, rd, k in (('budget', 7, 4, 2), ('accurate', 15, 8, 4)):
        poly_l = get(f'scale + polyval degree {deg}')
        att = get(f'attention row (v1 circuit), root {rd}, {k} steps')
        parts = dict(QKV=get('mm 192->576 (QKV)'), QK=get('ct.ct dot, 64 (QK^T entry)'), attention_row=att,
                     AV=get('ct*ct + sum, 65 (AV entry)'), proj=get('mm 192->192 (proj)'),
                     fc1=get('mm 192->768 (fc1)'), gelu=poly_l, fc2=get('mm 768->192 (fc2)'))
        # LN variance path: mean scaling (1) + square (1) + variance scaling (1) + inverse sqrt + multiply (1).
        ln = None if poly_l is None or get('square') is None else 3 + get('square') + poly_l
        for variant, lnv in (('layernorm_polynomial', ln), ('batchnorm_folded', 0)):
            terms = dict(parts, LN1=lnv, LN2=lnv)
            total = None if any(v is None for v in terms.values()) else sum(terms.values())
            out[f'{prof}_{variant}'] = dict(terms=terms, total_levels=total, available_levels=available,
                                            fits_without_bootstrapping=None if total is None else total <= available)
    out['scope'] = ('Sum of separately measured primitive depths along one block\'s critical path; LN term '
                    'assumes the listed decomposition. Not a measured block, not an accuracy result.')
    return out


def run(root, index):
    root = Path(root); setting = F0['settings'][index]
    out = root / 'f0' / f"f0_n{setting['ring_degree']}"; out.mkdir(parents=True, exist_ok=True)
    from io_utils import write
    try:
        r = ledger(setting)
        r['composition'] = compose(r['rows'], r['parameters']['available_levels'])
        write(out / 'report.json', r); return r
    except Exception as e:
        write(out / 'error.json', dict(error=str(e), traceback=traceback.format_exc())); raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('root', nargs='?'); ap.add_argument('index', nargs='?', type=int)
    ap.add_argument('--local-test', action='store_true'); a = ap.parse_args()
    if a.local_test:
        r = ledger(dict(ring_degree=16384, modulus_bits=[60, 40, 40, 40, 40, 40, 60], scale_bits=40), threads=2)
        r['composition'] = compose(r['rows'], 5)
        print(json.dumps(r, indent=1)[:4000])
    else:
        print(json.dumps(run(a.root, a.index)['composition'], indent=1))
