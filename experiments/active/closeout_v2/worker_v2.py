"""One closeout_v2 plaintext job: calibrate, convert, replay gate + (dev | test), record.

Usage (inside a prepared run root, see prepare_v2.py):
    python worker_v2.py <run_root> <group> <index>        group in pilot, e1, e2, e3, confirm, med_e1, med_e2
    python worker_v2.py --synthetic-selftest               CPU, no data, no checkpoints
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from io_utils import sha, write, env                        # frozen v1 helpers
from metrics import classification
from pvit_lab.common import seed_all, device_for
from pvit_lab.data import datasets, loader, Indexed
from protocol_v2 import FIT_N, GATE_N, GATE_OFFSET, TEST_N, CP_ALPHA
from variants import calibrate_v2, convert_v2
from probes import Probes, aggregate_sites, agreement

CONFIG_KEYS = ('profile', 'gelu_degree', 'norm_degree', 'exp_root_degree', 'reciprocal_steps', 'domain')


def splits(root, synthetic=False, dataset='cifar10', medical_root=None):
    """(fit, gate, dev, test). Medical: fit/gate indices exactly as in
    archive/submission_v2/scripts/medical_conversion.py; dev = official validation split;
    test is never opened by this campaign for medical data."""
    if dataset != 'cifar10' and not synthetic:
        from medical_data import Medical
        from protocol_v2 import MEDICAL_FIT_RNG
        n = len(Medical(medical_root, dataset, 'train'))
        order = np.random.default_rng(MEDICAL_FIT_RNG).permutation(n)
        return (Medical(medical_root, dataset, 'train', indices=order[:FIT_N]),
                Medical(medical_root, dataset, 'train', indices=order[GATE_OFFSET:GATE_OFFSET + GATE_N]),
                Medical(medical_root, dataset, 'val'), None)
    _, val, clean = datasets(root, 'cifar10', synthetic=synthetic)
    if synthetic:
        return Subset(clean, range(8)), Subset(clean, range(8, 16)), val, clean
    from torchvision import datasets as D, transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((.4914, .4822, .4465), (.2470, .2435, .2616))])
    test = Indexed(D.CIFAR10(root, train=False, download=False, transform=tf), range(TEST_N))
    return Subset(clean, range(FIT_N)), Subset(clean, range(GATE_OFFSET, GATE_OFFSET + GATE_N)), val, test


def load_model(src, synthetic=False):
    if synthetic:
        from pvit_lab.model import ViT, legacy_spec
        torch.manual_seed(1)
        return ViT(legacy_spec('000')).eval()
    from worker import load  # frozen v1 loader: verifies checkpoint SHA-256 before use
    return load(src)


def source_key(j):
    d = j.get('dataset', 'cifar10')
    return f"regularized_ln_seed{j['seed']}" if d == 'cifar10' else f"{d}_s{j['seed']}_000"


class cast:
    """Re-iterable dtype cast over a loader. Must be re-iterable: the frozen calibrate()
    makes two passes (activation bounds, then reciprocal domains). A one-shot generator
    would silently leave the reciprocal interval at +-inf."""

    def __init__(self, batches, dtype):
        self.batches, self.dtype = batches, dtype

    def __iter__(self):
        for x, y, i in self.batches:
            yield x.to(self.dtype), y, i


def max_rel_diff(a, b):
    """Largest relative difference between two parameter trees (inf on structural mismatch)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return float('inf')
        return max([max_rel_diff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return float('inf')
        return max([max_rel_diff(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) / max(1.0, abs(b))
    return 0.0 if a == b else float('inf')


def json_normalized(tree):
    """The form io_utils.write stores: numpy scalars as Python numbers, non-finite floats as None."""
    def clean(x):
        if isinstance(x, np.generic):
            x = x.item()
        if isinstance(x, float) and not np.isfinite(x):
            return None
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [clean(v) for v in x]
        return x
    return clean(tree)


def medical_parity(src, params, profile):
    """Recalibrated medical fit must equal the stored medical_replication_v1 fit."""
    stored = json.loads((Path(src['fit_dir']) / f'{profile}_all.json').read_text())['parameters']
    d = max_rel_diff(json_normalized(params), stored)
    return dict(stored_fit=str(Path(src['fit_dir']) / f'{profile}_all.json'), max_relative_difference=d,
                passed=d <= 1e-9)


@torch.no_grad()
def replay(base, conv, rec, params, sites, batches, device, dtype=torch.float32):
    probes = Probes(base, conv, params, sites)
    ref, out, labels, ids, outside = [], [], [], [], []
    try:
        for x, y, i in batches:
            x = x.to(device, dtype)
            r = base(x)
            rec.begin(len(x), x.device)
            z = conv(x)
            ref.append(r.cpu().numpy()); out.append(z.cpu().numpy())
            labels.append(y.numpy()); ids.append(i.numpy()); outside.append(rec.flags.cpu().numpy())
    finally:
        probes.remove()
    arrays = dict(reference=np.concatenate(ref), converted=np.concatenate(out), labels=np.concatenate(labels),
                  ids=np.concatenate(ids), outside=np.concatenate(outside))
    return arrays, probes.summary()


def run(root, group, index, synthetic=False, synthetic_model=None):
    synthetic_model = synthetic if synthetic_model is None else synthetic_model
    root = Path(root)
    m = json.loads((root / 'manifest.json').read_text())
    j = m[group][index]
    if j.get('kind') == 'pilot':
        return run_pilot(root, m, group, j, synthetic, synthetic_model)
    out = root / group / j['name']
    out.mkdir(parents=True, exist_ok=False)
    try:
        if j['split'] == 'test':
            frozen = root / 'screen_frozen.json'
            if not frozen.is_file():
                raise RuntimeError('Confirm jobs require screen_frozen.json (freeze the screen first)')
            screen_sha = sha(frozen)
        else:
            screen_sha = None
        seed_all(j['seed'])
        device = device_for(os.environ.get('PVIT_DEVICE', 'cpu' if synthetic_model else 'cuda:0'))
        dtype = torch.float64 if j.get('dtype') == 'float64' else torch.float32
        src = None if synthetic_model else m['sources'][source_key(j)]
        base = load_model(src, synthetic_model).to(device, dtype).eval()
        fit, gate, dev, test = splits(m['data_root'], synthetic, j.get('dataset', 'cifar10'), m.get('medical_data_root'))
        workers = 0 if synthetic else 2
        cfg = {k: j[k] for k in CONFIG_KEYS if k in j}
        t0 = time.perf_counter()
        params, fit_ids, meta = calibrate_v2(base, lambda: cast(loader(fit, workers=workers), dtype), device, cfg)
        fit_seconds = time.perf_counter() - t0
        parity = None
        if src and 'fit_dir' in src and set(cfg) <= {'profile'}:
            parity = medical_parity(src, params, cfg.get('profile', 'budget'))
            if not parity['passed']:
                raise ValueError(f"Medical recalibration differs from the stored fit: {parity}")
        conv, rec, sites = convert_v2(base, params, j['mask'], j.get('variant', 'poly'),
                                      j.get('only_block'), j.get('except_block'))
        src_w, conv_w = dict(base.named_parameters()), dict(conv.named_parameters())
        if src_w.keys() != conv_w.keys() or any(not torch.equal(v, conv_w[k]) for k, v in src_w.items()):
            raise ValueError('Learned weights changed during conversion')
        write(out / 'params.json', dict(parameters=params, fit_image_ids=fit_ids, calibration=meta,
                                        converted_sites=sites))
        report = dict(status='measured', job=j, converted_sites=sites, calibration=meta,
                      fit_seconds=fit_seconds, params_sha256=sha(out / 'params.json'),
                      screen_frozen_sha256=screen_sha, medical_fit_parity=parity, splits={})
        for split, ds in (('gate', gate), (j['split'], test if j['split'] == 'test' else dev)):
            t0 = time.perf_counter()
            a, per_site = replay(base, conv, rec, params, sites, loader(ds, workers=workers), device, dtype)
            np.savez_compressed(out / f'{split}_predictions.npz', **a)
            report['splits'][split] = dict(
                seconds=time.perf_counter() - t0,
                agreement=agreement(a['reference'], a['converted'], a['labels'], CP_ALPHA),
                reference=classification(a['reference'], a['labels']),
                converted=classification(a['converted'], a['labels']),
                inputs_with_any_outside_flag=int(a['outside'].sum()),
                primitive=aggregate_sites(per_site), primitive_per_site=per_site)
        if j.get('variant') == 'exact_both' and j['mask'] == 'attention':
            ag = report['splits'][j['split']]['agreement']
            report['sanity'] = dict(check='exact_both attention must reproduce the source predictions',
                                    prediction_changes=ag['prediction_changes'], passed=ag['prediction_changes'] == 0)
        report['environment'] = env()
        report['scope'] = ('Frozen weights; plaintext floating-point replay. Oracle variants are attribution '
                           'controls, not deployable arithmetic. dev = validation split; test is opened only by '
                           'confirm jobs after the screen is frozen.')
        write(out / 'report.json', report)
        return report
    except Exception as e:
        write(out / 'error.json', dict(error=str(e), traceback=traceback.format_exc()))
        raise


@torch.no_grad()
def run_pilot(root, m, group, j, synthetic=False, synthetic_model=False):
    """Correctness gate on the real checkpoint before any array runs. Exits non-zero on failure.

    1. No-op conversion (no site replaced) must reproduce the source logits bit-for-bit.
    2. Polynomial attention with exact numerator AND exact division must reproduce the source
       predictions (0 changes) with max logit difference <= 1e-4.
    3. Medical: the recalibrated budget and accurate fits must equal the stored v1 medical fits.
    Deployed polynomial variants are also recorded, for information only.
    """
    out = root / group / j['name']; out.mkdir(parents=True, exist_ok=False)
    try:
        seed_all(j['seed'])
        device = device_for(os.environ.get('PVIT_DEVICE', 'cpu' if synthetic_model else 'cuda:0'))
        dtype = torch.float64 if j.get('dtype') == 'float64' else torch.float32
        src = None if synthetic_model else m['sources'][source_key(j)]
        base = load_model(src, synthetic_model).to(device, dtype).eval()
        fit, _, dev, _ = splits(m['data_root'], synthetic, j.get('dataset', 'cifar10'), m.get('medical_data_root'))
        workers = 0 if synthetic else 2
        x, y, _ = next(iter(loader(Subset(dev, range(min(j['images'], len(dev)))), batch=j['images'], workers=workers)))
        x = x.to(device, dtype)
        ref = base(x)
        checks, info = {}, {}
        for profile in ('budget', 'accurate'):
            params, _, _ = calibrate_v2(base, lambda: cast(loader(fit, workers=workers), dtype), device, dict(profile=profile))
            if src and 'fit_dir' in src:
                checks[f'medical_fit_parity_{profile}'] = medical_parity(src, params, profile)
            noop, rec, sites = convert_v2(base, params, 'none')
            rec.begin(len(x), x.device)
            checks[f'noop_bitwise_{profile}'] = dict(sites=len(sites), passed=len(sites) == 0 and torch.equal(noop(x), ref))
            conv, rec, _ = convert_v2(base, params, 'attention', 'exact_both')
            rec.begin(len(x), x.device); z = conv(x)
            d = float((z - ref).abs().max())
            checks[f'exact_both_attention_{profile}'] = dict(max_logit_difference=d,
                prediction_changes=int((z.argmax(1) != ref.argmax(1)).sum()),
                passed=d <= 1e-4 and bool((z.argmax(1) == ref.argmax(1)).all()))
            for mask in ('attention', 'all', 'gelu'):
                conv, rec, _ = convert_v2(base, params, mask)
                rec.begin(len(x), x.device); z = conv(x)
                info[f'{profile}_{mask}'] = agreement(ref.cpu().numpy(), z.cpu().numpy(), y.numpy(), CP_ALPHA)
        passed = all(c['passed'] for c in checks.values())
        write(out / 'report.json', dict(status='measured', job=j, passed=passed, checks=checks,
                                        deployed_variants_information_only=info, environment=env()))
        if not passed:
            raise SystemExit(f"Pilot gate FAILED: {json.dumps(checks)}")
        return dict(passed=passed, checks=checks)
    except SystemExit:
        raise
    except Exception as e:
        write(out / 'error.json', dict(error=str(e), traceback=traceback.format_exc()))
        raise


def selftest():
    """Every job kind on synthetic data (CPU). Used by test_v2.py and the GPU smoke job."""
    import tempfile
    from protocol_v2 import schedule
    plan = schedule()
    picks = dict(e1=[j for j in plan['e1'] if j['seed'] == 42][:8], e2=plan['e2'][:1] + plan['e2'][-1:],
                 e3=plan['e3'][:2])
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        manifest = dict(data_root='.', sources={}, **{k: v for k, v in picks.items()})
        write(root / 'manifest.json', manifest)
        results = {}
        for g, jobs in picks.items():
            for i in range(len(jobs)):
                results[jobs[i]['name']] = run(root, g, i, synthetic=True)
        return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?'); ap.add_argument('group', nargs='?'); ap.add_argument('index', nargs='?', type=int)
    ap.add_argument('--synthetic-selftest', action='store_true')
    a = ap.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if a.synthetic_selftest:
        r = selftest()
        print(json.dumps({k: v['splits'][v['job']['split']]['agreement']['change_rate'] for k, v in r.items()}, indent=1))
    else:
        run(a.root, a.group, a.index)
