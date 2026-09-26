"""closeout_v2 tests. Run with the frozen v1 code on the path, e.g. from a run root's code/:
    python -m unittest test_v2 -v
Torch-dependent and TenSEAL-dependent classes skip cleanly in environments without them.
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_TS = importlib.util.find_spec('tenseal') is not None


class PlanTests(unittest.TestCase):
    def test_schedule_sizes_and_unique_names(self):
        from protocol_v2 import schedule
        p = schedule()
        self.assertEqual({k: len(v) for k, v in p.items()},
                         dict(pilot=4, e1=48, e2=81, e3=72, confirm=81, med_e1=144, med_e2=243, f0=2, f1=32))
        names = [j['name'] for g in p.values() for j in g]
        self.assertEqual(len(names), len(set(names)))

    def test_confirm_mirrors_e2(self):
        from protocol_v2 import schedule
        p = schedule()
        strip = lambda j: {k: v for k, v in j.items() if k not in ('name', 'kind', 'split')}
        self.assertEqual([strip(j) for j in p['e2']], [strip(j) for j in p['confirm']])
        self.assertTrue(all(j['split'] == 'test' for j in p['confirm']))
        self.assertTrue(all(j['split'] == 'dev' for j in p['e1'] + p['e2'] + p['e3'] + p['med_e1'] + p['med_e2']))
        self.assertTrue(all(j['dtype'] == 'float64' for j in p['med_e1'] + p['med_e2']))

    def test_agreement_metrics(self):
        from probes import agreement
        ref = np.array([[3., 1., 0.], [0., 2., 1.9], [1., 0., 0.]])
        same_but_shifted = ref + 5.0                      # shared offset: no semantic change
        a = agreement(ref, same_but_shifted, np.array([0, 1, 0]))
        self.assertEqual(a['prediction_changes'], 0)
        self.assertAlmostEqual(a['centered_logit_error_max'], 0.0)
        self.assertEqual(a['margin_sufficient_count'], 3)
        self.assertAlmostEqual(a['probability_tv_mean'], 0.0)
        flipped = ref.copy(); flipped[1] = [0., 1.8, 2.0]
        b = agreement(ref, flipped, np.array([0, 1, 0]))
        self.assertEqual(b['prediction_changes'], 1); self.assertEqual(b['correct_to_wrong'], 1)
        self.assertEqual(b['margin_sufficient_count'], 2)

    def test_clopper_pearson(self):
        from probes import clopper_pearson_upper
        self.assertAlmostEqual(clopper_pearson_upper(0, 100), 1 - 0.05 ** (1 / 100), places=6)
        self.assertEqual(clopper_pearson_upper(5, 5), 1.0)
        self.assertLess(clopper_pearson_upper(1, 2048), 0.01)
        self.assertGreater(clopper_pearson_upper(30, 2048), 0.01)

    def test_goldschmidt_bound_from_table_domains(self):
        """The reciprocal bound the draft prints but never evaluates (Table I domains)."""
        bound = lambda a, b, k: ((b - a) / (b + a)) ** (2 ** (k + 1))
        self.assertAlmostEqual(bound(0.01166, 1.66655, 2), 0.894, places=2)
        self.assertAlmostEqual(bound(0.01017, 1.49650, 4), 0.647, places=2)


@unittest.skipUnless(HAS_TORCH, 'torch not installed')
class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from pvit_lab.model import ViT, legacy_spec
        from pvit_lab.data import loader
        from worker_v2 import splits
        torch.set_num_threads(2); torch.manual_seed(1)
        cls.torch = torch
        cls.m = ViT(legacy_spec('000')).eval()
        cls.fit, cls.gate, cls.dev, _ = splits('.', synthetic=True)
        cls.loader = staticmethod(loader)

    def batches(self, ds):
        return lambda: self.loader(ds, workers=0)

    def test_v1_parity(self):
        import polynomial as P
        from variants import calibrate_v2
        for prof in ('budget', 'accurate'):
            a, _ = P.calibrate(self.m, self.loader(self.fit, workers=0), 'cpu', prof)
            b, _, _ = calibrate_v2(self.m, self.batches(self.fit), 'cpu', dict(profile=prof))
            self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_calibration_through_worker_cast_matches_v1(self):
        """Regression: calibrate() makes two passes; a one-shot batch stream broke the second."""
        import polynomial as P
        from variants import calibrate_v2
        from worker_v2 import cast
        a, _ = P.calibrate(self.m, self.loader(self.fit, workers=0), 'cpu', 'budget')
        b, _, _ = calibrate_v2(self.m, lambda: cast(self.loader(self.fit, workers=0), self.torch.float32), 'cpu',
                               dict(profile='budget'))
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        self.assertTrue(all(np.isfinite(v['reciprocal']['hi']) for v in b.values() if v['kind'] == 'attention'))

    def test_exact_both_reproduces_source(self):
        from variants import calibrate_v2, convert_v2
        p, _, _ = calibrate_v2(self.m, self.batches(self.fit), 'cpu', dict(profile='budget'))
        conv, rec, sites = convert_v2(self.m, p, 'attention', 'exact_both')
        self.assertEqual(len(sites), 6)
        x = next(iter(self.loader(self.gate, workers=0)))[0]
        with self.torch.no_grad():
            rec.begin(len(x), x.device)
            d = (conv(x) - self.m(x)).abs().max().item()
        self.assertLess(d, 1e-4)

    def test_block_filters(self):
        from variants import calibrate_v2, convert_v2
        p, _, _ = calibrate_v2(self.m, self.batches(self.fit), 'cpu', dict(profile='budget'))
        _, _, only = convert_v2(self.m, p, 'gelu', only_block=3)
        _, _, but = convert_v2(self.m, p, 'gelu', except_block=3)
        self.assertEqual(only, ['blocks.3.act'])
        self.assertEqual(len(but), 5); self.assertNotIn('blocks.3.act', but)

    def test_central_domain_refit(self):
        from variants import calibrate_v2
        a, _, _ = calibrate_v2(self.m, self.batches(self.fit), 'cpu', dict(profile='budget'))
        b, _, meta = calibrate_v2(self.m, self.batches(self.fit), 'cpu', dict(profile='budget', domain='central'))
        g = [k for k in a if a[k]['kind'] == 'gelu']
        self.assertTrue(all(b[k]['poly']['hi'] - b[k]['poly']['lo'] < a[k]['poly']['hi'] - a[k]['poly']['lo'] for k in g))
        self.assertEqual(set(meta['central_refit']), set(g))

    def test_worker_and_screen_end_to_end(self):
        from worker_v2 import selftest
        r = selftest()
        rep = next(v for k, v in r.items() if 'exact_both' in k and 'mask-attention' in k)
        self.assertTrue(rep['sanity']['passed'])
        prim = rep['splits']['gate']['primitive']
        for k in ('grid_sup_error_worst_site', 'converted_mean_abs_worst_site', 'reciprocal_bound_worst_site'):
            self.assertIn(k, prim)


@unittest.skipUnless(HAS_TORCH, 'torch not installed')
class PilotAndMedicalTests(unittest.TestCase):
    def test_cifar_pilot_gate_passes_on_correct_code(self):
        from io_utils import write
        from protocol_v2 import pilot_jobs
        import worker_v2
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write(root / 'manifest.json', dict(data_root='.', sources={}, pilot=[dict(pilot_jobs()[0], images=8)]))
            r = worker_v2.run(root, 'pilot', 0, synthetic=True)
            self.assertTrue(r['passed'])

    def test_medical_adapter_prepare_pilot_and_parity(self):
        """Fake medical run: npz data, arm-000 checkpoint, stored fit made with the frozen calibrate
        exactly as archive/submission_v2/scripts/medical_conversion.py does."""
        import os, dataclasses, torch
        import polynomial as P
        from io_utils import write, sha
        from pvit_lab.model import ViT, legacy_spec
        from pvit_lab.data import loader
        from medical_data import Medical
        from protocol_v2 import pilot_jobs, med_e1_jobs, MEDICAL_FIT_RNG
        import prepare_v2, worker_v2
        os.environ['PVIT_DEVICE'] = 'cpu'
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); data = d / 'data'; data.mkdir(); run = d / 'medrun'
            for name in ('bloodmnist', 'pathmnist', 'dermamnist'):
                np.savez(data / f'{name}.npz', train_images=rng.integers(0, 255, (4200, 28, 28, 3), dtype=np.uint8),
                         train_labels=rng.integers(0, 8, (4200, 1)), val_images=rng.integers(0, 255, (40, 28, 28, 3), dtype=np.uint8),
                         val_labels=rng.integers(0, 8, (40, 1)), test_images=np.zeros((1, 28, 28, 3), np.uint8), test_labels=np.zeros((1, 1)))
            torch.manual_seed(3); spec = legacy_spec('000', 8); model = ViT(spec).eval()
            jid = 'bloodmnist_s42_000'
            (run / 'train' / jid).mkdir(parents=True); ck = run / 'train' / jid / 'best.pth'
            torch.save(dict(state_dict=model.state_dict(), model_spec=dataclasses.asdict(spec)), ck)
            write(run / 'train' / jid / 'status.json', dict(status='COMPLETED', checkpoint_sha256=sha(ck)))
            write(run / 'manifest.json', dict(jobs=[dict(id=jid, dataset='bloodmnist', seed=42, arm='000'),
                                                   dict(id='pathmnist_s42_000', dataset='pathmnist', seed=42, arm='000')]))
            order = np.random.default_rng(MEDICAL_FIT_RNG).permutation(4200)
            fitds = Medical(data, 'bloodmnist', 'train', indices=order[:2048])
            m64 = model.double()
            batches = [(x.double(), y, i) for x, y, i in loader(fitds, workers=0)]
            (run / 'conversion_fit' / jid).mkdir(parents=True)
            for prof in ('budget', 'accurate'):
                params, _ = P.calibrate(m64, batches, 'cpu', prof)
                write(run / 'conversion_fit' / jid / f'{prof}_all.json', dict(parameters=params))
            srcs, missing, mrun, mdata = prepare_v2.medical_sources(run, data, sha)
            self.assertIn('bloodmnist_s42_000', srcs); self.assertIn('pathmnist_s42_000', missing)
            pilot = [dict(j, images=8) for j in pilot_jobs() if j['dataset'] == 'bloodmnist']
            e1 = [j for j in med_e1_jobs() if j['dataset'] == 'bloodmnist' and j['seed'] == 42][:2]
            write(d / 'root' / 'manifest.json', dict(data_root='.', medical_data_root=str(data), sources=srcs, pilot=pilot, med_e1=e1))
            r = worker_v2.run(d / 'root', 'pilot', 0, synthetic=False, synthetic_model=False)
            self.assertTrue(r['passed']); self.assertTrue(r['checks']['medical_fit_parity_budget']['passed'])
            rep = worker_v2.run(d / 'root', 'med_e1', 0, synthetic=False, synthetic_model=False)
            self.assertTrue(rep['medical_fit_parity']['passed'])
            self.assertEqual(rep['splits']['dev']['agreement']['n_reference_finite'], 40)


@unittest.skipUnless(HAS_TS, 'tenseal not installed')
class FHETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tenseal as ts
        from fhe_common_v2 import contexts
        cls.ts = ts
        cls.client, cls.server = contexts(ts, 16384, [60, 40, 40, 40, 40, 40, 60], 40, threads=2)

    def test_server_has_no_secret_key(self):
        self.assertFalse(self.server.has_secret_key())

    def test_encrypted_mlp_matches_plaintext(self):
        from fhe_common_v2 import mlp_weights, plaintext_mlp, encrypted_mlp
        rng = np.random.default_rng(0)
        d, m = 16, 64
        w = mlp_weights(rng.normal(size=(m, d)) * .1, rng.normal(size=m) * .1, rng.normal(size=(d, m)) * .1,
                        rng.normal(size=d) * .1, dict(center=.1, radius=3., coefficients=list(rng.normal(size=8) * .3 ** np.arange(8))))
        h, x = rng.normal(size=d), rng.normal(size=d)
        y, cost = encrypted_mlp(self.ts, self.client, self.server, h, x, w)
        self.assertLess(np.abs(y - plaintext_mlp(h, x, w)).max(), 1e-3)
        self.assertEqual(cost['intermediate_decryptions'], 0)
        self.assertEqual(cost['levels_consumed'], 5)

    def test_list_bias_breaks_following_mm(self):
        """Regression for the TenSEAL layout pitfall documented in fhe_common_v2."""
        ts = self.ts; rng = np.random.default_rng(1)
        h = rng.normal(size=16); W1 = rng.normal(size=(16, 64)) * .1; b = rng.normal(size=64); W2 = rng.normal(size=(64, 16)) * .1
        ref = (h @ W1 + b) @ W2
        e = ts.ckks_vector(self.client, h.tolist())
        bad = np.abs(np.array((e.mm(W1.tolist()) + b.tolist()).mm(W2.tolist()).decrypt()) - ref).max()
        good = np.abs(np.array((e.mm(W1.tolist()) + ts.ckks_vector(self.client, b.tolist())).mm(W2.tolist()).decrypt()) - ref).max()
        self.assertLess(good, 1e-3)
        self.assertGreater(bad, 10 * good)


if __name__ == '__main__':
    unittest.main()
