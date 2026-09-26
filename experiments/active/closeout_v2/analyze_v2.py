"""Aggregate closeout_v2 reports into paper-facing CSV/JSON tables (NumPy + stdlib only).

    python analyze_v2.py <run_root> --out <repo>/results/closeout_v2

Writes: e1_rows.csv, e1_summary.csv, e2_rows.csv, e2_spearman.csv, e3_rows.csv,
f0_rows.csv, f0_composition.json, f1_summary.json, screen_evaluation.csv (when present),
completion.json (expected vs recorded vs failed jobs, per group).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def reports(root, group):
    out = []
    for p in sorted((Path(root) / group).glob('*/report.json')):
        out.append(json.loads(p.read_text()))
    return out


def errors(root, group):
    return sorted(p.parent.name for p in (Path(root) / group).glob('*/error.json'))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def flat(rep):
    j = rep['job']; t = rep['splits'][j['split']]; g = rep['splits']['gate']
    row = {k: j.get(k) for k in ('name', 'kind', 'dataset', 'seed', 'split', 'profile', 'mask', 'variant', 'gelu_degree',
                                 'domain', 'exp_root_degree', 'reciprocal_steps', 'only_block', 'except_block')}
    ag = t['agreement']
    row.update(reference_accuracy=ag['reference_accuracy'], converted_accuracy=ag['converted_accuracy_invalid_as_failure'],
               accuracy_drop_pp=None if ag['reference_accuracy'] is None else ag['reference_accuracy'] - ag['converted_accuracy_invalid_as_failure'],
               change_rate=ag['change_rate'], change_rate_cp_upper=ag['change_rate_cp_upper'],
               correct_to_wrong=ag['correct_to_wrong'], wrong_to_correct=ag.get('wrong_to_correct'),
               converted_invalid=ag['converted_invalid'], probability_tv_mean=ag.get('probability_tv_mean'),
               centered_logit_error_max=ag.get('centered_logit_error_max'),
               margin_sufficient_fraction=(ag['margin_sufficient_count'] / ag['n_reference_finite']
                                           if ag.get('margin_sufficient_count') is not None and ag['n_reference_finite'] else None),
               kl_mean=ag['kl_mean_finite'], n=ag['n_reference_finite'],
               gate_change_rate=g['agreement']['change_rate'], gate_change_cp_upper=g['agreement']['change_rate_cp_upper'],
               gate_kl_mean=g['agreement']['kl_mean_finite'], gate_inputs_outside=g['inputs_with_any_outside_flag'],
               target_inputs_outside=t['inputs_with_any_outside_flag'])
    for k, v in g['primitive'].items():
        row['gate_' + k] = v
    for k, v in t['primitive'].items():
        row['target_' + k] = v
    if 'sanity' in rep:
        row['sanity_passed'] = rep['sanity']['passed']
    return row


def rank(x):
    x = np.asarray(x, float); order = x.argsort(); r = np.empty(len(x)); r[order] = np.arange(len(x))
    for v in np.unique(x):                      # average ties
        m = x == v; r[m] = r[m].mean()
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None, int(ok.sum())
    ra, rb = rank(a[ok]), rank(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return None, int(ok.sum())
    return float(np.corrcoef(ra, rb)[0, 1]), int(ok.sum())


def summarize(rows, keys, value_fields):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        d = dict(zip(keys, key)); d['n_seeds'] = len(rs)
        for f in value_fields:
            v = np.array([r[f] for r in rs if r[f] is not None], float)
            d[f + '_mean'] = float(v.mean()) if v.size else None
            d[f + '_sd'] = float(v.std(ddof=1)) if v.size > 1 else None
            d[f + '_per_seed'] = ';'.join(f'{x:.4g}' for x in v)
        out.append(d)
    return out


def main(root, out):
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    completion = {}
    for g in ('pilot', 'e1', 'e2', 'e3', 'confirm', 'med_e1', 'med_e2', 'f0', 'f1'):
        exp = len(manifest.get(g, []))
        rec = len(list((root / g).glob('*/report.json'))) if (root / g).exists() else 0
        completion[g] = dict(expected=exp, recorded=rec, failed=errors(root, g) if (root / g).exists() else [])
    completion['excluded_sources'] = manifest.get('excluded', {})
    (out / 'completion.json').write_text(json.dumps(completion, indent=1))
    pilots = [json.loads(p.read_text()) for p in sorted((root / 'pilot').glob('*/report.json'))] if (root / 'pilot').exists() else []
    if pilots:
        (out / 'pilot_checks.json').write_text(json.dumps({p['job']['name']: dict(passed=p['passed'], checks=p['checks'])
                                                           for p in pilots}, indent=1))

    e1 = [flat(r) for r in reports(root, 'e1')]
    if e1:
        write_csv(out / 'e1_rows.csv', e1)
        write_csv(out / 'e1_summary.csv', summarize(e1, ('profile', 'mask', 'variant'),
                                                    ('converted_accuracy', 'accuracy_drop_pp', 'change_rate')))
    e2 = [flat(r) for r in reports(root, 'e2')]
    if e2:
        write_csv(out / 'e2_rows.csv', e2)
        cands = [k for k in e2[0] if k.startswith('gate_') and isinstance(e2[0][k], (int, float))]
        sp = []
        for fam in sorted({r['mask'] for r in e2}):
            fr = [r for r in e2 if r['mask'] == fam]
            for k in cands + ['target_grid_sup_error_worst_site']:
                rho, n = spearman([r.get(k) if r.get(k) is not None else np.nan for r in fr],
                                  [r['change_rate'] for r in fr])
                sp.append(dict(family=fam, predictor=k, outcome='dev change_rate', spearman_rho=rho, n_configs=n))
        write_csv(out / 'e2_spearman.csv', sp)
    e3 = [flat(r) for r in reports(root, 'e3')]
    if e3:
        write_csv(out / 'e3_rows.csv', e3)
        write_csv(out / 'e3_summary.csv', summarize(e3, ('mask', 'only_block', 'except_block'),
                                                    ('accuracy_drop_pp', 'change_rate')))
    conf = [flat(r) for r in reports(root, 'confirm')]
    if conf:
        write_csv(out / 'confirm_rows.csv', conf)
    for g in ('med_e1', 'med_e2'):
        rows = [flat(r) for r in reports(root, g)]
        if rows:
            write_csv(out / f'{g}_rows.csv', rows)
            keys = ('dataset', 'profile', 'mask', 'variant') if g == 'med_e1' else \
                   ('dataset', 'mask', 'gelu_degree', 'domain', 'exp_root_degree', 'reciprocal_steps')
            write_csv(out / f'{g}_summary.csv', summarize(rows, keys, ('converted_accuracy', 'accuracy_drop_pp', 'change_rate')))
    f0 = reports(root, 'f0')
    if f0:
        rows = []
        for r in f0:
            for row in r['rows']:
                rows.append(dict(ring_degree=r['parameters']['ring_degree'], available_levels=r['parameters']['available_levels'],
                                 **{k: row.get(k) for k in ('primitive', 'status', 'levels', 'max_abs_error', 'seconds', 'error')}))
        write_csv(out / 'f0_rows.csv', rows)
        (out / 'f0_composition.json').write_text(json.dumps({str(r['parameters']['ring_degree']): r['composition'] for r in f0}, indent=1))
    f1 = root / 'f1'
    if f1.exists():
        parts = [json.loads(p.read_text()) for p in sorted(f1.glob('f1_b*_part*/report.json'))]
        s = dict(parts_recorded=len(parts), parts_failed=sorted(p.parent.name for p in f1.glob('f1_b*_part*/error.json')))
        if parts:
            s['ckks_minus_plaintext_max_abs'] = max(p['ckks_minus_plaintext_max_abs']['max'] for p in parts)
            s['levels_consumed'] = sorted({l for p in parts for l in p['levels_consumed']})
            s['server_seconds_median_of_medians'] = float(np.median([p['server_seconds']['median'] for p in parts]))
            s['parameters'] = parts[0]['parameters']
        if (f1 / 'f1_report.json').exists():
            s['hybrid_replay'] = json.loads((f1 / 'f1_report.json').read_text())
        (out / 'f1_summary.json').write_text(json.dumps(s, indent=1))
    if (root / 'screen_evaluation.json').exists():
        ev = json.loads((root / 'screen_evaluation.json').read_text())
        write_csv(out / 'screen_evaluation.csv', ev['rows'])
        (out / 'screen_frozen.json').write_text((root / 'screen_frozen.json').read_text())
    print(json.dumps(completion, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--out', required=True)
    a = ap.parse_args(); main(a.root, a.out)
