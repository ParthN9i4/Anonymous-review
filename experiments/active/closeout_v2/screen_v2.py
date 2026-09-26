"""E4: prospective configuration screens (freeze on dev seed 42, then evaluate).

A configuration is ACCEPTABLE if its prediction-change rate against the reference on the
target split is <= ACCEPT_FLIP_RATE (invalid converted outputs count as changes). The
target is fixed a priori in protocol_v2.py.

Each screen may use only pre-deployment information: the fitted parameters and the GATE
split (training images 2048-4095), never the dev or test outcome of the configuration
being judged. Rules:

  v1_static          gate v1-failure rate (flip | invalid | max logit err > 0.1) <= 1%   (as v1)
  domain_coverage    no gate input outside any fitted interval                           (as v1 'range')
  gate_change_cp     Clopper-Pearson 95% upper bound of the gate change rate <= 1%       (threshold-free)
  grid_sup           worst-site fitted-interval sup-error <= tau
  source_mean_abs    worst-site mean |P - O| on gate source-path inputs <= tau
  source_p99_abs     worst-site p99 |P - O| on gate source-path inputs <= tau
  converted_p99_abs  worst-site p99 |P - O| on gate converted-path inputs <= tau
  gate_kl            mean KL(reference || converted) on gate <= tau

tau is chosen per operator family (gelu / attention) on DEV_SEED dev only: the largest
observed value that admits no unacceptable development configuration. It is then frozen
(screen_frozen.json, SHA-256 printed) BEFORE seeds 43/44 are scored and before any test
split is opened.

    python screen_v2.py freeze   <run_root>
    python screen_v2.py evaluate <run_root>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from protocol_v2 import ACCEPT_FLIP_RATE, SECONDARY_FLIP_RATE, DEV_SEED, CP_ALPHA
from probes import clopper_pearson_upper

TAU_RULES = {'grid_sup': ('primitive', 'grid_sup_error_worst_site'),
             'source_mean_abs': ('primitive', 'source_mean_abs_worst_site'),
             'source_p99_abs': ('primitive', 'source_p99_abs_worst_site'),
             'converted_p99_abs': ('primitive', 'converted_p99_abs_worst_site'),
             'gate_kl': ('agreement', 'kl_mean_finite')}
FIXED_RULES = ('v1_static', 'domain_coverage', 'gate_change_cp')


def v1_gate_failure_rate(job_dir):
    a = np.load(job_dir / 'gate_predictions.npz')
    r, c = a['reference'].astype(np.float64), a['converted'].astype(np.float64)
    rf, cf = np.isfinite(r).all(1), np.isfinite(c).all(1); both = rf & cf
    err = np.full(len(r), np.inf); err[both] = np.abs(r[both] - c[both]).max(1)
    flip = np.zeros(len(r), bool); flip[both] = r[both].argmax(1) != c[both].argmax(1)
    bad = rf & (~cf | flip | (err > 0.1))
    return float(bad[rf].mean()) if rf.any() else 1.0


def load_rows(root, group):
    rows = []
    for rep in sorted((Path(root) / group).glob('*/report.json')):
        d = json.loads(rep.read_text())
        j = d['job']; target = d['splits'][j['split']]; gate = d['splits']['gate']
        row = dict(name=j['name'], seed=j['seed'], family=j['mask'], split=j['split'],
                   target_change_rate=target['agreement']['change_rate'],
                   gate=gate, report_dir=rep.parent,
                   config={k: j[k] for k in ('gelu_degree', 'domain', 'exp_root_degree', 'reciprocal_steps') if k in j})
        row['acceptable'] = row['target_change_rate'] is not None and row['target_change_rate'] <= ACCEPT_FLIP_RATE
        row['acceptable_5pct'] = row['target_change_rate'] is not None and row['target_change_rate'] <= SECONDARY_FLIP_RATE
        row['v1_static_rate'] = v1_gate_failure_rate(rep.parent)
        rows.append(row)
    return rows


def metric(row, rule):
    sec, key = TAU_RULES[rule]
    v = row['gate'][sec].get(key)
    return np.inf if v is None or not np.isfinite(v) else float(v)


def fixed_accept(row, rule):
    g = row['gate']
    if rule == 'v1_static':
        return row['v1_static_rate'] <= 0.01
    if rule == 'domain_coverage':
        return g['inputs_with_any_outside_flag'] == 0
    if rule == 'gate_change_cp':
        u = g['agreement']['change_rate_cp_upper']
        return u is not None and u <= ACCEPT_FLIP_RATE
    raise KeyError(rule)


def freeze(root):
    rows = [r for r in load_rows(root, 'e2') if r['seed'] == DEV_SEED]
    if not rows:
        sys.exit('No e2 dev reports for the development seed; nothing to freeze.')
    frozen = dict(dev_seed=DEV_SEED, accept_flip_rate=ACCEPT_FLIP_RATE, fixed_rules=list(FIXED_RULES),
                  thresholds={}, development_counts={},
                  rule_text=__doc__, source_reports=sorted(r['name'] for r in rows))
    for fam in sorted({r['family'] for r in rows}):
        fr = [r for r in rows if r['family'] == fam]
        frozen['development_counts'][fam] = dict(configs=len(fr), acceptable=sum(r['acceptable'] for r in fr))
        for rule in TAU_RULES:
            vals = [(metric(r, rule), r['acceptable']) for r in fr]
            bad = [v for v, ok in vals if not ok]
            limit = min(bad) if bad else np.inf
            admissible = [v for v, ok in vals if ok and v < limit]
            frozen['thresholds'][f'{fam}:{rule}'] = max(admissible) if admissible else None
    out = Path(root) / 'screen_frozen.json'
    if out.exists():
        sys.exit('screen_frozen.json already exists; the screen is frozen once.')
    from io_utils import write, sha
    write(out, frozen)
    print('FROZEN', out, 'sha256', sha(out))
    return frozen


def evaluate(root):
    root = Path(root)
    frozen = json.loads((root / 'screen_frozen.json').read_text())
    cohorts = {'dev_seeds_43_44': [r for r in load_rows(root, 'e2') if r['seed'] != DEV_SEED],
               'dev_seed_42_development': [r for r in load_rows(root, 'e2') if r['seed'] == DEV_SEED],
               'test_confirm_all_seeds': load_rows(root, 'confirm')}
    table = []
    for cohort, rows in cohorts.items():
        for fam in sorted({r['family'] for r in rows}):
            fr = [r for r in rows if r['family'] == fam]
            for rule in list(FIXED_RULES) + list(TAU_RULES):
                if rule in TAU_RULES:
                    tau = frozen['thresholds'].get(f'{fam}:{rule}')
                    acc = [tau is not None and metric(r, rule) <= tau for r in fr]
                else:
                    tau = None; acc = [fixed_accept(r, rule) for r in fr]
                n = len(fr); a = sum(acc)
                fa = sum(1 for r, x in zip(fr, acc) if x and not r['acceptable'])
                miss = sum(1 for r, x in zip(fr, acc) if not x and r['acceptable'])
                table.append(dict(cohort=cohort, family=fam, rule=rule, threshold=tau, configs=n,
                                  acceptable_configs=sum(r['acceptable'] for r in fr),
                                  accepted=a, coverage=a / n if n else None,
                                  false_accepts=fa, false_accept_rate_among_accepted=fa / a if a else None,
                                  false_accept_cp_upper=clopper_pearson_upper(fa, a, CP_ALPHA) if a else None,
                                  missed_acceptable=miss))
    from io_utils import write
    write(root / 'screen_evaluation.json', dict(frozen_thresholds=frozen['thresholds'], rows=table,
          scope='Configuration-level decisions; configurations share checkpoints and fits, so counts are not '
                'independent trials. Test cohort scored once, after freezing.'))
    return table


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'evaluate']); ap.add_argument('root')
    a = ap.parse_args()
    if a.action == 'freeze':
        print(json.dumps(freeze(a.root)['thresholds'], indent=1))
    else:
        for r in evaluate(a.root):
            print({k: r[k] for k in ('cohort', 'family', 'rule', 'accepted', 'configs', 'false_accepts', 'missed_acceptable')})
