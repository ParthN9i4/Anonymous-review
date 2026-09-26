"""Create an isolated closeout_v2 run root on H200 with verified checkpoint bytes.

    python prepare_v2.py --repo ~/Private-ViT-FHE --results ~/pvit_results [--data-root DIR]
        [--medical-run ~/pvit_results/medical_replication_v1 --medical-data DIR_WITH_NPZ]

Without --medical-run the medical groups (med_e1, med_e2, medical pilots) are removed
from the manifest and recorded as "not configured". A medical checkpoint whose training
did not complete, or whose bytes changed, removes its jobs and is recorded under
`excluded` (never imputed).

Copies the frozen v1 campaign code (results/campaign_2026-09-20/source_final/) and this
package into <run_root>/code, writes manifest.json (the full job schedule, resolved
checkpoints, data root, protocol hashes) and source_hashes.json (checked by every job).
Prints the run root. Existing files are never modified.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def medical_sources(run, data, sha):
    run = Path(run).expanduser().resolve(); data = Path(data).expanduser().resolve()
    m = json.loads((run / 'manifest.json').read_text())
    sources, missing = {}, {}
    for j in m['jobs']:
        if j['arm'] != '000':
            continue
        key = f"{j['dataset']}_s{j['seed']}_000"
        status_p = run / 'train' / j['id'] / 'status.json'; ck = run / 'train' / j['id'] / 'best.pth'
        fit = run / 'conversion_fit' / j['id']
        st = json.loads(status_p.read_text()) if status_p.is_file() else {}
        if st.get('status') != 'COMPLETED':
            missing[key] = f"training status {st.get('status', 'absent')}"; continue
        if not ck.is_file() or sha(ck) != st.get('checkpoint_sha256'):
            missing[key] = 'checkpoint missing or SHA-256 mismatch'; continue
        if not (fit / 'budget_all.json').is_file() or not (fit / 'accurate_all.json').is_file():
            missing[key] = 'stored conversion fit missing'; continue
        sources[key] = dict(name=key, kind='baseline', mask=None, checkpoint=str(ck),
                            checkpoint_sha256=st['checkpoint_sha256'], fit_dir=str(fit), dataset=j['dataset'])
    for d in ('bloodmnist', 'pathmnist', 'dermamnist'):
        if not (data / f'{d}.npz').is_file():
            raise SystemExit(f'Medical data missing: {data / (d + ".npz")}')
    return sources, missing, str(run), str(data)


def main(repo, results, data_root=None, medical_run=None, medical_data=None):
    import sys
    repo = Path(repo).expanduser().resolve(); results = Path(results).expanduser().resolve()
    frozen = repo / 'results' / 'campaign_2026-09-20' / 'source_final'
    sys.path.insert(0, str(frozen)); sys.path.insert(0, str(HERE))
    from io_utils import sha, write
    from protocol_v2 import schedule
    inventory = json.loads((frozen / 'checkpoint_inventory.json').read_text())
    sources, missing = {}, []
    for j in inventory:
        if not j['name'].startswith('regularized_ln_seed'):
            continue
        original = Path(j['checkpoint']); cands = [Path.home() / original.relative_to('/home/anonuser')]
        if not cands[0].is_file():
            cands = list(results.rglob(original.name))
        hit = next((p for p in cands if p.is_file() and sha(p) == j['checkpoint_sha256']), None)
        if hit is None:
            missing.append(j['name']); continue
        sources[j['name']] = {k: v for k, v in j.items() if k != 'out'} | dict(checkpoint=str(hit))
    if missing:
        raise SystemExit('Missing or changed checkpoints: ' + ', '.join(missing))
    data = Path(data_root).expanduser() if data_root else repo / 'data'
    if not (data / 'cifar-10-batches-py' / 'test_batch').is_file():
        raise SystemExit(f'CIFAR-10 cache missing at {data}; GPU jobs never download.')
    parent = results / 'closeout_v2'; parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='campaign_', dir=parent)); (root / 'logs').mkdir()
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '*.zip', 'local_validation*')
    shutil.copytree(frozen, root / 'code', ignore=ignore)
    for p in HERE.iterdir():
        if p.is_file() and (p.suffix in ('.py', '.sh', '.sbatch') or p.name == 'PREREGISTRATION_v2.md'):
            if (root / 'code' / p.name).exists():
                raise SystemExit(f'Name clash with frozen code: {p.name}')
            shutil.copy2(p, root / 'code' / p.name)
    v1 = sorted(results.glob('final_assessment/campaign_*/conversion/preserve_s43_budget_gelu_n2048/frozen.json'))
    plan = schedule()
    excluded = {}
    if medical_run:
        msrc, missing, mrun, mdata = medical_sources(medical_run, medical_data, sha)
        sources.update(msrc)
        plan.update(medical_run=mrun, medical_data_root=mdata)
        for g in ('pilot', 'med_e1', 'med_e2'):
            keep = []
            for j in plan[g]:
                key = f"{j['dataset']}_s{j['seed']}_000" if j.get('dataset', 'cifar10') != 'cifar10' else None
                if key and key not in msrc:
                    excluded.setdefault(key, dict(reason=missing.get(key, 'no arm-000 job'), jobs=[]))['jobs'].append(j['name'])
                else:
                    keep.append(j)
            plan[g] = keep
    else:
        plan['med_e1'], plan['med_e2'] = [], []
        plan['pilot'] = [j for j in plan['pilot'] if j['dataset'] == 'cifar10']
        excluded['medical'] = dict(reason='not configured (no --medical-run)')
    plan.update(sources=sources, data_root=str(data), repo=str(repo), excluded=excluded,
                protocol_sha256=sha(HERE / 'protocol_v2.py'), preregistration_sha256=sha(HERE / 'PREREGISTRATION_v2.md'),
                v1_frozen_s43_budget_gelu=str(v1[-1]) if v1 else None,
                historical_test_exposure=True, training=False, bootstrapping=False)
    write(root / 'manifest.json', plan)
    write(root / 'source_hashes.json', {str(p.relative_to(root / 'code')): sha(p)
                                        for p in sorted((root / 'code').rglob('*')) if p.is_file()})
    (results / 'latest_closeout_v2.txt').write_text(str(root) + '\n')
    print(root)
    return root


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=str(Path.home() / 'Private-ViT-FHE'))
    ap.add_argument('--results', default=str(Path.home() / 'pvit_results'))
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--medical-run', default=None); ap.add_argument('--medical-data', default=None)
    a = ap.parse_args()
    if bool(a.medical_run) != bool(a.medical_data):
        raise SystemExit('--medical-run and --medical-data go together')
    main(a.repo, a.results, a.data_root, a.medical_run, a.medical_data)
