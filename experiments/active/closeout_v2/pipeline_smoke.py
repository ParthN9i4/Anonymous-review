"""End-to-end synthetic rehearsal of the whole closeout_v2 analysis path (CPU, minutes).

Runs a reduced schedule on synthetic data and a random-init model:
workers (e1, e2, e3) -> screen freeze -> confirm -> screen evaluate -> analyze -> figures.
It checks plumbing only; no number it produces means anything.

    python pipeline_smoke.py <scratch_dir> [--figures-script ../../../paper/scripts/make_v2_figures.py]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from io_utils import write
from protocol_v2 import schedule
import worker_v2
import screen_v2
import analyze_v2


def main(scratch, figures_script=None):
    root = Path(scratch) / 'run'; out = Path(scratch) / 'tables'
    p = schedule()
    pick = dict(e1=[j for j in p['e1'] if j['seed'] == 42 and j['mask'] == 'attention'],
                e2=[j for j in p['e2'] if (j.get('gelu_degree') in (7, 15)) or (j.get('reciprocal_steps') in (2, 6) and j.get('exp_root_degree') == 4)],
                e3=[j for j in p['e3'] if j['seed'] == 42 and j['mask'] == 'gelu' and 'only_block' in j],
                confirm=[j for j in p['confirm'] if (j.get('gelu_degree') in (7, 15)) or (j.get('reciprocal_steps') in (2, 6) and j.get('exp_root_degree') == 4)])
    write(root / 'manifest.json', dict(data_root='.', sources={}, **pick))
    for g in ('e1', 'e2', 'e3'):
        for i in range(len(pick[g])):
            worker_v2.run(root, g, i, synthetic=True)
    screen_v2.freeze(root)
    for i in range(len(pick['confirm'])):
        worker_v2.run(root, 'confirm', i, synthetic=True)
    screen_v2.evaluate(root)
    analyze_v2.main(root, out)
    if figures_script:
        subprocess.run([sys.executable, figures_script, '--tables', str(out), '--out', str(Path(scratch) / 'figures')],
                       check=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('scratch'); ap.add_argument('--figures-script')
    a = ap.parse_args(); print(main(a.scratch, a.figures_script))
