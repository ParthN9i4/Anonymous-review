"""Explicit frozen checkpoint evaluation. Never imported/called by training.

Selection JSON: [{"checkpoint":"...","sha256":"...","dataset":"cifar10",
                  "mask":"001","label":"BN seed42"}, ...]
Freeze choices on validation before creating this file; do not tune from output.
"""
import argparse,json
from pathlib import Path
import numpy as np
from torchvision import datasets as D,transforms as T
from .model import load_checkpoint
from .data import Indexed,loader
from .measure import evaluate
from .common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);p.add_argument('--out',required=True)
    p.add_argument('--data-root',default='./data');p.add_argument('--device',default='auto');a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError('Final-test output already exists; preserve the original evaluation')
    rows=json.loads(Path(a.selection).read_text())
    for r in rows:
        if digest(r['checkpoint'])!=r['sha256']:raise ValueError('Frozen checkpoint hash mismatch')
    out.mkdir(parents=True);write_json(out/'frozen_selection.json',rows);device=device_for(a.device)
    results=[]
    for i,r in enumerate(rows):
        name=r['dataset'];cls=D.CIFAR10 if name=='cifar10' else D.CIFAR100
        assert name in ('cifar10','cifar100')
        mean=(.4914,.4822,.4465) if name=='cifar10' else (.5071,.4865,.4409)
        std=(.247,.2435,.2616) if name=='cifar10' else (.2673,.2564,.2762)
        ds=cls(a.data_root,train=False,download=False,transform=T.Compose([T.ToTensor(),T.Normalize(mean,std)]))
        m,_=load_checkpoint(r['checkpoint'],r.get('mask'));m.to(device)
        stats,pred=evaluate(m,loader(Indexed(ds,range(len(ds)))),device)
        results.append(r|dict(test=stats));np.savez_compressed(out/f'{i:03d}_predictions.npz',**pred)
        write_json(out/'results.json',results)
    write_json(out/'environment.json',environment(device))

if __name__=='__main__':main()
