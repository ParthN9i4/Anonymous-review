"""Read trained checkpoints, reproduce validation, census, recalibrate/fold BN."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .model import load_checkpoint
from .data import datasets,loader
from .measure import Census,evaluate
from .common import *


@torch.no_grad()
def recalibrate_bn(model,batches,device):
    model.eval();modules=[]
    for m in model.modules():
        if isinstance(m,nn.BatchNorm1d):
            modules.append((m,m.momentum));m.reset_running_stats();m.momentum=None;m.train()
    if not modules:raise ValueError('No BatchNorm modules')
    for x,_,_ in batches:model(x.to(device))
    for m,momentum in modules:m.momentum=momentum;m.eval()
    return model


@torch.no_grad()
def fold_pair(bn,linear):
    scale=bn.weight/(bn.running_var+bn.eps).sqrt()
    shift=bn.bias-bn.running_mean*scale
    old=linear.weight.clone()
    linear.weight.mul_(scale[None,:]);linear.bias.add_(old@shift)


def fold_bn(model):
    """Exact eval-mode affine composition for this pre-norm graph only."""
    model=copy.deepcopy(model).eval();count=0
    for b in model.blocks:
        for name,dest in [('norm1','qkv'),('norm2','fc1')]:
            m=getattr(b,name)
            if isinstance(m,nn.BatchNorm1d):fold_pair(m,getattr(b,dest));setattr(b,name,nn.Identity());count+=1
    if isinstance(model.norm,nn.BatchNorm1d):fold_pair(model.norm,model.head);model.norm=nn.Identity();count+=1
    return model,count


def paired(reference,other):
    assert np.array_equal(reference['ids'],other['ids']) and np.array_equal(reference['labels'],other['labels'])
    a,b=reference['logits'],other['logits'];finite=np.isfinite(a).all(1)&np.isfinite(b).all(1)
    d=np.abs(a[finite]-b[finite]);disagree=(a.argmax(1)!=b.argmax(1))|~finite
    rc=a.argmax(1)==reference['labels'];oc=(b.argmax(1)==other['labels'])&finite
    return dict(n=len(a),nonfinite_pairs=int((~finite).sum()),prediction_disagreements=int(disagree.sum()),
        correct_to_wrong=int((rc&~oc).sum()),wrong_to_correct=int((~rc&oc).sum()),
        max_logit_error=float(d.max()) if d.size else None,mean_logit_error=float(d.mean()) if d.size else None,
        disagreement_ids=reference['ids'][disagree].tolist())


def run(checkpoint,out,data_root,mask=None,dataset='cifar10',device_name='auto',calibration=2048,workers=2,bn_recalibrate=False,synthetic=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    device=device_for(device_name);model,ck=load_checkpoint(checkpoint,mask);model.to(device).eval()
    _,val,clean=datasets(data_root,dataset,synthetic=synthetic)
    ca=loader(clean,workers=workers,limit=calibration);va=loader(val,workers=workers)
    timer=time.perf_counter();c=Census();c.attach(model);calstats,_=evaluate(model,ca,device,c);c.detach(model)
    domains=c.fit_domains();v=Census(domains);v.attach(model);stats,ref=evaluate(model,va,device,v);v.detach(model)
    np.savez_compressed(out/'validation_predictions.npz',**ref)
    write_json(out/'calibration.json',dict(checkpoint_sha256=digest(checkpoint),split='first training split indices, clean transform',census=c.export(),domains=domains))
    write_json(out/'validation_census.json',v.export())
    recorded=ck.get('result',{}).get('best_val',ck.get('validation',{}).get('accuracy'))
    result=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),validation=stats,
        recorded_validation=recorded,validation_replay_difference_pp=stats['accuracy']-recorded if recorded is not None else None,
        calibration=calstats,test_evaluated=False,environment=environment(device))
    folded,count=fold_bn(model)
    if count:
        fs,fp=evaluate(folded,va,device);result['bn_fold']=dict(layers=count,validation=fs,comparison=paired(ref,fp))
    if bn_recalibrate:
        recal=recalibrate_bn(copy.deepcopy(model),ca,device);rs,rp=evaluate(recal,va,device)
        result['bn_recalibration']=dict(validation=rs,comparison=paired(ref,rp),
            method='reset buffers; cumulative per-batch statistics on clean training subset; all BN layers updated together')
        np.savez_compressed(out/'recalibrated_predictions.npz',**rp)
    result.update(elapsed_seconds=time.perf_counter()-timer,memory=memory(device))
    write_json(out/'audit.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--data-root',default='./data');p.add_argument('--mask');p.add_argument('--dataset',default='cifar10')
    p.add_argument('--device',default='auto');p.add_argument('--calibration',type=int,default=2048);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--bn-recalibrate',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.checkpoint,a.out,a.data_root,a.mask,a.dataset,a.device,a.calibration,a.workers,a.bn_recalibrate),indent=2))

if __name__=='__main__':main()
