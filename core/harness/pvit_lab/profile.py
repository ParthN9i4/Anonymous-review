"""Plaintext latency/memory and mathematical dense-MAC accounting; not HE cost."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .model import load_checkpoint
from .data import datasets,loader
from .common import *


@torch.no_grad()
def run(checkpoint,out,data_root,mask=None,dataset='cifar10',batch=1,warmup=10,repeats=50,device_name='auto'):
    device=device_for(device_name);m,_=load_checkpoint(checkpoint,mask);m.to(device).eval()
    _,val,_=datasets(data_root,dataset);x=next(iter(loader(val,batch,workers=0)))[0].to(device)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    for _ in range(warmup):m(x)
    sync()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    times=[]
    for _ in range(repeats):
        sync();start=time.perf_counter();z=m(x);sync();times.append(time.perf_counter()-start)
    macs={};handles=[]
    def hook(name):
        def count(module,inputs,output):
            if isinstance(module,nn.Linear):v=output.numel()*module.in_features
            else:v=output.numel()*module.in_channels*module.kernel_size[0]*module.kernel_size[1]//module.groups
            macs[name]=int(v)
        return count
    for name,module in m.named_modules():
        if isinstance(module,(nn.Linear,nn.Conv2d)):handles.append(module.register_forward_hook(hook(name)))
    m(x)
    for h in handles:h.remove()
    s=m.spec;n=(s.img_size//s.patch_size)**2+1
    macs['QK_and_AV_all_blocks']=int(2*batch*s.depth*n*n*s.embed_dim)
    result=dict(scope='plaintext FP32 inference, model call only; excludes data transfer/loading/key generation/encryption',
        checkpoint_sha256=digest(checkpoint),batch=batch,warmup=warmup,repeats=repeats,latency_seconds=times,
        p50_seconds=float(np.median(times)),p95_seconds=float(np.quantile(times,.95)),
        throughput_images_per_second=batch/float(np.mean(times)),parameters=sum(p.numel() for p in m.parameters()),
        state_tensor_bytes=sum(t.numel()*t.element_size() for t in m.state_dict().values()),
        checkpoint_file_bytes=Path(checkpoint).stat().st_size,dense_macs=macs,total_dense_macs=sum(macs.values()),
        counting_convention='One multiply-accumulate = one MAC = conventionally two FLOPs; excludes nonlinear/norm/elementwise ops. Not CKKS multiplication counts.',
        finite_outputs=bool(torch.isfinite(z).all()),environment=environment(device),memory=memory(device))
    write_json(out,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--mask');p.add_argument('--dataset',default='cifar10');p.add_argument('--data-root',default='./data')
    p.add_argument('--batch',type=int,default=1);p.add_argument('--warmup',type=int,default=10);p.add_argument('--repeats',type=int,default=50);p.add_argument('--device',default='auto')
    a=p.parse_args();print(json.dumps(run(a.checkpoint,a.out,a.data_root,a.mask,a.dataset,a.batch,a.warmup,a.repeats,a.device),indent=2))

if __name__=='__main__':main()
