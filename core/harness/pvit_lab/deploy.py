"""Frozen plaintext deployment circuit: polynomial GELU/inverses and folded BN.

This is P-mode, NOT CKKS. It explicitly refuses softmax, ReLU, clamps and
max-scaled attention. No intermediate clipping, decryption or adaptive ranges.
Calibration is fitted first on training data; validation is then a held-out audit.
"""
import argparse,copy,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .model import load_checkpoint,RMSNorm,BatchLN
from .data import datasets,loader
from .measure import Census,evaluate
from .audit import fold_bn,paired
from .common import *


class ChebyGELU(nn.Module):
    def __init__(self,lo,hi,degree):
        super().__init__();self.lo,self.hi=lo,hi
        n=max(512,16*(degree+1));z=np.cos(np.pi*(np.arange(n)+.5)/n);x=(lo+hi)/2+(hi-lo)/2*z
        y=torch.nn.functional.gelu(torch.tensor(x,dtype=torch.float64)).numpy()
        self.register_buffer('coefficients',torch.tensor(np.polynomial.chebyshev.chebfit(z,y,degree),dtype=torch.float32))
    def forward(self,x):
        z=(2*x-self.lo-self.hi)/(self.hi-self.lo);b1=torch.zeros_like(x);b2=torch.zeros_like(x)
        for c in self.coefficients[1:].flip(0):b0=2*z*b1-b2+c;b2,b1=b1,b0
        return z*b1-b2+self.coefficients[0]


class Reciprocal(nn.Module):
    def __init__(self,a,b,iterations):
        super().__init__();assert 0<a<=b;self.a,self.b,self.iterations=a,b,iterations
    def forward(self,d):
        y=torch.ones_like(d)*(2/(self.a+self.b))
        for _ in range(self.iterations):y=y*(2-d*y)
        return y


class ConstantInverse(nn.Module):
    def __init__(self,d):
        super().__init__();self.register_buffer('value',d.detach().reciprocal())
    def forward(self,d):return self.value


class FrozenBatchLN(nn.Module):
    def __init__(self,source):
        super().__init__()
        self.register_buffer('scale',source.weight.detach()/(source.running_denominator.detach()+1e-10))
        self.register_buffer('bias',source.bias.detach().clone())
    def forward(self,x):return (x-x.mean(-1,keepdim=True))*self.scale+self.bias


class PolyNorm(nn.Module):
    def __init__(self,source,a,b,iterations,observer=None):
        super().__init__();self.weight=nn.Parameter(source.weight.detach().clone(),requires_grad=False)
        self.bias=nn.Parameter(source.bias.detach().clone(),requires_grad=False) if getattr(source,'bias',None) is not None else None
        self.center=isinstance(source,nn.LayerNorm);self.eps=source.eps
        self.a,self.b,self.iterations=a,b,iterations;self.observer=observer
    def forward(self,x):
        z=x-x.mean(-1,keepdim=True) if self.center else x
        d=z.square().mean(-1,keepdim=True)+self.eps
        if self.observer:self.observer(d)
        y=torch.ones_like(d)/math.sqrt(self.b)
        for _ in range(self.iterations):y=y*(1.5-.5*d*y*y)
        out=z*y*self.weight
        return out if self.bias is None else out+self.bias


def build(model,calibration,degree=6,iterations=12):
    allowed={'raw','power_raw','square_row','square_fixed','bpmax'}
    if model.spec.attention not in allowed:raise ValueError('P-mode supports '+str(sorted(allowed))+'; max/ReLU/clamp/softmax needs a separately validated circuit')
    p,folded=fold_bn(model);report=dict(bn_folded=folded,gelu={},reciprocal={},norm={},
        execution='plaintext FP32 polynomial circuit; no CKKS timings or depth claims')
    rows=calibration['census'];domains=calibration['domains']
    for i,b in enumerate(p.blocks):
        prefix=f'blocks.{i}'
        if isinstance(b.act,nn.GELU):
            lo,hi=domains[prefix+'.gelu_in'];b.act=ChebyGELU(lo,hi,degree)
            report['gelu'][prefix]=dict(interval=[lo,hi],degree=degree,coefficients=b.act.coefficients.tolist())
        site=prefix+'.attn.denominator'
        if p.spec.attention in ('power_raw','square_row'):
            # Structural lower bound (not a sample quantile); capture upper bound.
            n=(p.spec.img_size//p.spec.patch_size)**2+1
            a=p.spec.epsilon/n if p.spec.attention=='power_raw' else p.spec.floor
            upper=rows[site]['maximum'];upper=max(a,upper*1.05)
            b.attn_act.inverse=Reciprocal(a,upper,iterations)
            q=(upper-a)/(upper+a)
            report['reciprocal'][site]=dict(interval=[a,upper],iterations=iterations,
                exact_arithmetic_relative_error_bound=q**(2**iterations),
                note='bound assumes every deployment input stays inside interval; excludes FP32/CKKS error')
        elif p.spec.attention in ('square_fixed','bpmax'):
            d=b.attn_act.running_denominator+(1e-10 if p.spec.attention=='bpmax' else 0.)
            b.attn_act.inverse=ConstantInverse(d)
        for name in ('norm1','norm2'):
            original=getattr(b,name)
            if isinstance(original,BatchLN):setattr(b,name,FrozenBatchLN(original))
            if isinstance(original,(nn.LayerNorm,RMSNorm)):
                key=prefix+'.'+name+'_in.variance'
                upper=rows[key]['maximum']*1.05
                if isinstance(original,RMSNorm):
                    # Variance census omits mean^2; must use a safe empirical
                    # upper bound from the observed input extrema for RMSNorm.
                    r=rows[prefix+'.'+name+'_in'];upper=max(abs(r['minimum']),abs(r['maximum']))**2*1.05+original.eps
                upper=max(original.eps,upper)
                setattr(b,name,PolyNorm(original,original.eps,upper,iterations))
                report['norm'][prefix+'.'+name]=dict(interval=[original.eps,upper],iterations=iterations)
    if isinstance(p.norm,BatchLN):p.norm=FrozenBatchLN(p.norm)
    if isinstance(p.norm,(nn.LayerNorm,RMSNorm)):
        r=rows['model.final_norm_in'];upper=max(abs(r['minimum']),abs(r['maximum']))**2*1.05+p.norm.eps
        p.norm=PolyNorm(p.norm,p.norm.eps,upper,iterations)
        report['norm']['norm']=dict(interval=[p.norm.a,p.norm.b],iterations=iterations)
    return p,report


def run(checkpoint,out,data_root,mask=None,dataset='cifar10',degree=6,iterations=12,workers=2,device_name='auto',synthetic=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);device=device_for(device_name)
    model,_=load_checkpoint(checkpoint,mask);model.to(device).eval()
    _,val,clean=datasets(data_root,dataset,synthetic=synthetic)
    ca=loader(clean,workers=workers,limit=2048);va=loader(val,workers=workers)
    c=Census();c.attach(model);evaluate(model,ca,device,c);c.detach(model)
    calibration=dict(census=c.export(),domains=c.fit_domains())
    p,circuit=build(model,calibration,degree,iterations);p.to(device).eval()
    # Use circuit-specific input bounds; all sites also retain generic S ranges.
    domains=dict(calibration['domains'])
    for name,row in circuit['reciprocal'].items():domains[name]=row['interval']
    for name,row in circuit['norm'].items():domains[name+'.p_denominator']=row['interval']
    v=Census(domains);v.attach(p)
    for name,module in p.named_modules():
        if isinstance(module,PolyNorm):module.observer=lambda d,name=name:v(name+'.p_denominator',d)
    sstats,sref=evaluate(model,va,device)
    timer=time.perf_counter();pstats,pref=evaluate(p,va,device,v);seconds=time.perf_counter()-timer;v.detach(p)
    compare=paired(sref,pref);census=v.export()
    result=dict(reference=sstats,polynomial=pstats,comparison=compare,circuit=circuit,
        p_mode_instrumented_seconds=seconds,memory=memory(device),test_evaluated=False,
        checkpoint_sha256=digest(checkpoint),environment=environment(device))
    write_json(out/'circuit.json',circuit);write_json(out/'calibration.json',calibration)
    write_json(out/'p_census.json',census);write_json(out/'p_result.json',result)
    np.savez_compressed(out/'s_predictions.npz',**sref);np.savez_compressed(out/'p_predictions.npz',**pref)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--mask');p.add_argument('--dataset',default='cifar10');p.add_argument('--data-root',default='./data')
    p.add_argument('--degree',type=int,default=6);p.add_argument('--iterations',type=int,default=12)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--device',default='auto');a=p.parse_args()
    print(json.dumps(run(a.checkpoint,a.out,a.data_root,a.mask,a.dataset,a.degree,a.iterations,a.workers,a.device),indent=2))

if __name__=='__main__':main()
