"""Frozen-weight, calibrated arithmetic baselines. No learning, clipping or oracle division."""
import copy,math
import numpy as np
import torch
from torch import nn

PROFILES={'budget':dict(gelu_degree=7,norm_degree=7,exp_root_degree=4,reciprocal_steps=2),
          'accurate':dict(gelu_degree=15,norm_degree=15,exp_root_degree=8,reciprocal_steps=4)}

def fit_poly(lo,hi,degree,fn):
 if not np.isfinite([lo,hi]).all() or hi<lo:raise ValueError('Invalid fit domain')
 if hi-lo<1e-8:lo-=1e-5;hi+=1e-5
 center=(hi+lo)/2;radius=(hi-lo)/2
 nodes=np.cos(np.pi*(np.arange(1024)+.5)/1024);x=center+radius*nodes
 cheb=np.polynomial.chebyshev.chebfit(nodes,fn(x),degree)
 co=np.polynomial.chebyshev.cheb2poly(cheb)
 grid=np.linspace(-1,1,4097);truth=fn(center+radius*grid);pred=np.polynomial.polynomial.polyval(grid,co)
 return dict(lo=float(lo),hi=float(hi),center=float(center),radius=float(radius),coefficients=co.tolist(),degree=degree,
    grid_max_abs_error=float(np.max(np.abs(pred-truth))),grid_min_output=float(pred.min()),error_scope='Dense grid diagnostic; not a proven uniform error bound')

def np_poly(x,p):return np.polynomial.polynomial.polyval((x-p['center'])/p['radius'],p['coefficients'])
def torch_poly(x,p):
 z=(x-p['center'])/p['radius'];y=torch.zeros_like(z)+p['coefficients'][-1]
 for c in p['coefficients'][-2::-1]:y=y*z+c
 return y

def reciprocal(x,p):
 scale=2/(p['lo']+p['hi']);e=1-x*scale;y=1+e
 for _ in range(p['steps']):e=e*e;y=y*(1+e)
 return y*scale

class Recorder:
 def __init__(self):self.flags=None;self.score=None;self.first_inputs={};self.enabled=True
 def begin(self,n,device):
  self.flags=torch.zeros(n,dtype=torch.bool,device=device);self.score=torch.zeros(n,dtype=torch.float64,device=device);self.first_inputs={}
 def check(self,name,x,p):
  if not self.enabled:return
  finite=torch.isfinite(x);bad=(~finite)|(x<p['lo'])|(x>p['hi']);self.flags|=bad.flatten(1).any(1)
  width=max(p['hi']-p['lo'],1e-12)
  excess=torch.maximum((p['lo']-x)/width,(x-p['hi'])/width).clamp(min=0)
  excess=torch.nan_to_num(excess,nan=1e30,posinf=1e30,neginf=1e30)
  self.score=torch.maximum(self.score,excess.flatten(1).amax(1).double())
  if name.endswith('attn_act') or name.endswith('act') or name.endswith('norm1'):
   self.first_inputs[name]=x[:4].detach()

class PolynomialGELU(nn.Module):
 def __init__(self,p,name,rec):super().__init__();self.p=p;self.name=name;self.rec=rec
 def forward(self,x):
  self.rec.check(self.name,x,self.p);return torch_poly(x,self.p)

class PolynomialLN(nn.Module):
 def __init__(self,old,p,name,rec):
  super().__init__();self.weight=nn.Parameter(old.weight.detach().clone(),requires_grad=False);self.bias=nn.Parameter(old.bias.detach().clone(),requires_grad=False)
  self.eps=old.eps;self.p=p;self.name=name;self.rec=rec
 def forward(self,x):
  h=x-x.mean(-1,keepdim=True);v=h.square().mean(-1,keepdim=True)+self.eps
  self.rec.check(self.name,v,self.p);return h*torch_poly(v,self.p)*self.weight+self.bias

class PolynomialSoftmax(nn.Module):
 def __init__(self,p,name,rec):super().__init__();self.p=p;self.name=name;self.rec=rec
 def forward(self,s):
  z=s-s.mean(-1,keepdim=True)
  if self.rec.enabled:self.rec.first_inputs[self.name+'.raw']=s[:4].detach()
  self.rec.check(self.name,z,self.p['root'])
  u=torch_poly(z,self.p['root']).square()+self.p['floor'];d=u.sum(-1,keepdim=True)
  self.rec.check(self.name+'.denominator',d,self.p['reciprocal'])
  return u*reciprocal(d,self.p['reciprocal'])

def convert(base,params,mask):
 model=copy.deepcopy(base).eval();rec=Recorder()
 for name,old in list(model.named_modules()):
  parent_name,_,leaf=name.rpartition('.');parent=model.get_submodule(parent_name) if parent_name else model
  if name not in params:continue
  p=params[name]
  if p['kind']=='gelu' and mask in ('gelu','all'):setattr(parent,leaf,PolynomialGELU(p['poly'],name,rec))
  if p['kind']=='norm' and mask in ('norm','all'):setattr(parent,leaf,PolynomialLN(old,p['poly'],name,rec))
  if p['kind']=='attention' and mask in ('attention','all'):setattr(parent,leaf,PolynomialSoftmax(p,name,rec))
 for p in model.parameters():p.requires_grad_(False)
 return model,rec

@torch.no_grad()
def calibrate(base,batches,device,profile):
 bounds={};handles=[];saved=[]
 def collect(name,kind,x):
  if kind=='norm':h=x-x.mean(-1,keepdim=True);x=h.square().mean(-1,keepdim=True)+base.get_submodule(name).eps
  if kind=='attention':x=x-x.mean(-1,keepdim=True)
  f=x[torch.isfinite(x)]
  if f.numel()!=x.numel():raise ValueError('Nonfinite reference during calibration: '+name)
  r=bounds.setdefault(name,dict(kind=kind,lo=math.inf,hi=-math.inf))
  r['lo']=min(r['lo'],float(f.min()));r['hi']=max(r['hi'],float(f.max()))
 for name,mod in base.named_modules():
  kind='norm' if isinstance(mod,nn.LayerNorm) else 'gelu' if isinstance(mod,nn.GELU) else 'attention' if name.endswith('.attn_act') else None
  if kind:handles.append(mod.register_forward_pre_hook(lambda mod,args,name=name,kind=kind:collect(name,kind,args[0])))
 try:
  for x,_,ids in batches:base(x.to(device));saved.extend(ids.tolist())
 finally:
  for h in handles:h.remove()
 pset=PROFILES[profile];params={}
 for name,r in bounds.items():
  span=max(r['hi']-r['lo'],1e-6);lo=r['lo']-.05*span;hi=r['hi']+.05*span
  if r['kind']=='norm':
   lo=max(r['lo']*.95,1e-8);hi=r['hi']*1.05
   params[name]=dict(kind='norm',poly=fit_poly(lo,hi,pset['norm_degree'],lambda x:1/np.sqrt(x)))
  elif r['kind']=='gelu':
   params[name]=dict(kind='gelu',poly=fit_poly(lo,hi,pset['gelu_degree'],lambda x:.5*x*(1+np.array([math.erf(float(v)/np.sqrt(2)) for v in x]))))
  else:
   params[name]=dict(kind='attention',root=fit_poly(lo,hi,pset['exp_root_degree'],lambda x:np.exp((x-hi)/2)),
     public_exp_shift=hi,floor=1e-12,reciprocal=dict(lo=math.inf,hi=-math.inf,steps=pset['reciprocal_steps']))
 # Second source pass fits denominator domains to the chosen numerator, never to held-out outcomes.
 def denom_hook(mod,args,name):
  p=params[name];s=args[0].double();z=s-s.mean(-1,keepdim=True);u=torch_poly(z,p['root']).square()+p['floor'];d=u.sum(-1)
  q=p['reciprocal'];q['lo']=min(q['lo'],float(d.min()));q['hi']=max(q['hi'],float(d.max()))
 for name,p in params.items():
  if p['kind']=='attention':handles.append(base.get_submodule(name).register_forward_pre_hook(lambda mod,args,name=name:denom_hook(mod,args,name)))
 try:
  for x,_,_ in batches:base(x.to(device))
 finally:
  for h in handles:h.remove()
 for p in params.values():
  if p['kind']=='attention':
   q=p['reciprocal'];q['lo']*=.95;q['hi']*=1.05
   q['relative_bound_on_interval']=((q['hi']-q['lo'])/(q['hi']+q['lo']))**(2**(q['steps']+1))
 return params,saved
