"""Exact streaming counts, uniform finite-value reservoirs, paired prediction data.

Reservoirs estimate quantiles; extrema, violations and image counts are exact
for the inspected data. None of these establishes a global domain guarantee.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F


class Census:
    def __init__(self, domains=None, capacity=2048, seed=701):
        self.domains=domains or {};self.capacity=capacity;self.rng=np.random.default_rng(seed)
        self.rows={};self.ids=[];self.last={}
    def begin(self, ids): self.ids=list(map(int,ids));self.last={}
    def __call__(self,name,x):
        x=x.detach().float()
        r=self.rows.setdefault(name,dict(count=0,finite=0,nonfinite=0,negative=0,zero=0,
            minimum=math.inf,maximum=-math.inf,sum=0.,sum_squares=0.,below=0,above=0,
            observed_images=set(),violating_images=set(),samples=np.array([],dtype=float)))
        finite=torch.isfinite(x);v=x[finite];old=r['finite'];n=v.numel()
        r['count']+=x.numel();r['finite']+=n;r['nonfinite']+=int((~finite).sum())
        r['negative']+=int((x<0).sum());r['zero']+=int((x==0).sum())
        bad=~finite
        if name in self.domains:
            a,b=self.domains[name];below=x<a;above=x>b
            r['below']+=int(below.sum());r['above']+=int(above.sum());bad=bad|below|above
        if len(self.ids)==x.shape[0]:
            r['observed_images'].update(self.ids)
            r['violating_images'].update(i for i,flag in zip(self.ids,bad.reshape(x.shape[0],-1).any(1).cpu().tolist()) if flag)
        if n:
            r['minimum']=min(r['minimum'],float(v.min()));r['maximum']=max(r['maximum'],float(v.max()))
            r['sum']+=float(v.double().sum());r['sum_squares']+=float(v.double().square().sum())
            # Exact reservoir merge: hypergeometric allocation between old stream
            # and incoming batch, then uniform sampling within each component.
            k=min(self.capacity,old+n)
            take=int(self.rng.hypergeometric(n,old,k)) if old else k
            keep=k-take
            prev=r['samples'][self.rng.choice(len(r['samples']),keep,replace=False)] if keep else np.array([])
            idx=self.rng.choice(n,take,replace=False)
            new=v[torch.as_tensor(idx,device=v.device)].cpu().numpy()
            r['samples']=np.concatenate([prev,new])
    def tap(self,prefix,name,x):
        self(prefix+'.'+name,x)
        if name=='attention':
            self(prefix+'.abs_row_mass',x.abs().sum(-1))
            self(prefix+'.signed_row_mass',x.sum(-1))
            self(prefix+'.zero_row',x.abs().sum(-1).eq(0).float())
            for h in range(x.shape[1]): self(prefix+f'.head{h}.abs_row_mass',x[:,h].abs().sum(-1))
        if name.endswith('_in') and 'norm' in name:
            self(prefix+'.'+name+'.variance',x.var(-1,unbiased=False)+1e-5)
        if name in ('residual_in','attention_branch','mlp_branch'):
            norm=x.square().mean(-1).sqrt();self(prefix+'.'+name+'.rms',norm)
            if name=='residual_in':self.last[prefix]=norm
            elif prefix in self.last:
                self(prefix+'.'+name+'.relative_to_block_input',norm/(self.last[prefix]+1e-12))
    def attach(self,model):
        for i,b in enumerate(model.blocks):
            p=f'blocks.{i}'
            b.observer=lambda n,x,p=p:self.tap(p,n,x)
            b.attn_act.observer=lambda n,x,p=p:self(p+'.attn.'+n,x)
        model.observer=lambda n,x:self.tap('model',n,x)
    def detach(self,model):
        for b in model.blocks:b.observer=None;b.attn_act.observer=None
        model.observer=None
    def export(self):
        out={}
        for name,r in self.rows.items():
            v=dict(r);samples=v.pop('samples');n=v['finite']
            v['mean']=v.pop('sum')/n if n else None
            v['rms']=math.sqrt(v.pop('sum_squares')/n) if n else None
            v['observed_images']=len(v['observed_images']);v['violating_images']=sorted(v['violating_images'])
            v['samples']=samples.tolist();v['quantiles']=dict(zip(['q001','q01','q50','q99','q999'],np.quantile(samples,[.001,.01,.5,.99,.999]).tolist())) if len(samples) else {}
            for key in ('minimum','maximum'):
                if not math.isfinite(v[key]):v[key]=None
            out[name]=v
        return out
    def fit_domains(self,margin=.05):
        out={}
        for name,r in self.rows.items():
            a,b=r['minimum'],r['maximum']
            if not(math.isfinite(a) and math.isfinite(b)):continue
            pad=max(b-a,abs(a),abs(b),1e-8)*margin
            out[name]=[a-pad,b+pad]
        return out


@torch.no_grad()
def evaluate(model, batches, device, census=None):
    model.eval();logits=[];labels=[];ids=[]
    for x,y,j in batches:
        if census:census.begin(j.tolist())
        logits.append(model(x.to(device)).cpu());labels.append(y);ids.append(j)
    z=torch.cat(logits);y=torch.cat(labels);j=torch.cat(ids)
    finite=torch.isfinite(z).all(1);p=z[finite].softmax(-1);yf=y[finite]
    correct=(z[finite].argmax(1)==yf)
    conf,pred=p.max(1);ece=0.
    for lo in torch.linspace(0,1,16)[:-1]:
        hit=(conf>=lo)&(conf<lo+1/15 if lo<14/15 else conf<=1)
        if hit.any():ece+=float(hit.float().mean()*(conf[hit].mean()-(pred[hit]==yf[hit]).float().mean()).abs())
    summary=dict(n=len(y),nonfinite_images=int((~finite).sum()),
        accuracy=100*int(correct.sum())/len(y),
        nll=float(F.cross_entropy(z[finite],yf)) if finite.any() else None,
        ece=ece if finite.any() else None,
        brier=float((p-F.one_hot(yf,z.shape[1])).square().sum(1).mean()) if finite.any() else None)
    return summary,dict(logits=z.numpy(),labels=y.numpy(),ids=j.numpy())


def kd_loss(student,teacher,temperature=4.,t_squared=True):
    loss=F.kl_div(F.log_softmax(student/temperature,-1),F.softmax(teacher/temperature,-1),reduction='batchmean')
    return loss*(temperature**2 if t_squared else 1)


def feature_loss(student,teacher):
    """Explicit ViT adaptation: mean of embedding/hidden/score/norm MSE terms.
    Not the published BERT feature loss: pre-LN geometry and reduction differ.
    """
    terms=[F.mse_loss(student['embedding'],teacher['embedding'].detach())]
    for k in student:
        if k=='embedding':continue
        for name in ('scores','norm1','norm2','hidden'):
            terms.append(F.mse_loss(student[k][name],teacher[k][name].detach()))
    return torch.stack(terms).mean()
