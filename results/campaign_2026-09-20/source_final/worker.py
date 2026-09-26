import argparse,copy,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from io_utils import sha,write,env
from pvit_lab.model import load_checkpoint,ViT,legacy_spec
from pvit_lab.common import seed_all,device_for
from pvit_lab.data import loader
from polynomial import calibrate,convert
from assessment import outcomes,summarize
from metrics import classification
from data_eval import subsets
from protocol import STATIC_BAD_RATE


def load(j):
 if sha(j['checkpoint'])!=j['checkpoint_sha256']:raise ValueError('Changed checkpoint '+j['name'])
 if j.get('kind')=='power':
  import native_reference as native
  model=ViT(legacy_spec('001' if j['arm']=='Q' else '000'))
  for b in model.blocks:b.attn_act=native.PowerNormAttn()
  ck=torch.load(j['checkpoint'],map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict'],strict=True)
 else:model,ck=load_checkpoint(j['checkpoint'],j.get('mask'))
 model.eval()
 for p in model.parameters():p.requires_grad_(False)
 return model

@torch.no_grad()
def forward(model,x,rec=None):
 if rec:rec.begin(len(x),x.device)
 y=model(x)
 return y

@torch.no_grad()
def replay(base,p,q,rec,rec64,batches,device,capture=False):
 arrays={k:[] for k in ('reference','converted','fp64','labels','ids','outside','range_score')};fixture={};done=False;times=[]
 for x,y,ids in batches:
  x=x.to(device);source={};hooks=[]
  if capture and not done:
   for i in (0,len(base.blocks)-1):
    b=base.blocks[i]
    hooks.append(b.attn_act.register_forward_pre_hook(lambda m,args,i=i:source.update({f'b{i}_scores':args[0][:4].detach().cpu().numpy()})))
    hooks.append(b.act.register_forward_pre_hook(lambda m,args,i=i:source.update({f'b{i}_gelu':args[0][:4].detach().cpu().numpy()})))
    def normhook(m,args,i=i):
     h=args[0]-args[0].mean(-1,keepdim=True);source[f'b{i}_variance']=(h.square().mean(-1,keepdim=True)+m.eps)[:4].cpu().numpy()
    hooks.append(b.norm1.register_forward_pre_hook(normhook))
  r=base(x)
  for h in hooks:h.remove()
  z=forward(p,x,rec);flags=rec.flags.detach().cpu().numpy();score=rec.score.detach().cpu().numpy()
  if capture and not done:
   fixture.update(source);fixture['ids']=ids[:4].numpy()
   for i in (0,len(base.blocks)-1):
    for suffix,key in [('attn_act.raw','scores'),('act','gelu'),('norm1','variance')]:
     v=rec.first_inputs.get(f'blocks.{i}.{suffix}')
     if v is not None:fixture[f'b{i}_converted_{key}']=v.cpu().numpy()
   done=True
  zz=forward(q,x.double(),rec64)
  for k,v in [('reference',r.cpu().numpy()),('converted',z.cpu().numpy()),('fp64',zz.cpu().numpy()),('labels',y.numpy()),('ids',ids.numpy()),('outside',flags),('range_score',score)]:arrays[k].append(v)
 return {k:np.concatenate(v) for k,v in arrays.items()},fixture

@torch.no_grad()
def benchmark(base,p,q,rec,rec64,x,device):
 def sync():
  if device.type=='cuda':torch.cuda.synchronize(device)
 result={}
 for name,model,r,dtype,enabled in [('reference_fp32',base,None,torch.float32,False),('converted_fp32',p,rec,torch.float32,False),('converted_with_ranges',p,rec,torch.float32,True),('converted_fp64',q,rec64,torch.float64,False)]:
  if r:r.enabled=enabled
  xx=x.to(device,dtype=dtype)
  def call():
   z=forward(model,xx,r)
   if enabled:r.flags.cpu();r.score.cpu()
   return z
  for _ in range(3):call()
  sync();times=[]
  for _ in range(10):
   sync();t=time.perf_counter();call();sync();times.append(time.perf_counter()-t)
  result[name]=dict(batch=len(xx),warmup=3,repeats=10,seconds=times,p50_seconds=float(np.median(times)),p95_seconds=float(np.quantile(times,.95)))
 rec.enabled=True;rec64.enabled=True
 result['scope']='Synchronized model calls on one fixed calibration-gate batch. Excludes data loading. Ranges includes flag transfer; FP64 is an additional complete forward, not an online encrypted check.'
 return result


def conversion_run(root,index,synthetic=False):
 root=Path(root);m=json.loads((root/'manifest.json').read_text());j=m['conversion'][index];out=root/'conversion'/j['name'];out.mkdir(parents=True,exist_ok=False)
 try:
  seed_all(j['seed']);device=device_for('cpu' if synthetic else 'cuda:0');base=load(j['source']).to(device)
  fit,gate,_=subsets(m['data_root'],j['fit_n'],synthetic,evaluation=False)
  workers=0 if synthetic else 2
  t=time.perf_counter();params,calids=calibrate(base,loader(fit,workers=workers),device,j['profile']);cal_seconds=time.perf_counter()-t
  p,rec=convert(base,params,j['mask']);q,rec64=convert(base.double(),params,j['mask']);base.float();p.float()
  # All learned tensors in the FP32 converted network must equal their source values.
  original=dict(base.named_parameters());converted=dict(p.named_parameters())
  if original.keys()!=converted.keys() or any(not torch.equal(v,converted[k]) for k,v in original.items()):raise ValueError('Learned weights changed during conversion')
  cal,unused=replay(base,p,q,rec,rec64,loader(gate,workers=workers),device)
  target=outcomes(cal['reference'],cal['converted'],cal['labels']);eligible=target['reference_valid']
  bad_rate=float(target['conversion_failure'][eligible].mean()) if eligible.any() else 1.
  static=bad_rate>STATIC_BAD_RATE
  # Freeze before constructing/evaluating held-out CIFAR test data.
  frozen=dict(profile=j['profile'],mask=j['mask'],parameters=params,fit_image_ids=calids,gate_image_ids=cal['ids'].tolist(),
    calibration_failure_rate=bad_rate,static_review=static,source_sha256=j['source']['checkpoint_sha256'],
    calibration_only_rule='Review configuration if >1% of reference-finite calibration-gate inputs violate fixed conversion-failure criterion.',
    parameter_fitting_seconds=cal_seconds,weights_identical=True)
  write(out/'frozen.json',frozen);freeze_hash=sha(out/'frozen.json')
  calreport,_,_,_,_=summarize(cal['reference'],cal['converted'],cal['fp64'],cal['labels'],cal['outside'],static)
  write(out/'calibration_gate.json',calreport)
  _,_,sets=subsets(m['data_root'],j['fit_n'],synthetic,evaluation=True);reports={}
  for split,ds in sets.items():
   t=time.perf_counter();a,fixture=replay(base,p,q,rec,rec64,loader(ds,workers=workers),device,capture=(split=='clean' and j['mask']=='all'))
   r,tgt,det,delta,margin=summarize(a['reference'],a['converted'],a['fp64'],a['labels'],a['outside'],static)
   r['instrumented_wall_seconds']=time.perf_counter()-t;reports[split]=r
   np.savez_compressed(out/f'{split}_predictions.npz',**a,**{f'target_{k}':v for k,v in tgt.items()},**{f'warning_{k}':v for k,v in det.items()},precision_delta=delta,converted_margin=margin)
   if fixture:
    np.savez_compressed(out/'fhe_inputs.npz',**fixture)
    write(out/'fhe_inputs_meta.json',dict(sha256=sha(out/'fhe_inputs.npz'),frozen_sha256=freeze_hash,ids=fixture['ids'].tolist(),selection='First four clean test examples fixed before outcomes; first and last block, source and actual converted path. No selecting easy/finite rows.'))
  x=next(iter(loader(gate,workers=0)))[0];cost=benchmark(base,p,q,rec,rec64,x,device)
  e=env();e.update(parameter_count=sum(v.numel() for v in base.parameters()),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,memory_scope='Whole assessment process, including reference and both converted models; not isolated inference memory',torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU')
  write(out/'report.json',dict(status='measured',job=j,frozen_sha256=freeze_hash,splits=reports,cost=cost,environment=e,
    scope='Same frozen weights; polynomial approximation baseline, not exact functional identity or faithful ATLAS reproduction.',
    historical_exposure='CIFAR test was used in earlier project runs. These rules are newly frozen before current outcomes; do not claim a historically pristine test set. Perturbations are fixed synthetic stress tests, not independent datasets.'))
 except Exception as e:write(out/'error.json',dict(error=str(e),traceback=traceback.format_exc()));raise

@torch.no_grad()
def endpoint_run(root,index,synthetic=False):
 root=Path(root);m=json.loads((root/'manifest.json').read_text());j=m['endpoints'][index];out=root/'endpoints'/j['name'];out.mkdir(parents=True,exist_ok=False)
 try:
  device=device_for('cpu' if synthetic else 'cuda:0');model=load(j['source']).to(device)
  _,_,sets=subsets(m['data_root'],synthetic=synthetic);result={}
  for split,ds in sets.items():
   zs=[];ys=[];ids=[]
   for x,y,i in loader(ds,workers=0 if synthetic else 2):zs.append(model(x.to(device)).cpu().numpy());ys.append(y.numpy());ids.append(i.numpy())
   z=np.concatenate(zs);y=np.concatenate(ys);result[split]=classification(z,y)
   np.savez_compressed(out/f'{split}_predictions.npz',logits=z,labels=y,ids=np.concatenate(ids))
  write(out/'report.json',dict(status='measured',job=j,splits=result,environment=env(),scope='Trained original checkpoint utility/failure context. P/Q are not same-weight conversions of reference; no predictive screen inference from these comparisons.'))
 except Exception as e:write(out/'error.json',dict(error=str(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('root');a.add_argument('kind',choices=['conversion','endpoint']);a.add_argument('index',type=int);v=a.parse_args()
 os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
 (conversion_run if v.kind=='conversion' else endpoint_run)(v.root,v.index)
