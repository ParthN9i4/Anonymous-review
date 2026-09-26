"""Controlled CE/KD, verified cold/warm initialization, validation-only selection.

Separate best/final/last checkpoints. Resume from the last completed epoch;
an interrupted partial epoch is rerun with restored random/loader states.
"""
import argparse,copy,json,math,os,random,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch.nn import functional as F
from .model import ModelSpec,ViT,load_checkpoint,transfer
from .data import datasets,loader
from .measure import evaluate,kd_loss,feature_loss,Census
from .common import *


DEFAULT=dict(dataset='cifar10',seed=42,epochs=100,batch_size=128,workers=2,lr=.001,
    weight_decay=.05,recipe='legacy',initialization='cold',loss='kd',alpha=.1,
    temperature=4.,t_squared=True,feature_weight=.01,kd_start_fraction=0.,kd_ramp_fraction=0.,
    probe_every=10,probe_images=256,calibration_images=2048,synthetic=False)


def run(config,out,data_root,device_name='auto',resume=False):
    c=DEFAULT|config;out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert c['recipe'] in ('legacy','regularized') and c['loss'] in ('ce','kd','feature_kd')
    assert c['initialization'] in ('cold','warm') and 0<=c['alpha']<=1 and c['temperature']>0
    assert c['kd_start_fraction']>=0 and c['kd_ramp_fraction']>=0
    assert c['kd_start_fraction']+c['kd_ramp_fraction']<1
    spec=ModelSpec(**c.get('model',{}));spec.num_classes=10 if c['dataset']=='cifar10' else 100
    if c['recipe']=='regularized':spec.drop_path=.1
    c['model']=asdict(spec)
    teacher=None;teacher_ck={};teacher_hash=None
    if c['loss']!='ce' or c['initialization']=='warm':
        if not c.get('teacher'):raise ValueError('Teacher checkpoint required')
        teacher,teacher_ck=load_checkpoint(c['teacher'],'000');teacher_hash=digest(c['teacher'])
        for key in ('num_classes','img_size','patch_size','embed_dim','depth','num_heads','mlp_ratio'):
            if getattr(teacher.spec,key)!=getattr(spec,key):raise ValueError('Teacher architecture mismatch: '+key)
    signature=dict(config=c,teacher_sha256=teacher_hash,code_sha256=code_digest())
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text())!=signature:raise ValueError('Output belongs to different protocol/code')
        if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
        if not resume:raise ValueError('Incomplete output; use --resume to restore last completed epoch')
    else:write_json(out/'protocol.json',signature)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    seed_all(c['seed']);device=device_for(device_name)
    # Common ordinary initialization, then copy all compatible tensors into
    # every cold arm; differing constructors cannot shift linear-weight RNG.
    base_spec=copy.deepcopy(spec);base_spec.attention='softmax';base_spec.activation='gelu';base_spec.norm='ln';base_spec.final_norm='same'
    base=ViT(base_spec);model=ViT(spec)
    init_report=transfer(teacher if c['initialization']=='warm' else base,model)
    del base
    model.to(device)
    if teacher is not None:
        teacher.to(device).eval()
        for p in teacher.parameters():p.requires_grad_(False)
    write_json(out/'initialization.json',init_report)
    write_json(out/'environment.json',environment(device))
    train,val,clean=datasets(data_root,c['dataset'],c['recipe']=='regularized',synthetic=c['synthetic'])
    tr=loader(train,c['batch_size'],c['workers'],True,c['seed'])
    va=loader(val,c['batch_size'],c['workers']);probe=loader(clean,c['batch_size'],c['workers'],limit=c['probe_images'])
    initial_stats,_=evaluate(model,va,device)
    write_json(out/'initial_validation.json',initial_stats)
    initial_census=Census(capacity=256);initial_census.attach(model)
    evaluate(model,probe,device,initial_census);initial_census.detach(model)
    reference_domains=initial_census.fit_domains()
    write_json(out/'initial_census.json',dict(census=initial_census.export(),reference_domains=reference_domains,
        note='Initial-model empirical ranges on clean training probe; drift diagnostic, not certified approximation domains'))
    # Shared RNG after constructors makes stochastic-depth masks reproducible
    # without coupling their sequence to which polynomial class was built.
    seed_all(c['seed']+100000)
    if c['recipe']=='legacy':params=model.parameters()
    else:
        decay=[];no_decay=[]
        for name,p in model.named_parameters():
            (no_decay if p.ndim<=1 or name in ('cls_token','pos_embed') else decay).append(p)
        params=[dict(params=decay,weight_decay=c['weight_decay']),dict(params=no_decay,weight_decay=0.)]
    opt=torch.optim.AdamW(params,lr=c['lr'],weight_decay=c['weight_decay'])
    history=[];start=0;best=-1.;elapsed_before=0.
    if resume and (out/'last.pth').exists():
        ck=torch.load(out/'last.pth',map_location='cpu',weights_only=True)
        model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer'])
        history=ck['history'];start=ck['epoch'];best=ck['best'];elapsed_before=ck['elapsed_seconds']
        torch.set_rng_state(ck['torch_rng']);tr.generator.set_state(ck['loader_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(ck['cuda_rng'])
        random.setstate(ck['python_rng'])
        ns=ck['numpy_rng'];np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
        core={k:ck[k] for k in ('state_dict','model_spec','epoch','validation','protocol','checkpoint_rule')}
    timer=time.perf_counter();best_epoch=next((h['epoch'] for h in history if h['validation']['accuracy']==best),None)
    if c['loss']=='feature_kd':model.collect_features=True;teacher.collect_features=True
    try:
        for epoch in range(start,c['epochs']):
            t0=time.perf_counter();model.train();ce_sum=kd_sum=feat_sum=grad_sum=0.;grad_max=0.;count=0;grad_diagnostic=None
            warmup=min(5,max(1,c['epochs']//10)) if c['recipe']=='regularized' else 0
            rate=(epoch+1)/warmup if epoch<warmup else .5*(1+math.cos(math.pi*(epoch-warmup)/max(1,c['epochs']-warmup)))
            for group in opt.param_groups:group['lr']=c['lr']*rate
            progress=epoch/c['epochs'];begin=c['kd_start_fraction'];ramp=c['kd_ramp_fraction']
            alpha=c['alpha']*(0 if progress<begin else min(1.,(progress-begin)/ramp) if ramp else 1.)
            if c['loss']=='ce':alpha=0.
            for x,y,_ in tr:
                x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True)
                z=model(x);ce=F.cross_entropy(z,y,label_smoothing=.1 if c['recipe']=='regularized' else 0.)
                kd=z.new_zeros(());feat=z.new_zeros(())
                if teacher is not None and c['loss']!='ce':
                    with torch.no_grad():tz=teacher(x)
                    kd=kd_loss(z,tz,c['temperature'],c['t_squared'])
                    if c['loss']=='feature_kd':feat=feature_loss(model.features,teacher.features)
                loss=(1-alpha)*ce+alpha*kd+c['feature_weight']*feat
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss epoch {epoch+1}')
                if count==0 and c['loss']!='ce':
                    params=list(model.head.parameters())
                    gc=torch.cat([g.flatten() for g in torch.autograd.grad((1-alpha)*ce,params,retain_graph=True)])
                    gk=torch.cat([g.flatten() for g in torch.autograd.grad(alpha*kd,params,retain_graph=True)])
                    nc,nk=float(gc.norm()),float(gk.norm())
                    grad_diagnostic=dict(scope='classifier head, first training batch only',ce_norm=nc,kd_norm=nk,
                        cosine=float(torch.dot(gc,gk)/(gc.norm()*gk.norm())) if nc>0 and nk>0 else None)
                loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
                n=len(y);count+=n;ce_sum+=float(ce.detach())*n;kd_sum+=float(kd.detach())*n;feat_sum+=float(feat.detach())*n
                grad_sum+=float(gn)*n;grad_max=max(grad_max,float(gn))
            stats,_=evaluate(model,va,device)
            row=dict(epoch=epoch+1,ce=ce_sum/count,kd_scaled=kd_sum/count,feature_mse=feat_sum/count,
                alpha=alpha,lr=c['lr']*rate,gradient_norm_mean=grad_sum/count,gradient_norm_max=grad_max,
                head_gradient_diagnostic=grad_diagnostic,validation=stats,seconds=time.perf_counter()-t0)
            history.append(row);is_best=stats['accuracy']>best
            if is_best:best=stats['accuracy'];best_epoch=epoch+1
            core=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},model_spec=asdict(spec),
                epoch=epoch+1,validation=stats,protocol=signature,checkpoint_rule='best validation accuracy; earliest tie')
            if is_best:save_torch(out/'best.pth',core)
            if (epoch+1)%c['probe_every']==0 or epoch==0 or epoch+1==c['epochs']:
                census=Census(reference_domains,capacity=256);census.attach(model);evaluate(model,probe,device,census);census.detach(model)
                coeff={n:float(p.detach()) for n,p in model.named_parameters() if p.numel()==1}
                write_json(out/'telemetry'/f'epoch_{epoch+1:03d}.json',dict(census=census.export(),scalar_parameters=coeff))
            ns=np.random.get_state()
            save_torch(out/'last.pth',core|dict(optimizer=opt.state_dict(),history=history,best=best,
                torch_rng=torch.get_rng_state(),loader_rng=tr.generator.get_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],python_rng=random.getstate(),
                numpy_rng=[ns[0],ns[1].tolist(),int(ns[2]),int(ns[3]),float(ns[4])],
                elapsed_seconds=elapsed_before+time.perf_counter()-timer))
            write_json(out/'history.json',history)
            print(f'epoch={epoch+1} val={stats["accuracy"]:.2f} best={best:.2f} ce={row["ce"]:.4f} kd={row["kd_scaled"]:.4f}',flush=True)
        save_torch(out/'final.pth',core)
        final,_=evaluate(model,va,device)
        selected,_=load_checkpoint(out/'best.pth');selected.to(device)
        train_clean,_=evaluate(selected,loader(clean,c['batch_size'],c['workers'],limit=c['calibration_images']),device)
        result=dict(status='complete',config=c,best_epoch=best_epoch,best_validation=best,final_validation=final,
            selected_train_clean=train_clean,train_clean_minus_validation_pp=train_clean['accuracy']-best,
            generalization_gap_scope='fixed clean training subset vs full validation at selected checkpoint',
            elapsed_seconds=elapsed_before+time.perf_counter()-timer,memory=memory(device),
            teacher_sha256=teacher_hash,test_evaluated=False)
        write_json(out/'result.json',result);return result
    except Exception as exc:
        write_json(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),completed_epochs=len(history),memory=memory(device)))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--data-root',default='./data');p.add_argument('--device',default='auto');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.out,a.data_root,a.device,a.resume)

if __name__=='__main__':main()
