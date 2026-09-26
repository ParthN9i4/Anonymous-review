"""Generate reviewable manifests; execute one Slurm array entry. Never submits."""
import argparse,json
from pathlib import Path
from .common import code_digest,write_json,digest
from .model import legacy_spec
from dataclasses import asdict

MASKS={'0':'000','A':'100','B':'010','C':'001','D':'110','E':'111','F':'011','G':'101'}


def generate(phase,repo,out,dataset,seeds,epochs):
    repo=Path(repo).resolve();out=Path(out).resolve();jobs=[]
    def checkpoint(seed,arm='teacher'):
        pattern=f'substitution_ablation_{dataset}_clean8_seeds{seed}_{arm}_seed{seed}.pth'
        path=repo/'checkpoints_substitution'/pattern
        if not path.is_file():raise FileNotFoundError(f'Missing required trained checkpoint: {path}')
        return str(path)
    def add(seed,name,model=None,**kwargs):
        c=dict(dataset=dataset,seed=seed,epochs=epochs,model=model or {},**kwargs)
        if c.get('loss','kd')!='ce' or c.get('initialization','cold')=='warm':c['teacher']=checkpoint(seed)
        jobs.append(dict(kind='train',name=name,config=c,out=str(out/phase/f'{name}_seed{seed}')))
    for seed in seeds:
        if phase=='audit':
            for arm,mask in {'teacher':'000',**MASKS}.items():
                jobs.append(dict(kind='audit',name=arm,checkpoint=checkpoint(seed,arm),mask=mask,dataset=dataset,
                    bn_recalibrate=mask[2]=='1',out=str(out/phase/f'{arm}_seed{seed}')))
        elif phase=='baseline':
            for recipe in ('legacy','regularized'):
                for norm in ('ln','bn'):add(seed,f'{recipe}_{norm}',dict(norm=norm),recipe=recipe,loss='ce')
        elif phase=='norm':
            for norm,final in [('ln','ln'),('bn','ln'),('ln','bn'),('bn','bn')]:
                add(seed,f'body_{norm}_final_{final}',dict(norm=norm,final_norm=final))
        elif phase=='screen':
            for attn in ('softmax','raw','power_stable','power_raw','bpmax'):
                add(seed,attn,dict(attention=attn))
        elif phase=='kd':
            for init in ('cold','warm'):
                for loss in ('ce','kd'):
                    add(seed,f'{init}_{loss}',dict(attention='power_stable'),initialization=init,loss=loss)
        elif phase=='temperature':
            for temp in (1.,2.,4.,8.):
                for init in ('cold','warm'):
                    add(seed,f'{init}_T{int(temp)}',dict(attention='power_stable'),initialization=init,temperature=temp)
        elif phase=='kd_scale':
            for attn in ('softmax','power_stable'):
                for scale in (False,True):
                    add(seed,f'{attn}_T2_{scale}',dict(attention=attn),t_squared=scale)
        elif phase=='schedule':
            for init in ('cold','warm'):
                for schedule in ('immediate','delay_ramp'):
                    add(seed,f'{init}_{schedule}',dict(attention='power_stable'),initialization=init,
                        kd_start_fraction=.3 if schedule=='delay_ramp' else 0.,kd_ramp_fraction=.1 if schedule=='delay_ramp' else 0.)
        elif phase=='mechanism':
            for attn in ('square_row','square_fixed'):
                for norm in ('ln','bn'):add(seed,f'{attn}_{norm}',dict(attention=attn,norm=norm))
        elif phase=='gain':
            for attn in ('raw','raw_relu','raw_l1','relu_row'):
                for norm in ('ln','bn'):add(seed,f'{attn}_{norm}',dict(attention=attn,norm=norm))
        elif phase=='powerformer':
            for attn,norm in [('softmax','ln'),('bpmax','ln'),('softmax','batchln'),('bpmax','batchln')]:
                for loss in ('kd','feature_kd'):
                    add(seed,f'{attn}_{norm}_{loss}',dict(attention=attn,norm=norm),initialization='warm',loss=loss)
        elif phase=='fixes':
            for attn,norm in [('raw','bn'),('clamp','bn'),('relu_row','bn'),('clamp_row','bn'),('clamp01_row','bn'),('raw','rms')]:
                add(seed,f'{attn}_{norm}',dict(attention=attn,norm=norm,activation='quadratic'),t_squared=False)
        elif phase=='gelu':
            for attn in ('softmax','power_raw'):
                for norm in ('ln','bn'):
                    for act in ('gelu','quadratic'):add(seed,f'{attn}_{norm}_{act}',dict(attention=attn,norm=norm,activation=act))
        else:raise ValueError(phase)
    for j in jobs:
        path=j.get('checkpoint',j.get('config',{}).get('teacher'))
        if path:j['input_sha256']=digest(path)
    return dict(schema=1,phase=phase,code_sha256=code_digest(),data_root=str(repo/'data'),jobs=jobs,
        note='No jobs have been submitted. Training selects only on validation. Audit never opens test.')


def execute(path,index,device='auto'):
    manifest=json.loads(Path(path).read_text())
    if manifest['code_sha256']!=code_digest():raise ValueError('Code changed after manifest generation; generate a new campaign directory')
    j=manifest['jobs'][index];source=j.get('checkpoint',j.get('config',{}).get('teacher'))
    if source and digest(source)!=j['input_sha256']:raise ValueError('Input checkpoint changed since manifest creation')
    if j['kind']=='train':
        from .train import run
        return run(j['config'],j['out'],manifest['data_root'],device,resume=True)
    from .audit import run
    return run(j['checkpoint'],j['out'],manifest['data_root'],j['mask'],j['dataset'],device,bn_recalibrate=j['bn_recalibrate'])


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('make');g.add_argument('--phase',required=True,choices=['audit','baseline','norm','screen','kd','temperature','kd_scale','schedule','mechanism','gain','powerformer','fixes','gelu'])
    g.add_argument('--repo',required=True);g.add_argument('--out',required=True);g.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100'])
    g.add_argument('--seeds',type=int,nargs='+',default=[42,43]);g.add_argument('--epochs',type=int,default=100)
    r=sub.add_parser('run');r.add_argument('--manifest',required=True);r.add_argument('--index',required=True,type=int);r.add_argument('--device',default='auto')
    a=p.parse_args()
    if a.command=='make':
        m=generate(a.phase,a.repo,a.out,a.dataset,a.seeds,a.epochs);path=Path(a.out)/f'{a.phase}.json'
        if path.exists():raise FileExistsError(f'{path}: preserve existing manifest; choose a new --out')
        write_json(path,m);print(f'{path}\n{len(m["jobs"])} jobs; not submitted')
    else:execute(a.manifest,a.index,a.device)

if __name__=='__main__':main()
