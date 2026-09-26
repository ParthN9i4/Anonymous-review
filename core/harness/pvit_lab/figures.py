"""Scientific figures from measured JSON only. No fabricated trajectories."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t


ORDER=['000','100','010','001','110','101','011','111']


def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def legacy(path,out):
    docs=json.loads(Path(path).read_text());best={m:{} for m in ORDER};final={m:{} for m in ORDER};teachers={}
    for doc in docs.values():
        for seed,r in doc['teacher_per_seed'].items():teachers[int(seed)]=r['test_at_best_val']
        for name,rows in doc['with_kd_per_seed'].items():
            mask=doc['config_masks'][name]
            for seed,r in rows.items():best[mask][int(seed)]=r['test_at_best_val'];final[mask][int(seed)]=r['test_final']
    seeds=sorted(set.intersection(*(set(v) for v in best.values())))
    a=np.array([[best[m][s] for s in seeds] for m in ORDER]);b=np.array([[final[m][s] for s in seeds] for m in ORDER])
    fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained')
    for ax,values,title in zip(axes,[a,b],['Selected by validation','Final epoch']):
        im=ax.imshow(values,vmin=0,vmax=100,cmap='viridis',aspect='auto');ax.set_title(title)
        ax.set_xticks(range(len(seeds)),seeds);ax.set_yticks(range(8),ORDER);ax.set_xlabel('Seed');ax.set_ylabel('GELU / attention / BN mask')
        for i in range(8):
            for j in range(len(seeds)):ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',color='black' if values[i,j]>65 else 'white',fontsize=9)
    fig.colorbar(im,ax=axes,label='Test accuracy (%)',shrink=.8);save(fig,out,'clean_seed_heatmaps')
    contrasts={
        'GELU | ordinary':a[1]-a[0], 'GELU | raw attention':a[4]-a[2],
        'GELU | BN':a[5]-a[3], 'GELU | raw attention + BN':a[7]-a[6],
        'BN | ordinary':a[3]-a[0],
        'Attention × BN | GELU off':a[6]-a[2]-a[3]+a[0],
        'Attention × BN | GELU on':a[7]-a[4]-a[5]+a[1]}
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for i,(name,v) in enumerate(contrasts.items()):
        ax.scatter(v,np.full(len(v),i)+np.linspace(-.12,.12,len(v)),label=None,color=plt.cm.tab10(np.arange(len(v))))
        ax.plot([v.mean()],[i],marker='D',color='black',markersize=5)
    ax.axvline(0,color='grey',lw=1);ax.set_yticks(range(len(contrasts)),contrasts);ax.set_xlabel('Paired accuracy difference / interaction (percentage points)')
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=plt.cm.tab10(j),marker='o',linestyle='',label=str(s)) for j,s in enumerate(seeds)]
    ax.legend(handles=handles,title='Seed',loc='upper right',fontsize=8,title_fontsize=8)
    ax.set_title('Every seed shown; black diamond = mean');save(fig,out,'paired_effects')
    rows=[]
    for i,m in enumerate(ORDER):rows.append(dict(mask=m,n=len(seeds),mean=float(a[i].mean()),sample_sd=float(a[i].std(ddof=1)),final_mean=float(b[i].mean()),**{f'seed{s}':a[i,j] for j,s in enumerate(seeds)}))
    with open(out/'clean_summary.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    cr={}
    for name,v in contrasts.items():
        se=v.std(ddof=1)/np.sqrt(len(v));half=t.ppf(.975,len(v)-1)*se
        cr[name]=dict(mean=float(v.mean()),sample_sd=float(v.std(ddof=1)),paired_t_interval_95=[float(v.mean()-half),float(v.mean()+half)],per_seed=dict(zip(map(str,seeds),v.tolist())))
    (out/'contrasts.json').write_text(json.dumps(cr,indent=2))


def histories(root,out):
    paths=sorted(Path(root).rglob('history.json'))
    for index,path in enumerate(paths):
        rows=json.loads(path.read_text())
        if not rows:continue
        x=[r['epoch'] for r in rows];fig,ax=plt.subplots(3,1,figsize=(8,7),sharex=True,layout='constrained')
        ax[0].plot(x,[r['validation']['accuracy'] for r in rows]);ax[0].set_ylabel('Validation accuracy (%)')
        ax[1].plot(x,[r['gradient_norm_mean'] for r in rows],label='Mean');ax[1].plot(x,[r['gradient_norm_max'] for r in rows],label='Maximum');ax[1].set_yscale('log');ax[1].set_ylabel('Gradient norm');ax[1].legend()
        for site in ('blocks.0.abs_row_mass','blocks.5.abs_row_mass'):
            epochs=[];values=[]
            for p in sorted((path.parent/'telemetry').glob('epoch_*.json')):
                data=json.loads(p.read_text())['census']
                if site in data:epochs.append(int(p.stem.split('_')[-1]));values.append(data[site]['quantiles'].get('q99'))
            if epochs:ax[2].plot(epochs,values,marker='.',label=site)
        ax[2].set_ylabel('Attention |row mass| q99');ax[2].set_yscale('symlog',linthresh=1);ax[2].set_xlabel('Epoch');ax[2].legend()
        fig.suptitle(path.parent.name);save(fig,out,f'trajectory_{index:03d}_{path.parent.name}')
    (out/'trajectory_sources.json').write_text(json.dumps([str(p) for p in paths],indent=2))


def census_plot(path,out):
    data=json.loads(Path(path).read_text());data=data.get('census',data)
    sites=['gelu_in','scores','attn.denominator','abs_row_mass','residual_out.rms']
    depths=sorted({int(k.split('.')[1]) for k in data if k.startswith('blocks.')})
    arr=np.full((len(depths),len(sites)),np.nan)
    for i,d in enumerate(depths):
        for j,s in enumerate(sites):
            r=data.get(f'blocks.{d}.{s}')
            if r:arr[i,j]=100*(r['below']+r['above']+r['nonfinite'])/max(1,r['count'])
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');im=ax.imshow(arr,vmin=0,vmax=100,cmap='magma',aspect='auto')
    ax.set_xticks(range(len(sites)),sites,rotation=25,ha='right');ax.set_yticks(range(len(depths)),depths)
    ax.set_ylabel('Block');ax.set_title('Observed values outside supplied ranges or nonfinite (%)')
    fig.colorbar(im,ax=ax);save(fig,out,'range_violations')


def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy-results');p.add_argument('--runs');p.add_argument('--census');p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.legacy_results:legacy(a.legacy_results,out)
    if a.runs:histories(a.runs,out)
    if a.census:census_plot(a.census,out)

if __name__=='__main__':main()
