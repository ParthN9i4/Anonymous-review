"""Link actual P-mode range violations to per-image errors; summarize collapse.
Associations are not causal claims. Thresholds must be fixed before inspecting
new confirmatory runs. Historical test data is not used here.
"""
import argparse,json
from pathlib import Path
import numpy as np
from .common import write_json


def p_mode(path):
    path=Path(path);s=np.load(path/'s_predictions.npz');p=np.load(path/'p_predictions.npz')
    assert np.array_equal(s['ids'],p['ids']) and np.array_equal(s['labels'],p['labels'])
    circuit=json.loads((path/'circuit.json').read_text());census=json.loads((path/'p_census.json').read_text())
    active=[k+'.gelu_in' for k in circuit['gelu']]+list(circuit['reciprocal'])+[k+'.p_denominator' for k in circuit['norm']]
    bad=set()
    for site in active:bad.update(census.get(site,{}).get('violating_images',[]))
    violated=np.array([int(i) in bad for i in s['ids']]);finite=np.isfinite(s['logits']).all(1)&np.isfinite(p['logits']).all(1)
    err=np.full(len(finite),np.inf);err[finite]=np.abs(s['logits'][finite]-p['logits'][finite]).max(1)
    ranked=np.sort(s['logits'],axis=1);margin=ranked[:,-1]-ranked[:,-2]
    flipped=(s['logits'].argmax(1)!=p['logits'].argmax(1))|~finite
    rows=[]
    for state in (False,True):
        subset=violated==state
        rows.append(dict(violated=state,n=int(subset.sum()),flips=int((flipped&subset).sum()),
            flip_rate=float(flipped[subset].mean()) if subset.any() else None))
    cert=finite&(2*err<margin)
    result=dict(active_sites=active,groups=rows,finite_pairs=int(finite.sum()),
        margin_condition_holds=int(cert.sum()),flips_despite_margin_condition=int((cert&flipped).sum()),
        note='2*max logit error < original top-two margin is a per-image sufficient condition for unchanged argmax; no population guarantee')
    write_json(path/'range_error_association.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for state,label in [(False,'Inside all active ranges'),(True,'At least one range violation')]:
        idx=(violated==state)&finite
        if idx.any():
            axes[0].scatter(np.maximum(margin[idx],1e-12),np.maximum(2*err[idx],1e-12),s=8,alpha=.4,label=label)
            x=np.sort(err[idx]);axes[1].plot(x,np.arange(1,len(x)+1)/len(x),label=label)
    axes[0].plot([1e-8,1e4],[1e-8,1e4],'k--',lw=1);axes[0].set_xscale('log');axes[0].set_yscale('log')
    axes[0].set_xlabel('S-mode top-two logit margin');axes[0].set_ylabel('Twice maximum logit error')
    axes[1].set_xscale('symlog',linthresh=1e-6);axes[1].set_xlabel('Maximum logit error');axes[1].set_ylabel('Empirical CDF')
    axes[1].legend(fontsize=8);fig.suptitle('P-mode validation: measured errors, not approximation guarantees')
    fig.savefig(path/'range_error.png',dpi=180);fig.savefig(path/'range_error.pdf');plt.close(fig)
    return result


def training(root,out,tolerance=2.,consecutive=5):
    rows=[]
    for path in sorted(Path(root).rglob('history.json')):
        protocol=json.loads((path.parent/'protocol.json').read_text());c=protocol['config'];chance=100/(10 if c['dataset']=='cifar10' else 100)
        hist=json.loads(path.read_text());values=np.array([r['validation']['accuracy'] for r in hist]);near=values<=chance+tolerance
        streak=0;first=None;post_learning=None;learned=False
        for i,flag in enumerate(near):
            learned=learned or values[i]>=chance+20
            streak=streak+1 if flag else 0
            if streak>=consecutive:
                if first is None:first=i+2-consecutive
                if learned and post_learning is None:post_learning=i+2-consecutive
        rows.append(dict(path=str(path.parent),seed=c['seed'],epochs_observed=len(values),
            epochs_near_chance=int(near.sum()),first_near_chance_streak=first,post_learning_collapse_start=post_learning,
            final_accuracy=float(values[-1]) if len(values) else None,
            best_minus_final=float(values.max()-values[-1]) if len(values) else None,
            result_complete=(path.parent/'result.json').exists(),exception_recorded=(path.parent/'failure.json').exists()))
    write_json(out,dict(criterion=dict(chance_tolerance_pp=tolerance,consecutive_epochs=consecutive,
        post_learning_threshold='chance+20 percentage points',note='Initial failure to learn and post-learning collapse are separate outcomes'),runs=rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--p-dir');p.add_argument('--runs');p.add_argument('--out',default='collapse_summary.json')
    p.add_argument('--tolerance-pp',type=float,default=2.);p.add_argument('--consecutive',type=int,default=5);a=p.parse_args()
    if a.p_dir:print(json.dumps(p_mode(a.p_dir),indent=2))
    if a.runs:training(a.runs,a.out,a.tolerance_pp,a.consecutive)

if __name__=='__main__':main()
