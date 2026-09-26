"""Plot actual learned GELUs against the trained input distribution, per layer."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .model import load_checkpoint,PolyGELU
from .common import digest,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--calibration',required=True)
    p.add_argument('--mask');p.add_argument('--out',required=True);a=p.parse_args()
    c=json.loads(Path(a.calibration).read_text())
    if c['checkpoint_sha256']!=digest(a.checkpoint):raise ValueError('Checkpoint/calibration mismatch')
    m,_=load_checkpoint(a.checkpoint,a.mask);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[]
    for i,b in enumerate(m.blocks):
        if not isinstance(b.act,PolyGELU):continue
        r=c['census'][f'blocks.{i}.gelu_in'];s=np.array(r['samples']);lo,hi=r['minimum'],r['maximum']
        if not len(s):continue
        grid=np.linspace(lo-1e-8,hi+1e-8,1500);aa,bb,cc=[float(getattr(b.act,k).detach()) for k in ('a','b','c')]
        true=torch.nn.functional.gelu(torch.tensor(grid)).numpy();approx=aa*grid**2+bb*grid+cc
        actual=torch.nn.functional.gelu(torch.tensor(s)).numpy();error=np.abs(aa*s*s+bb*s+cc-actual)
        x=torch.tensor(grid,requires_grad=True);g=torch.nn.functional.gelu(x);derivative=torch.autograd.grad(g.sum(),x)[0].detach().numpy()
        fig,ax=plt.subplots(2,2,figsize=(10,7),layout='constrained')
        ax[0,0].plot(grid,true,label='GELU');ax[0,0].plot(grid,approx,'--',label='Learned quadratic');ax[0,0].legend();ax[0,0].set_ylabel('Output')
        ax[0,1].plot(grid,np.abs(approx-true));ax[0,1].set_ylabel('Absolute function error')
        ax[1,0].hist(s,bins=60,density=True);ax[1,0].set_ylabel('Training-input reservoir density')
        ax[1,1].plot(grid,derivative,label="GELU derivative");ax[1,1].plot(grid,2*aa*grid+bb,'--',label='Quadratic derivative');ax[1,1].legend();ax[1,1].set_ylabel('Derivative')
        for axes in ax.flat:
            axes.set_xlabel('Activation input');axes.grid(alpha=.15)
            if lo<3 and hi>-3:axes.axvspan(max(lo,-3),min(hi,3),color='green',alpha=.07)
        fig.suptitle(f'Block {i}: {aa:.5g} x² + {bb:.5g} x + {cc:.5g}\nShading = original initialization fit interval, not a validated operating interval')
        fig.savefig(out/f'gelu_block_{i}.png',dpi=180);fig.savefig(out/f'gelu_block_{i}.pdf');plt.close(fig)
        rows.append(dict(block=i,coefficients=[aa,bb,cc],observed_min=lo,observed_max=hi,
            reservoir_n=len(s),reservoir_mae=float(error.mean()),reservoir_p99_error=float(np.quantile(error,.99)),
            grid_max_error=float(np.max(np.abs(approx-true))),
            note='Sample/grid errors do not prove uniform bounds or end-to-end usefulness'))
    write_json(out/'activation_errors.json',rows)

if __name__=='__main__':main()
