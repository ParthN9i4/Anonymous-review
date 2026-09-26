"""Descriptive scientific plots; no seed aggregation or fitted decision thresholds."""
import csv,json
from pathlib import Path
import numpy as np

def make(root):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 root=Path(root);out=root/'summary';rows=list(csv.DictReader((out/'detectors.csv').open())) if (out/'detectors.csv').exists() else []
 if rows:
  fig,axes=plt.subplots(1,3,figsize=(13,4),sharex=True,sharey=True)
  methods=['calibration_only','finite_only','range','range_and_precision'];markers=['s','o','^','D']
  for ax,split in zip(axes,['clean','noise','brightness']):
   for method,marker in zip(methods,markers):
    rs=[r for r in rows if r['split']==split and r['detector']==method and r['accepted_failure_rate']!='']
    ax.scatter([float(r['accepted_fraction']) for r in rs],[float(r['accepted_failure_rate']) for r in rs],label=method,marker=marker,alpha=.65,s=30)
   ax.set(title=split,xlabel='Fraction accepted',xlim=(-.02,1.02),ylim=(-.02,1.02));ax.grid(alpha=.2)
  axes[0].set_ylabel('Failure rate among accepted inputs');axes[-1].legend(fontsize=7)
  fig.suptitle('Fixed screens: each point is one checkpoint/conversion configuration\nAll-review configurations have undefined accepted risk and are omitted; see CSV')
  fig.tight_layout();fig.savefig(out/'risk_coverage.png',dpi=180);fig.savefig(out/'risk_coverage.pdf');plt.close(fig)
  # Pair each configuration with itself to isolate the added precision check.
  fig,ax=plt.subplots(figsize=(7,5));lookup={(r['configuration'],r['split'],r['detector']):r for r in rows}
  for r in rows:
   if r['detector']!='range' or r['split']!='clean':continue
   s=lookup[(r['configuration'],'clean','range_and_precision')]
   ax.plot([0,1],[int(r['fn']),int(s['fn'])],alpha=.45,marker='o')
  ax.set(xticks=[0,1],xticklabels=['Range check','Range + FP64 check'],ylabel='Missed conversion failures',title='Incremental detection on clean inputs (paired configurations)');ax.grid(axis='y',alpha=.2)
  fig.tight_layout();fig.savefig(out/'incremental_detection.png',dpi=180);fig.savefig(out/'incremental_detection.pdf');plt.close(fig)
  # Numeric error versus domain exceedance, one fixed configuration, no threshold tuning.
  first=next(iter(sorted((root/'conversion').glob('*budget_all_n2048/clean_predictions.npz'))),None)
  if first:
   a=np.load(first);x=a['range_score'];y=a['target_max_logit_error'];finite=np.isfinite(x)&np.isfinite(y)
   fig,ax=plt.subplots(figsize=(7,4));ax.scatter(x[finite],y[finite],s=5,alpha=.25)
   ax.set_xscale('symlog',linthresh=1e-4);ax.set_yscale('symlog',linthresh=1e-3);ax.axhline(.1,color='red',ls='--')
   ax.set(xlabel='Maximum normalized domain exceedance',ylabel='Maximum absolute logit error',title=f'{first.parent.name}\nNonfinite errors separately recorded: {int((~np.isfinite(y)).sum())}')
   fig.tight_layout();fig.savefig(out/'domain_vs_error.png',dpi=180);fig.savefig(out/'domain_vs_error.pdf');plt.close(fig)
