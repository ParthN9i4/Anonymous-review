"""Frozen experiment plan. Change only by creating a new version/campaign."""
SEEDS=[42,43,44]
MASKS=['gelu','norm','attention','all']
FIT_N=2048
GATE_N=2048
GATE_OFFSET=2048
TEST_N=10000
SHIFT_N=1000
LOGIT_ERROR=.1
STATIC_BAD_RATE=.01
MARGIN_ERROR_FRACTION=.25
SHIFT_SEED=20260920


def schedule():
 conversion=[dict(name=f'preserve_s{s}_{p}_{mask}_n2048',seed=s,profile=p,mask=mask,fit_n=FIT_N) for s in SEEDS for p in ('budget','accurate') for mask in MASKS]
 conversion += [dict(name=f'preserve_s42_{p}_all_n128',seed=42,profile=p,mask='all',fit_n=128) for p in ('budget','accurate')]
 endpoint=[dict(name=f'{prefix}_{arm}_seed{s}',seed=s) for s in SEEDS for prefix,arm in [('power','P'),('power','Q'),('legacy','B'),('legacy','E'),('legacy','F')]]
 fhe=[dict(name=j['name']+f'_b{b}',parent=j['name'],block=b,profile=j['profile']) for j in conversion if j['mask']=='all' and j['fit_n']==FIT_N for b in (0,5)]
 return dict(conversion=conversion,endpoints=endpoint,fhe=fhe)
