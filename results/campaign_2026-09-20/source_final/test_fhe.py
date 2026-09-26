"""Real CKKS smoke: full 65-column softmax row, plus degree15 polynomial.
Synthetic inputs only; does not establish trained-model accuracy.
"""
import os,time
import numpy as np
import tenseal as ts
from fhe import execute
from polynomial_numpy import softmax_poly,polynomial
from io_utils import write

def run():
 n=32768;bits=40;t=time.perf_counter()
 client=ts.context(ts.SCHEME_TYPE.CKKS,poly_modulus_degree=n,coeff_mod_bit_sizes=[60]+[bits]*18+[60],n_threads=2);client.global_scale=2**bits
 public=ts.context_from(client.serialize(save_secret_key=False),n_threads=2)
 assert not public.has_secret_key()
 # An exponential-root Taylor polynomial on a narrow domain, coefficients fixed for a transport/arithmetic test.
 root=dict(center=0.,radius=1.,lo=-1.,hi=1.,coefficients=[1.,.5,.125,1/48,1/384,1/3840,1/46080,1/645120,1/10321920])
 p=dict(root=root,floor=1e-12,reciprocal=dict(lo=50.,hi=90.,steps=4))
 values=np.stack([np.linspace(-.5,.5,65),np.linspace(.4,-.4,65)])
 actual,cost=execute(values,p,'softmax_row',client,public,ts);truth=softmax_poly(values,p)
 error=float(np.abs(actual-truth).max());assert np.isfinite(actual).all() and error<1e-3,('CKKS softmax smoke error',error)
 poly=dict(center=1.,radius=1.,coefficients=[1.,.2,.03,.004]+[1e-4]*12)
 y,c=execute(np.array([[.5,1.,1.5]]),poly,'gelu',client,public,ts);e=float(np.abs(y-polynomial(np.array([[.5,1.,1.5]]),poly)).max());assert e<1e-3,e
 print(dict(status='passed',softmax_max_error=error,degree15_max_error=e,softmax_cost=cost,total_seconds=time.perf_counter()-t))
if __name__=='__main__':run()
