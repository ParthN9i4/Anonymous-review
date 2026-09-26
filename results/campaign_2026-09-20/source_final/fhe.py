"""Actual leveled CKKS on frozen conversion arithmetic; no intermediate decryption."""
import argparse,json,os,time,traceback
from pathlib import Path
import numpy as np
from io_utils import write,sha,env,errors
from polynomial_numpy import polynomial,reciprocal,softmax_poly,softmax_exact

def crypto_softmax(columns,p):
 mean=columns[0]
 for c in columns[1:]:mean=mean+c
 mean=mean*(1/len(columns));r=p['root']
 numerator=[]
 for c in columns:
  z=(c-mean-r['center'])*(1/r['radius']);u=z.polyval(r['coefficients']).square()+p['floor'];numerator.append(u)
 denominator=numerator[0]
 for u in numerator[1:]:denominator=denominator+u
 rp=p['reciprocal'];scale=2/(rp['lo']+rp['hi']);e=-(denominator*scale)+1;y=e+1
 for _ in range(rp['steps']):e=e.square();y=y*(e+1)
 inverse=y*scale
 return [u*inverse for u in numerator]


def execute(values,params,operation,client,server,ts):
 x=np.asarray(values,dtype=np.float64)
 # Independent inputs occupy SIMD lanes, one ciphertext per attention-column.
 inp=[x[:,j] for j in range(x.shape[1])] if operation=='softmax_row' else [x.reshape(-1)]
 start=time.perf_counter();encrypted=[ts.ckks_vector(client,v.tolist()) for v in inp];enc_time=time.perf_counter()-start
 start=time.perf_counter();payload=[v.serialize() for v in encrypted];public=[ts.ckks_vector_from(server,b) for b in payload];upload=time.perf_counter()-start
 initial=public[0].ciphertext()[0].coeff_modulus_size();start=time.perf_counter()
 if operation=='softmax_row':out=crypto_softmax(public,params)
 else:
  p=params;out=[((public[0]-p['center'])*(1/p['radius'])).polyval(p['coefficients'])]
 elapsed=time.perf_counter()-start;final=out[0].ciphertext()[0].coeff_modulus_size()
 start=time.perf_counter();reply=[v.serialize() for v in out];ser_time=time.perf_counter()-start
 start=time.perf_counter();decoded=[np.asarray(ts.ckks_vector_from(client,b).decrypt()) for b in reply];dec_time=time.perf_counter()-start
 y=np.stack(decoded,axis=1) if operation=='softmax_row' else decoded[0].reshape(x.shape)
 logical=(dict(ciphertext_add_or_subtract_outside_polyval=3*len(inp)-2,ciphertext_multiply_outside_polyval=len(inp)+params['reciprocal']['steps'],ciphertext_square_outside_polyval=len(inp)+params['reciprocal']['steps'],ciphertext_plaintext_scalar_multiply_outside_polyval=len(inp)+3,plaintext_add_or_subtract_outside_polyval=2*len(inp)+2+params['reciprocal']['steps'],polyval_calls=len(inp)) if operation=='softmax_row' else dict(ciphertext_plaintext_scalar_multiply_outside_polyval=1,plaintext_subtract_outside_polyval=1,polyval_calls=1))
 return y,dict(logical_outer_operations=logical,logical_count_scope='Explicit vector operations excluding backend polynomial evaluation internals; not total physical multiplications or scalar MACs.',encrypt_seconds=enc_time,input_serialization_load_seconds=upload,server_evaluation_seconds=elapsed,
   output_serialization_seconds=ser_time,load_and_decrypt_seconds=dec_time,input_ciphertext_bytes=sum(map(len,payload)),output_ciphertext_bytes=sum(map(len,reply)),
   input_ciphertexts=len(inp),output_ciphertexts=len(out),input_modulus_primes=initial,output_modulus_primes=final,levels_consumed=initial-final,
   intermediate_decryptions=0,physical_rotations=None,physical_relinearizations=None,physical_rescales=None,
   ciphertext_scale=float(out[0].ciphertext()[0].scale))


def benchmark_fixture(fixture,frozen,metadata,out,block,bits=40,repeats=2):
 import tenseal as ts
 out=Path(out);out.mkdir(parents=True,exist_ok=True);raw=np.load(fixture);frozen=json.loads(Path(frozen).read_text());meta=json.loads(Path(metadata).read_text())
 if sha(fixture)!=meta['sha256']:raise ValueError('Fixture changed')
 if sha(Path(metadata).parent/'frozen.json')!=meta['frozen_sha256']:raise ValueError('Frozen arithmetic changed')
 degree=32768;chain=[60]+[bits]*18+[60]
 if sum(chain)>881:raise ValueError('Exceeds SEAL tc128 modulus allowance')
 t=time.perf_counter();client=ts.context(ts.SCHEME_TYPE.CKKS,poly_modulus_degree=degree,coeff_mod_bit_sizes=chain,n_threads=int(os.environ.get('SLURM_CPUS_PER_TASK','4')));client.global_scale=2**bits
 setup=time.perf_counter()-t;t=time.perf_counter();public_bytes=client.serialize(save_secret_key=False);server=ts.context_from(public_bytes,n_threads=int(os.environ.get('SLURM_CPUS_PER_TASK','4')));context_transfer=time.perf_counter()-t
 if server.has_secret_key():raise RuntimeError('Server has a secret key')
 pp=frozen['parameters'];rows=[];decoded={}
 for path in ('source','converted'):
  prefix=f'b{block}_' if path=='source' else f'b{block}_converted_'
  for opname,key,site in [('softmax_row','scores',f'blocks.{block}.attn_act'),('gelu','gelu',f'blocks.{block}.act'),('ln_inverse_sqrt','variance',f'blocks.{block}.norm1')]:
   values=raw[prefix+key];p=pp[site]
   if opname=='softmax_row':
    values=values[:,0,0,:].astype(np.float64)
    arithmetic=softmax_poly(values,p);exact=softmax_exact(values)
    z=values-values.mean(-1,keepdims=True);bounds=p['root'];outside=((z<bounds['lo'])|(z>bounds['hi'])|~np.isfinite(z)).any(1)
   else:
    values=(values[:,0,:] if opname=='gelu' else values[:,0,0]).astype(np.float64);p=p['poly'];arithmetic=polynomial(values,p)
    if opname=='gelu':
     import math
     exact=.5*values*(1+np.vectorize(math.erf)(values/np.sqrt(2)))
    else:
     with np.errstate(invalid='ignore',divide='ignore'):exact=1/np.sqrt(values)
    outside=((values<p['lo'])|(values>p['hi'])|~np.isfinite(values)).reshape(len(values),-1).any(1)
   record=dict(path=path,operation=opname,shape=list(values.shape),input_domain_warning=outside.tolist(),
     plaintext_approximation_error=errors(arithmetic,exact))
   if not np.isfinite(values).all() or not np.isfinite(arithmetic).all() or not np.isfinite(exact).all():
    rows.append(record|dict(status='INVALID_PLAINTEXT_INPUT_OR_TARGET',invalid_inputs=int((~np.isfinite(values)).sum()),invalid_arithmetic=int((~np.isfinite(arithmetic)).sum())));continue
   for rep in range(repeats):
    try:
     y,cost=execute(values,p,opname,client,server,ts)
     e=errors(y,arithmetic);total=errors(y,exact)
     perinput=np.max(np.abs(y-arithmetic).reshape(len(y),-1),axis=1)
     if opname=='softmax_row':cost.update(row_sum=y.sum(-1).tolist(),minimum_weight=y.min(-1).tolist(),negative_weight_count=int((y<0).sum()))
     rows.append(record|dict(status='MEASURED',repeat=rep,ckks_error=e,total_error=total,per_input_ckks_max_abs=perinput.tolist(),
        ckks_error_gt_1e_minus3=(perinput>.001).tolist(),**cost));decoded[f'{path}_{opname}_{rep}']=y
    except Exception as exc:rows.append(record|dict(status='BACKEND_EXCEPTION',repeat=rep,error=str(exc)))
 report=dict(status='recorded',block=block,rows=rows,fixture_sha256=sha(fixture),frozen_sha256=meta['frozen_sha256'],
  parameters=dict(scheme='CKKS',backend='TenSEAL',version=ts.__version__,ring_degree=degree,slots=degree//2,scale_bits=bits,modulus_bits=chain,
    security='SEAL default tc128 validation enabled',bootstraps=0,galois_keys=False),context_key_setup_seconds=setup,public_context_load_seconds=context_transfer,
  public_context_bytes=len(public_bytes),server_has_secret_key=server.has_secret_key(),environment=env(),
  packing_scope='Four fixed images packed across SIMD lanes. Softmax uses head0/query0, 65 columns; low slot utilization is disclosed. GELU uses class-token features; LN inverse uses class-token variance.',
  memory_scope='Peak RSS covers client and server contexts in one process and all previous operations; it is not isolated remote-server memory.',
  timing_scope='Two trials, no discarded warm-up; timings per entire packed primitive. Not full-model inference latency.',
  boundary='Attention starts from trained raw QK/sqrt(d) scores; mean-centering, fitted positive exponential polynomial, denominator sum, reciprocal and normalization are encrypted. QK, AV and upstream network are not encrypted.',
  normalization_scope='LN inverse-square-root primitive only, not a complete encrypted LN layer.',
  count_scope='No fabricated physical operation counts. Softmax row uses column packing without rotations; backend physical counts remain uninstrumented.')
 write(out/'report.json',report);np.savez_compressed(out/'decrypted.npz',**decoded);return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('index',type=int);a=p.parse_args();root=Path(a.root);m=json.loads((root/'manifest.json').read_text());j=m['fhe'][a.index];src=root/'conversion'/j['parent'];out=root/'fhe'/j['name'];out.mkdir(parents=True,exist_ok=True)
 try:benchmark_fixture(src/'fhe_inputs.npz',src/'frozen.json',src/'fhe_inputs_meta.json',out,j['block'],35 if j['profile']=='budget' else 40)
 except Exception as e:write(out/'error.json',dict(error=str(e),traceback=traceback.format_exc()));raise
