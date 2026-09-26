"""Optional TenSEAL replay of one TRAINED quadratic on captured training inputs.

CPU primitive only. Does not implement an encrypted block or full-model FHE.
The evaluator gets a public context; decryption occurs only after the polynomial.
API operation counts are not internal RNS/rescale counts or a compiler depth trace.
"""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
from .model import load_checkpoint,PolyGELU
from .common import digest,write_json


def run(checkpoint,calibration,out,mask=None,layer=0,repeats=20,threads=2):
    import tenseal as ts
    c=json.loads(Path(calibration).read_text())
    if c['checkpoint_sha256']!=digest(checkpoint):raise ValueError('Calibration/checkpoint provenance mismatch')
    model,_=load_checkpoint(checkpoint,mask);act=model.blocks[layer].act
    if not isinstance(act,PolyGELU):raise ValueError('This primitive replays trained quadratic GELU only')
    x=np.asarray(c['census'][f'blocks.{layer}.gelu_in']['samples'][:4096],dtype=np.float64)
    if not len(x) or not np.isfinite(x).all():raise ValueError('No finite captured inputs')
    a,b,c0=[float(getattr(act,k).detach()) for k in ('a','b','c')];reference=a*x*x+b*x+c0
    t=time.perf_counter()
    client=ts.context(ts.SCHEME_TYPE.CKKS,poly_modulus_degree=8192,coeff_mod_bit_sizes=[60,40,40,60],n_threads=threads)
    client.global_scale=2**40;client.generate_relin_keys()
    keygen=time.perf_counter()-t
    public_bytes=client.serialize(save_secret_key=False,save_galois_keys=False,save_relin_keys=True)
    public=ts.context_from(public_bytes,n_threads=threads)
    assert not public.has_secret_key()
    def once():
        t=time.perf_counter();encrypted=ts.ckks_vector(client,x.tolist());encrypt=time.perf_counter()-t
        t=time.perf_counter();incoming=encrypted.serialize();serialize_in=time.perf_counter()-t
        t=time.perf_counter();v=ts.ckks_vector_from(public,incoming);deserialize_in=time.perf_counter()-t
        t=time.perf_counter();y=v.square()*a+v*b+c0;compute=time.perf_counter()-t
        t=time.perf_counter();outgoing=y.serialize();serialize_out=time.perf_counter()-t
        t=time.perf_counter();answer=ts.ckks_vector_from(client,outgoing).decrypt();decrypt=time.perf_counter()-t
        return dict(encryption_seconds=encrypt,input_serialization_seconds=serialize_in,
            evaluator_deserialization_seconds=deserialize_in,evaluator_compute_seconds=compute,
            output_serialization_seconds=serialize_out,client_deserialize_decrypt_seconds=decrypt,
            input_ciphertext_bytes=len(incoming),output_ciphertext_bytes=len(outgoing),
            max_absolute_error=float(np.max(np.abs(np.asarray(answer)-reference))))
    once() # one excluded warm-up evaluation, including fresh encryption
    trials=[once() for _ in range(repeats)]
    result=dict(scope='one trained quadratic GELU; captured training activation reservoir; no full attention/block/model execution',
        backend='TenSEAL',version=getattr(ts,'__version__','unknown'),threads=threads,
        ring_dimension=8192,coeff_modulus_bits=[60,40,40,60],initial_scale=2**40,
        security='SEAL default context checks; no independent estimator run in this harness',
        packed_values=len(x),allocated_slots=4096,slot_utilization=len(x)/4096,
        coefficients=[c0,b,a],checkpoint_sha256=digest(checkpoint),calibration_sha256=digest(calibration),
        key_generation_seconds=keygen,public_context_with_relin_bytes=len(public_bytes),
        evaluator_has_secret_key=public.has_secret_key(),bootstrap_calls=0,rotation_calls=0,
        source_level_calls=dict(ciphertext_square=1,ciphertext_plaintext_multiply=2,ciphertext_add=1,plaintext_add=1),
        source_count_scope='one packed evaluation; library may perform automatic rescaling/relinearization/modulus alignment internally',
        trials=trials,p50_evaluator_seconds=float(np.median([r['evaluator_compute_seconds'] for r in trials])),
        p95_evaluator_seconds=float(np.quantile([r['evaluator_compute_seconds'] for r in trials],.95)),
        host_process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        memory_scope='combined client/evaluator in one process; not isolated server peak RAM',
        network_latency_measured=False)
    write_json(out,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--calibration',required=True)
    p.add_argument('--out',required=True);p.add_argument('--mask');p.add_argument('--layer',type=int,default=0)
    p.add_argument('--repeats',type=int,default=20);p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    print(json.dumps(run(a.checkpoint,a.calibration,a.out,a.mask,a.layer,a.repeats,a.threads),indent=2))

if __name__=='__main__':main()
