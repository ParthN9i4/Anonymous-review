import hashlib,json,os,platform,resource,time
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,d):
 def clean(x):
  if isinstance(x,np.generic):x=x.item()
  if isinstance(x,float) and not np.isfinite(x):return None
  if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
  if isinstance(x,(tuple,list)):return [clean(v) for v in x]
  return x
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(clean(d),indent=2,allow_nan=False)+'\n');t.replace(p)
def env():
 cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor()) if Path('/proc/cpuinfo').exists() else platform.processor()
 return dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),host=platform.node(),cpu_model=cpu,affinity_cpus=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
  slurm_job=os.environ.get('SLURM_JOB_ID'),cpu_threads=os.environ.get('SLURM_CPUS_PER_TASK'),
  process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  memory_scope='Whole Python process high-water mark; excludes child workers; Linux KiB')
def verify(root):
 for relative,expected in json.loads((Path(root)/'source_hashes.json').read_text()).items():
  if sha(Path(root)/'code'/relative)!=expected:raise ValueError('Changed snapshot: '+relative)
def errors(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);f=np.isfinite(a)&np.isfinite(b)
 d=np.abs(a[f]-b[f]);den=np.linalg.norm(b[f])
 return dict(n=a.size,both_finite=int(f.sum()),invalid_reference=int((~np.isfinite(b)).sum()),
  invalid_other=int((~np.isfinite(a)).sum()),max_abs=float(d.max()) if d.size else None,
  rmse=float(np.sqrt(np.mean(d*d))) if d.size else None,
  relative_l2=float(np.linalg.norm(a[f]-b[f])/max(den,1e-12)) if d.size else None)
