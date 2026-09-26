import hashlib,json,os,platform,random,resource
from pathlib import Path
import numpy as np
import torch


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def code_digest():
    return {p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False));os.replace(tmp,path)


def save_torch(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(data,tmp);os.replace(tmp,path)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def environment(device):
    return dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
        device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
        visible_memory_bytes=torch.cuda.get_device_properties(device).total_memory if device.type=='cuda' else None,
        slurm_job=os.environ.get('SLURM_JOB_ID'),slurm_array_task=os.environ.get('SLURM_ARRAY_TASK_ID'),
        cpu_threads=torch.get_num_threads(),code_sha256=code_digest())


def memory(device):
    return dict(host_process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
        scope='PyTorch process only; RSS excludes worker aggregation; not encrypted-inference cost')


def device_for(name='auto',fraction=.2):
    d=torch.device('cuda' if name=='auto' and torch.cuda.is_available() else 'cpu' if name=='auto' else name)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','2')))
    if d.type=='cuda':
        # Resolve the process-visible CUDA/MIG device to a concrete index.
        index = d.index if d.index is not None else torch.cuda.current_device()
        d = torch.device('cuda', index)
        torch.cuda.set_per_process_memory_fraction(fraction, index)
    return d
