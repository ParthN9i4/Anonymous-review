import argparse,json
from pathlib import Path
import torch
from worker import load
from data_eval import subsets
from pvit_lab.data import loader
from polynomial import calibrate,convert
from io_utils import write,verify
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root);verify(root)
m=json.loads((root/'manifest.json').read_text());torch.set_num_threads(2)
for j in m['endpoints']:load(j['source'])
base=load(m['conversion'][0]['source']).cuda().eval();fit,gate,_=subsets(m['data_root'],128,evaluation=False)
params,ids=calibrate(base,loader(fit,workers=2),'cuda','budget');model,rec=convert(base,params,'all')
x=next(iter(loader(gate,workers=2)))[0].cuda()
with torch.no_grad():
 rec.begin(len(x),x.device);z=model(x)
write(root/'smoke.json',dict(status='structural_checks_passed',reference_finite=bool(torch.isfinite(base(x)).all()),converted_invalid_rows=int((~torch.isfinite(z).all(1)).sum()),note='Numeric failure is recorded, not a reason to suppress the campaign.'))
