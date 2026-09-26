import argparse,csv,json,tarfile
from pathlib import Path
from io_utils import write,sha

def csvwrite(path,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r));path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def report(root):
 root=Path(root);m=json.loads((root/'manifest.json').read_text());out=root/'summary';out.mkdir(exist_ok=True)
 gaps=[];detectors=[];cost=[];endpoints=[];crypt=[];complete={}
 for kind,key in [('conversion','conversion'),('endpoints','endpoints'),('fhe','fhe')]:
  complete[kind]=0
  for job in m[key]:
   path=root/kind/job['name']/'report.json'
   if not path.is_file():gaps.append(dict(kind=kind,name=job['name'],error_file=str(path.parent/'error.json')));continue
   d=json.loads(path.read_text());complete[kind]+=1
   if kind=='conversion':
    for split,r in d['splits'].items():
     for name,v in r['detectors'].items():
      detectors.append(dict(configuration=job['name'],seed=job['seed'],profile=job['profile'],mask=job['mask'],fit_n=job['fit_n'],split=split,detector=name,conversion_failures=r['conversion_failure_count'],prediction_flips=r['prediction_flips'],correct_to_wrong=r['correct_to_wrong'],**v))
    for name,v in d['cost'].items():
     if isinstance(v,dict):cost.append(dict(configuration=job['name'],method=name,**{k:x for k,x in v.items() if k!='seconds'}))
   elif kind=='endpoints':
    for split,r in d['splits'].items():endpoints.append(dict(configuration=job['name'],split=split,**{k:v for k,v in r.items() if not isinstance(v,(dict,list))}))
   else:
    for r in d['rows']:
     e=r.get('ckks_error',{});t=r.get('total_error',{});pa=r.get('plaintext_approximation_error',{})
     crypt.append(dict(configuration=job['name'],profile=job['profile'],block=job['block'],path=r['path'],operation=r['operation'],repeat=r.get('repeat'),status=r['status'],error=r.get('error'),ckks_max_abs=e.get('max_abs'),ckks_invalid=e.get('invalid_other'),total_max_abs=t.get('max_abs'),approximation_max_abs=pa.get('max_abs'),**{k:r.get(k) for k in ('server_evaluation_seconds','encrypt_seconds','load_and_decrypt_seconds','input_ciphertext_bytes','output_ciphertext_bytes','levels_consumed')}))
 csvwrite(out/'detectors.csv',detectors);csvwrite(out/'costs.csv',cost);csvwrite(out/'endpoints.csv',endpoints);csvwrite(out/'ckks.csv',crypt)
 counts={}
 for r in crypt:counts[r['status']]=counts.get(r['status'],0)+1
 write(out/'completion.json',dict(planned={k:len(m[k]) for k in complete},recorded=complete,gaps=gaps,ckks_record_statuses=counts,full_encrypted_backbone=False,bootstrapping=False,independent_external_dataset=False))
 lines=['# Frozen-conversion assessment','',f'Recorded reports: {complete}. Structural gaps: {len(gaps)}.',f'CKKS trial/status counts: {counts}.','',
 'Read detector rows per seed/configuration/split. Do not pool correlated inputs, repeated CKKS trials or layers as independent model seeds.',
 'Calibration-only is a configuration-level screen. An all-review decision has zero coverage, not proven zero accepted risk.',
 'The higher-degree profile is named accurate for identification only; neither profile is guaranteed accurate or optimal.',
 'Test data was previously used in the project. Current criteria were frozen before these evaluations; no claim of historically pristine confirmation.',
 'FHE measures encrypted attention-row arithmetic, GELU and LN inverse square root on captured trained values. It excludes QK/AV and the full backbone; no bootstrap.',
 '', 'Primary inspection order: structural gaps; frozen calibration failure rate; detector false negatives AND coverage; FP64 incremental detection AND cost; CKKS numerical errors AND backend exceptions.']
 (out/'READOUT.md').write_text('\n'.join(lines)+'\n')
 try:
  from figures import make
  make(root)
 except Exception as e:write(out/'figure_error.json',dict(error=str(e),note='CSV/JSON results remain available'))
 archive=root.parent/(root.name+'_return.tgz')
 with tarfile.open(archive,'w:gz') as t:
  for p in sorted(root.rglob('*')):
   if p.is_file() and p.name not in ('return_archive.txt','return_sha256.txt') and p.suffix not in ('.pth','.pt','.pyc') and '__pycache__' not in p.parts:t.add(p,arcname=str(p.relative_to(root)))
 (root/'return_archive.txt').write_text(str(archive)+'\n');(root/'return_sha256.txt').write_text(sha(archive)+'\n');print('\n'.join(lines));print('ARCHIVE:',archive)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();report(a.root)
