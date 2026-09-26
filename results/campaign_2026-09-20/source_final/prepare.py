"""Create an isolated campaign using verified existing checkpoint bytes."""
import argparse,json,shutil,tempfile
from pathlib import Path
from protocol import schedule
from io_utils import sha,write

def prepare(repo,results):
 repo=Path(repo).resolve();results=Path(results).resolve();here=Path(__file__).resolve().parent
 inventory=json.loads((here/'checkpoint_inventory.json').read_text());resolved={};missing=[]
 for j in inventory:
  original=Path(j['checkpoint']);p=Path.home()/original.relative_to('/home/anonuser')
  candidates=[p]
  if not p.is_file():
   candidates=list(repo.rglob(original.name))+list(results.rglob(original.name))
  match=next((p for p in candidates if p.is_file() and sha(p)==j['checkpoint_sha256']),None)
  if match is None:missing.append(j['name']);continue
  resolved[j['name']]={k:v for k,v in j.items() if k!='out'};resolved[j['name']]['checkpoint']=str(match)
 if missing:raise ValueError('Missing or changed checkpoints (no submission): '+', '.join(missing))
 data=repo/'data'
 if not (data/'cifar-10-batches-py'/'test_batch').is_file():raise ValueError('Cached CIFAR-10 missing at '+str(data)+'; no automatic download in GPU jobs')
 parent=results/'final_assessment';parent.mkdir(parents=True,exist_ok=True);root=Path(tempfile.mkdtemp(prefix='campaign_',dir=parent));(root/'logs').mkdir()
 shutil.copytree(here,root/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.zip','local_validation*'))
 plan=schedule()
 for j in plan['conversion']:j['source']=resolved[f"regularized_ln_seed{j['seed']}"]
 for j in plan['endpoints']:j['source']=resolved[j['name']]
 plan.update(data_root=str(data),repo=str(repo),historical_test_exposure=True,training=False,full_encrypted_backbone=False,bootstrapping=False)
 write(root/'manifest.json',plan);write(root/'source_hashes.json',{str(p.relative_to(root/'code')):sha(p) for p in sorted((root/'code').rglob('*')) if p.is_file()})
 write(root/'checkpoint_provenance.json',resolved)
 # Snapshot previous summaries; this does not interpret 72 recorded circuits as 72 successes.
 previous=[]
 for p in sorted((results/'next_phase').glob('campaign_*/summary/completion.json')):
  previous.append(dict(path=str(p),sha256=sha(p),completion=json.loads(p.read_text())))
 write(root/'previous_campaigns.json',previous)
 (results/'latest_final_assessment.txt').write_text(str(root)+'\n');return root
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',default=str(Path.home()/'Private-ViT-FHE'));p.add_argument('--results',default=str(Path.home()/'pvit_results'));a=p.parse_args();print(prepare(a.repo,a.results))
