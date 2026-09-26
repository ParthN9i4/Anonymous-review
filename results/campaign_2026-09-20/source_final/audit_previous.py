"""Read completed next_phase outputs without rerunning training or circuits."""
import argparse,json,shutil
from pathlib import Path
from io_utils import write,sha

def audit(results,destination):
 results=Path(results);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);campaigns=[]
 for file in sorted((results/'next_phase').glob('campaign_*/manifest.json')):
  m=json.loads(file.read_text());jobs=m.get('ckks',[])+m.get('new_ckks',[])
  if not jobs:continue
  entries=[];missing=[];counts={};failed_tolerance=0
  target=destination/file.parent.name;target.mkdir(exist_ok=True)
  for j in jobs:
   p=Path(j['out'])/'report.json'
   if not p.is_file():missing.append(j['name']);continue
   d=json.loads(p.read_text());name=j['name'];write(target/(name+'.json'),d)
   entries.append(dict(name=name,report_sha256=sha(p),observed_failure=d.get('observed_failure'),server_has_secret_key=d.get('server_has_secret_key')))
   for r in d.get('rows',[]):
    if r.get('warmup'):continue
    k=r.get('status','UNSPECIFIED');counts[k]=counts.get(k,0)+1
    failed_tolerance+=int(bool(r.get('fails_frozen_tolerance',False)))
  summaries=target/'summary';summaries.mkdir(exist_ok=True)
  for p in (file.parent/'summary').glob('*'):
   if p.is_file() and p.suffix in ('.json','.csv','.md'):shutil.copy2(p,summaries/p.name)
  campaigns.append(dict(campaign=file.parent.name,planned_reports=len(jobs),recorded_reports=len(entries),missing=missing,trial_statuses_excluding_warmup=counts,trials_failing_recorded_tolerance=failed_tolerance,report_provenance=entries))
 write(destination/'AUDIT.json',campaigns)
 for d in campaigns:print({k:v for k,v in d.items() if k!='report_provenance'})
 return campaigns
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('destination');a=p.parse_args();audit(a.results,a.destination)
