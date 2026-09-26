import numpy as np
from protocol import LOGIT_ERROR,STATIC_BAD_RATE,MARGIN_ERROR_FRACTION
from metrics import classification

def outcomes(reference,converted,labels):
 r=np.asarray(reference,dtype=np.float64);p=np.asarray(converted,dtype=np.float64);y=np.asarray(labels)
 rf=np.isfinite(r).all(1);pf=np.isfinite(p).all(1);valid=rf&pf
 err=np.full(len(r),np.inf);err[valid]=np.abs(r[valid]-p[valid]).max(1)
 flip=np.zeros(len(r),dtype=bool);flip[valid]=r[valid].argmax(1)!=p[valid].argmax(1)
 cw=np.zeros(len(r),dtype=bool);cw[valid]=(r[valid].argmax(1)==y[valid])&(p[valid].argmax(1)!=y[valid])
 bad=rf&((~pf)|flip|(err>LOGIT_ERROR))
 return dict(reference_valid=rf,converted_valid=pf,conversion_failure=bad,prediction_flip=flip,correct_to_wrong=cw,max_logit_error=err)

def detectors(z32,z64,outside,static_review):
 p=np.asarray(z32,dtype=np.float64);q=np.asarray(z64,dtype=np.float64);f=np.isfinite(p).all(1)&np.isfinite(q).all(1)
 margin=np.zeros(len(p));delta=np.full(len(p),np.inf)
 sort=np.sort(q[f],axis=1);margin[f]=sort[:,-1]-sort[:,-2];delta[f]=np.abs(p[f]-q[f]).max(1)
 selfwarn=~f;selfwarn[f]|=(p[f].argmax(1)!=q[f].argmax(1))|(delta[f]>MARGIN_ERROR_FRACTION*np.maximum(margin[f],1e-8))
 outside=np.asarray(outside,dtype=bool)
 return dict(calibration_only=np.full(len(p),static_review,dtype=bool),finite_only=~np.isfinite(p).all(1),
   range=outside|~np.isfinite(p).all(1),range_and_precision=outside|selfwarn),delta,margin

def performance(flag,target,eligible):
 w=np.asarray(flag)[eligible];y=np.asarray(target)[eligible];n=len(y)
 tp=int((w&y).sum());fp=int((w&~y).sum());fn=int((~w&y).sum());tn=int((~w&~y).sum())
 return dict(n=n,tp=tp,fp=fp,fn=fn,tn=tn,recall=tp/(tp+fn) if tp+fn else None,
  false_positive_rate=fp/(fp+tn) if fp+tn else None,precision=tp/(tp+fp) if tp+fp else None,
  accepted_fraction=(tn+fn)/n if n else None,accepted_failure_rate=fn/(tn+fn) if tn+fn else None,
  all_reviewed=bool(n and tp+fp==n))

def summarize(ref,p,q,y,flags,static_review):
 target=outcomes(ref,p,y);det,delta,margin=detectors(p,q,flags,static_review)
 return dict(reference=classification(ref,y),converted=classification(p,y),converted_fp64=classification(q,y),
   target_definition='Reference finite AND (converted nonfinite OR argmax changed OR max absolute logit error > 0.1)',
   reference_invalid_excluded=int((~target['reference_valid']).sum()),
   detectors={k:performance(v,target['conversion_failure'],target['reference_valid']) for k,v in det.items()},
   conversion_failure_count=int(target['conversion_failure'].sum()),prediction_flips=int(target['prediction_flip'].sum()),
   correct_to_wrong=int(target['correct_to_wrong'].sum())),target,det,delta,margin
