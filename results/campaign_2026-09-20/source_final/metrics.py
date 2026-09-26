"""NumPy-only metrics. Invalid predictions are failures, never class-zero guesses."""
import numpy as np


def classification(logits, labels, bins=15):
    z=np.asarray(logits,dtype=np.float64); y=np.asarray(labels,dtype=np.int64)
    if z.ndim!=2 or len(z)!=len(y) or not len(y): raise ValueError('Invalid prediction shape')
    n,k=z.shape
    if np.any((y<0)|(y>=k)): raise ValueError('Label outside class range')
    finite=np.isfinite(z).all(1); pred=np.full(n,-1,dtype=int)
    pred[finite]=z[finite].argmax(1)
    cm=np.zeros((k,k+1),dtype=int)
    np.add.at(cm,(y,np.where(finite,pred,k)),1)
    support=cm.sum(1); tp=np.diag(cm[:,:k]); predicted=cm[:,:k].sum(0)
    recall=np.divide(tp,support,out=np.zeros(k,dtype=float),where=support>0)
    precision=np.divide(tp,predicted,out=np.zeros(k,dtype=float),where=predicted>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(k,dtype=float),where=(support+predicted)>0)
    out=dict(n=n,finite_images=int(finite.sum()),nonfinite_images=int((~finite).sum()),
        finite_coverage=float(finite.mean()),accuracy_invalid_as_failure=100*float((pred==y).mean()),
        legacy_argmax_accuracy_diagnostic=100*float((z.argmax(1)==y).mean()),
        invalid_argmax_matches_diagnostic=int(((z.argmax(1)==y)&~finite).sum()),
        balanced_accuracy_present_classes=100*float(recall[support>0].mean()),
        macro_f1_present_classes=float(f1[support>0].mean()),
        per_class_support=support.tolist(),per_class_recall=recall.tolist(),
        per_class_precision=precision.tolist(),confusion_with_invalid_column=cm.tolist(),
        nll_finite=None,nll_max_finite=None,nll_p99_finite=None,nll_p999_finite=None,finite_nll_gt100=0,brier_finite=None,ece15_finite=None,reliability_bins=[],
        probabilistic_metric_scope='Conditional on finite logits; always report finite coverage alongside these metrics')
    if not finite.any(): return out
    f=z[finite]; yf=y[finite]; shifted=f-f.max(1,keepdims=True)
    ex=np.exp(shifted); p=ex/ex.sum(1,keepdims=True)
    logp=shifted-np.log(ex.sum(1,keepdims=True))
    nll=-logp[np.arange(len(yf)),yf]
    out['nll_finite']=float(nll.mean());out['nll_max_finite']=float(nll.max())
    out['nll_p99_finite']=float(np.quantile(nll,.99));out['nll_p999_finite']=float(np.quantile(nll,.999));out['finite_nll_gt100']=int((nll>100).sum())
    out['finite_prediction_class_counts']=np.bincount(p.argmax(1),minlength=k).tolist()
    ordered=np.sort(f,axis=1);out['finite_top2_margin_quantiles']=np.quantile(ordered[:,-1]-ordered[:,-2],[0,.01,.5,.99,1]).tolist()
    onehot=np.eye(k)[yf];out['brier_finite']=float(((p-onehot)**2).sum(1).mean())
    conf=p.max(1);correct=p.argmax(1)==yf;index=np.minimum((conf*bins).astype(int),bins-1)
    ece=0.
    for b in range(bins):
        hit=index==b;count=int(hit.sum());c=float(conf[hit].mean()) if count else None;a=float(correct[hit].mean()) if count else None
        if count: ece+=count/len(yf)*abs(c-a)
        out['reliability_bins'].append(dict(lower=b/bins,upper=(b+1)/bins,n=count,confidence=c,accuracy=a))
    out['ece15_finite']=ece
    return out


def compare(a,b):
    for key in ('ids','labels'):
        if not np.array_equal(a[key],b[key]):raise ValueError('Comparison example alignment mismatch')
    x=np.asarray(a['logits'],dtype=np.float64);y=np.asarray(b['logits'],dtype=np.float64)
    fa=np.isfinite(x).all(1);fb=np.isfinite(y).all(1);both=fa&fb
    d=np.abs(x[both]-y[both]);pa=x[both].argmax(1);pb=y[both].argmax(1);labels=a['labels'][both]
    err=d.max(1) if len(d) else np.array([])
    sx=np.sort(x[both],axis=1);margin=sx[:,-1]-sx[:,-2]
    return dict(n=len(x),both_finite=int(both.sum()),both_invalid=int((~fa&~fb).sum()),
        reference_invalid_only=int((~fa&fb).sum()),other_invalid_only=int((fa&~fb).sum()),
        finite_prediction_disagreements=int((pa!=pb).sum()),
        finite_correct_to_wrong=int(((pa==labels)&(pb!=labels)).sum()),
        finite_wrong_to_correct=int(((pa!=labels)&(pb==labels)).sum()),
        max_logit_error=float(d.max()) if d.size else None,
        mean_logit_error=float(d.mean()) if d.size else None,
        p99_max_logit_error=float(np.quantile(err,.99)) if len(err) else None,
        sufficient_margin_condition_count=int((2*err<margin).sum()),
        margin_condition_violations=int(((2*err<margin)&(pa!=pb)).sum()))


def collapse(values, threshold=12., consecutive=5, learned_threshold=30.):
    v=np.asarray(values,dtype=float); near=v<=threshold; streak=0; first=None; post=None; learned=False;maxstreak=0
    for i,flag in enumerate(near):
        learned=learned or v[i]>learned_threshold
        streak=streak+1 if flag else 0;maxstreak=max(maxstreak,streak)
        if streak>=consecutive:
            if first is None:first=i+2-consecutive
            if learned and post is None:post=i+2-consecutive
    return dict(epochs=len(v),epochs_near_chance=int(near.sum()),longest_near_chance_streak=maxstreak,
        first_near_chance_streak=first,post_learning_near_chance_start=post,
        best_minus_final_pp=float(v.max()-v[-1]),final_near_chance=bool(near[-1]),
        criterion=dict(threshold_percent=threshold,consecutive_epochs=consecutive,learned_threshold_percent=learned_threshold),
        scope='Retrospective descriptive criterion, not a preregistered predictor; early failure to learn is distinct from post-learning collapse')
