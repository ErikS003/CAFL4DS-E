"""Support-only stratified cross-validation for fixed native-coordinate L2 priors."""
from __future__ import annotations

import numpy as np

from cafl4ds.conditioning_fitting import augment, feature_maps, newton

FAMILIES=('raw','whiten_monitor','whiten_audit')
LAMBDAS=(0.0001,0.001,0.01,0.1)


def stratified_folds(labels,domains,seed,folds=5):
    labels,domains=np.asarray(labels),np.asarray(domains)
    if labels.shape!=domains.shape or labels.ndim!=1:
        raise ValueError('Aligned support labels/domains required')
    rng=np.random.default_rng(seed)
    result=np.full(len(labels),-1,dtype=np.int64)
    for domain in (0,1):
        for label in (0,1):
            ids=np.where((domains==domain)&(labels==label))[0]
            if len(ids)<2*folds:
                raise ValueError('At least two held-out examples per cell/fold required')
            for i,subset in enumerate(np.array_split(rng.permutation(ids),folds)):
                result[subset]=i
    assert np.all(result>=0)
    return result


def macro_ce(scores,labels,domains):
    labels,domains=np.asarray(labels),np.asarray(domains)
    loss=np.logaddexp(0,-(2*labels-1)*np.asarray(scores))
    cells=[loss[(labels==c)&(domains==d)] for d in (0,1) for c in (0,1)]
    if any(len(v)==0 for v in cells):
        raise ValueError('All four support cells are required')
    return float(np.mean([v.mean() for v in cells]))


def choose(losses,lambdas=LAMBDAS,tolerance=1e-12):
    a=np.asarray(losses,dtype=np.float64)
    if a.shape!=(len(lambdas),) or not np.isfinite(a).all():
        raise ValueError('Finite aligned validation losses required')
    candidates=np.where(a<=a.min()+tolerance)[0]
    return int(max(candidates,key=lambda i:lambdas[i]))


def solve(x,y,lam):
    return newton(x,y,np.eye(x.shape[1]),lam,tolerance=1e-10,max_steps=100)


def support_selection(support,labels,domains,banks,draw_seed,fit=solve):
    """No API argument accepts query features, query labels or old outcomes.

    Callback permits explicit per-fit accounting; every fit starts at zero.
    All full candidates are fit for the declared fixed comparisons, not selected
    using their training loss. Return OOF predictions and all candidate heads.
    """
    support=np.asarray(support,dtype=np.float64)
    labels,domains=np.asarray(labels),np.asarray(domains)
    if len(support)!=len(labels) or labels.ndim!=1:
        raise ValueError('Aligned support required')
    fold=stratified_folds(labels,domains,194046+int(draw_seed))
    allmaps=feature_maps(banks,ridge=0.01)
    maps={name:allmaps[name] for name in FAMILIES}
    x=augment(support)
    d=x.shape[1]
    oof=np.empty((3,4,len(x)))
    cv_heads=np.empty((3,4,5,d)); full_heads=np.empty((3,4,d))
    losses=np.empty((3,4)); selected=np.empty(3,dtype=np.int64)
    receipts=[]
    for m,(name,t) in enumerate(maps.items()):
        xm=x@t
        for k,lam in enumerate(LAMBDAS):
            for f in range(5):
                train=fold!=f; valid=~train
                theta,receipt=fit(xm[train],labels[train],lam,
                                  key=(name,k,f)) if fit is not solve else solve(xm[train],labels[train],lam)
                cv_heads[m,k,f]=t@theta
                oof[m,k,valid]=x[valid]@cv_heads[m,k,f]
                receipts.append({'family':name,'lambda_index':k,'fold':f,**receipt})
            theta,receipt=fit(xm,labels,lam,key=(name,k,5)) if fit is not solve else solve(xm,labels,lam)
            full_heads[m,k]=t@theta
            receipts.append({'family':name,'lambda_index':k,'fold':5,**receipt})
            losses[m,k]=macro_ce(oof[m,k],labels,domains)
        selected[m]=choose(losses[m])
    # Fold-deletion sensitivity uses saved OOF errors only, no additional fits.
    deletion=np.empty((3,5),dtype=np.int64)
    for m in range(3):
        for f in range(5):
            keep=fold!=f
            deletion[m,f]=choose([macro_ce(s[keep],labels[keep],domains[keep]) for s in oof[m]])
    return {'fold':fold,'oof':oof,'cv_heads':cv_heads,'full_heads':full_heads,
            'cv_loss':losses,'selected':selected,'deletion_selected':deletion,
            'transforms':np.array(list(maps.values())),'receipts':receipts}
