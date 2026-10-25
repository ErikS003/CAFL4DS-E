"""Disjoint population roles, nested label budgets and repeated-fold selection."""
from __future__ import annotations

import numpy as np

from cafl4ds.conditioning_fitting import augment
from cafl4ds.support_regularization import LAMBDAS, choose, macro_ce, solve, stratified_folds

POPULATIONS=('val','train')
BUDGETS=(32,64,128)
QUERIES=('val','train_matched','train_full')
DRAWS=(61001,61007,61013,61019,61027,61031)


def make_roles(labels,domains,partitions,seed=194047):
    labels,domains,partitions=map(np.asarray,(labels,domains,partitions))
    if not labels.shape==domains.shape==partitions.shape:
        raise ValueError('Aligned population metadata required')
    rng=np.random.default_rng(seed)
    pools=[]; tests=[]
    for population in POPULATIONS:
        pool=[]; query=[]
        for domain in (0,1):
            for label in (0,1):
                ids=np.where((partitions==population)&(domains==domain)&(labels==label))[0]
                if len(ids)<50:
                    raise ValueError('At least50 images per source cell required')
                shuffled=rng.permutation(ids)
                pool.extend(shuffled[:32]); query.append(shuffled[32:])
        pools.append(np.array(pool,dtype=np.int64))
        tests.append(query)
    val=np.concatenate(tests[0]); train=np.concatenate(tests[1])
    matched=np.concatenate([q[:18] for q in tests[1]])
    role={'pools':np.stack(pools),'query_val':val,'query_train_matched':matched,'query_train_full':train}
    assert len(set(role['pools'].ravel())&set(np.r_[val,train]))==0
    assert len(set(val)&set(train))==0 and set(matched)<=set(train)
    return role


def nested_draws(labels,domains,draw_seed):
    labels,domains=np.asarray(labels),np.asarray(domains)
    if len(labels)!=128:
        raise ValueError('128-image source pool required')
    rng=np.random.default_rng(194047+int(draw_seed))
    cells=[rng.permutation(np.where((domains==d)&(labels==c))[0]) for d in (0,1) for c in (0,1)]
    if any(len(c)!=32 for c in cells):
        raise ValueError('32 examples in every pool cell required')
    return {n:np.sort(np.concatenate([c[:n//4] for c in cells])) for n in BUDGETS}


def condition(support,labels,domains,draw_seed,fit,full=None):
    """Two CV partitions on the SAME support set; no query argument or access."""
    support=np.asarray(support,dtype=np.float64)
    labels,domains=np.asarray(labels),np.asarray(domains)
    x=augment(support); n,d=x.shape
    folds=np.stack([stratified_folds(labels,domains,194047+17*int(draw_seed)+r,folds=4) for r in range(2)])
    cv_heads=np.empty((2,4,4,d)); oof=np.empty((2,4,n)); losses=np.empty((2,4))
    choices=np.empty(2,dtype=np.int64)
    for r in range(2):
        for k,lam in enumerate(LAMBDAS):
            for f in range(4):
                train=folds[r]!=f; valid=~train
                beta,_=fit(x[train],labels[train],lam,key=('cv',r,k,f))
                cv_heads[r,k,f]=beta; oof[r,k,valid]=x[valid]@beta
            losses[r,k]=macro_ce(oof[r,k],labels,domains)
        choices[r]=choose(losses[r])
    if full is None:
        full=np.stack([fit(x,labels,lam,key=('full',k))[0] for k,lam in enumerate(LAMBDAS)])
    return {'folds':folds,'cv_heads':cv_heads,'oof':oof,'cv_loss':losses,'selected':choices,'full_heads':full}


def checked_fit(x,y,lam,**_):
    return solve(x,y,lam)
