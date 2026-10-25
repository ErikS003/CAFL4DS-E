import inspect

import numpy as np

from cafl4ds.support_regularization import (
    LAMBDAS, choose, macro_ce, stratified_folds, support_selection,
)


def panel():
    rng=np.random.default_rng(46)
    labels=np.tile(np.repeat([0,1],12),2)
    domains=np.repeat([0,1],24)
    z=rng.normal(size=(48,6))*np.geomspace(.02,1,6)+labels[:,None]*.2
    banks={'monitor':rng.normal(size=(64,6)), 'audit':rng.normal(size=(64,6))+.1}
    return z,labels,domains,banks


def test_folds_are_deterministic_stratified_and_cover_support():
    _,y,d,_=panel()
    a=stratified_folds(y,d,194046)
    np.testing.assert_array_equal(a,stratified_folds(y,d,194046))
    for f in range(5):
        for domain in (0,1):
            for label in (0,1):
                assert np.sum((a==f)&(d==domain)&(y==label))>=2
    assert set(a)==set(range(5))


def test_tie_prefers_stronger_prior_and_macro_cells_are_equal():
    assert choose([1,1,1,1])==3
    assert choose([1,1,1,1.1])==2
    assert choose([.1,.2,.3,.4])==0
    y=np.array([0,1,0,1,1]); d=np.array([0,0,1,1,1])
    s=np.array([-1,2,-3,4,4])
    expected=np.mean(np.logaddexp(0,-np.array([1,2,3,4])))
    np.testing.assert_allclose(macro_ce(s,y,d),expected)


def test_selector_is_query_free_and_oof_heads_reconstruct_selection():
    z,y,d,banks=panel()
    calls=[]
    from cafl4ds.support_regularization import solve
    def tracked(x,labels,lam,key):
        calls.append((len(x),key))
        return solve(x,labels,lam)
    result=support_selection(z,y,d,banks,61001,fit=tracked)
    assert len(calls)==72
    assert len(inspect.signature(support_selection).parameters)==6
    assert not any('query' in name for name in inspect.signature(support_selection).parameters)
    x=np.column_stack((z,np.ones(len(z))))
    for m in range(3):
        reconstructed=np.empty((4,len(z)))
        for k in range(4):
            for f in range(5):
                keep=result['fold']==f
                reconstructed[k,keep]=x[keep]@result['cv_heads'][m,k,f]
        np.testing.assert_array_equal(result['oof'][m],reconstructed)
        losses=[macro_ce(s,y,d) for s in reconstructed]
        np.testing.assert_allclose(losses,result['cv_loss'][m],atol=1e-14)
        assert result['selected'][m]==choose(losses)
    assert len(LAMBDAS)==4


def test_validation_label_is_not_used_to_fit_its_oof_head():
    # Fit call spy verifies each fold's input has exactly the complementary rows.
    z,y,d,banks=panel()
    folds=stratified_folds(y,d,194046+61001)
    from cafl4ds.conditioning_fitting import augment, feature_maps
    from cafl4ds.support_regularization import solve
    transforms=feature_maps(banks)
    def tracked(x,labels,lam,key):
        name,_,f=key
        keep=np.ones(len(z),bool) if f==5 else folds!=f
        np.testing.assert_allclose(x,(augment(z)@transforms[name])[keep])
        np.testing.assert_array_equal(labels,y[keep])
        return solve(x,labels,lam)
    support_selection(z,y,d,banks,61001,fit=tracked)
