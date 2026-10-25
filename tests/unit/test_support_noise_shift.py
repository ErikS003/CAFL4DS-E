import inspect

import numpy as np

from cafl4ds.conditioning_fitting import augment
from cafl4ds.support_noise_shift import BUDGETS, checked_fit, condition, make_roles, nested_draws
from cafl4ds.support_regularization import choose, macro_ce


def population():
    y=np.tile(np.repeat([0,1],50),2)
    d=np.repeat([0,1],100)
    return np.tile(y,2),np.tile(d,2),np.repeat(['val','train'],200)


def test_roles_are_disjoint_and_randomization_is_metadata_only():
    y,d,p=population()
    role=make_roles(y,d,p)
    other=make_roles(y,d,p)
    for k in role:
        np.testing.assert_array_equal(role[k],other[k])
    assert role['pools'].shape==(2,128)
    assert len(role['query_val'])==len(role['query_train_full'])==72
    assert set(role['pools'].ravel()).isdisjoint(set(role['query_val'])|set(role['query_train_full']))


def test_label_budgets_are_nested_and_largest_has_no_sampling_variation():
    y=np.tile(np.repeat([0,1],32),2); d=np.repeat([0,1],64)
    a=nested_draws(y,d,61001); b=nested_draws(y,d,61007)
    assert set(a[32])<set(a[64])<set(a[128])
    np.testing.assert_array_equal(a[128],b[128])
    assert not np.array_equal(a[32],b[32])
    for n in BUDGETS:
        assert all(np.sum((y[a[n]]==c)&(d[a[n]]==v))==n//4 for c in (0,1) for v in (0,1))


def test_repeated_cv_never_receives_heldout_rows_and_choices_reconstruct():
    rng=np.random.default_rng(47)
    y=np.tile(np.repeat([0,1],8),2); d=np.repeat([0,1],16)
    z=rng.normal(size=(32,6)); calls=[]
    def fitting(x,labels,lam,key):
        calls.append(key)
        return checked_fit(x,labels,lam)
    result=condition(z,y,d,61001,fitting)
    assert len(calls)==36 and not any('query' in p for p in inspect.signature(condition).parameters)
    x=augment(z)
    for r in range(2):
        scores=np.empty((4,32))
        for k in range(4):
            for f in range(4):
                keep=result['folds'][r]==f
                scores[k,keep]=x[keep]@result['cv_heads'][r,k,f]
        np.testing.assert_array_equal(scores,result['oof'][r])
        assert choose([macro_ce(s,y,d) for s in scores])==result['selected'][r]
    # Reusing a full fit changes neither the CV problem nor selected index.
    reused=condition(z,y,d,61001,fitting,full=result['full_heads'])
    assert len(calls)==68
    np.testing.assert_array_equal(result['selected'],reused['selected'])


def test_population_interaction_is_a_difference_of_risks_not_an_accuracy_oracle():
    risk=np.array([[.4,.8],[.7,.5]]) # support population x test population
    interaction=(risk[0,1]-risk[1,1])-(risk[0,0]-risk[1,0])
    np.testing.assert_allclose(interaction,.6)
