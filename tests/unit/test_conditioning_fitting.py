import numpy as np

from cafl4ds.conditioning_fitting import (
    MAPS, augment, cell_metrics, conditional_geometry, feature_maps, finite_adam, newton, objective,
)


def panel(seed=4):
    rng=np.random.default_rng(seed)
    z=rng.normal(size=(80,8))*np.geomspace(.05,3,8)+rng.normal(size=8)*2
    y=(z[:,0]+.1*z[:,7]+rng.normal(size=80)*.1>np.median(z[:,0]+.1*z[:,7])).astype(float)
    return rng,z,y


def test_affine_objective_and_equivalent_prediction():
    rng,z,y=panel()
    x=augment(z)
    maps=feature_maps({'monitor':z[:40],'audit':z[40:]})
    beta=rng.normal(size=9)
    value,g,_=objective(beta,x,y,np.eye(9),.01)
    for t in maps.values():
        th=np.linalg.solve(t,beta)
        other,gm,_=objective(th,x@t,y,t.T@t,.01)
        np.testing.assert_allclose(value,other,atol=1e-10)
        np.testing.assert_allclose(gm,t.T@g,atol=1e-10)
        np.testing.assert_allclose((x@t)@th,x@beta,atol=1e-10)


def test_gradient_hessian_and_converged_invariance():
    rng,z,y=panel()
    x=augment(z)
    beta=rng.normal(size=9)*.1
    _,g,h=objective(beta,x,y,np.eye(9),.01)
    eps=1e-5
    numeric=np.array([(objective(beta+e,x,y,np.eye(9),.01)[0]-
                       objective(beta-e,x,y,np.eye(9),.01)[0])/(2*eps) for e in np.eye(9)*eps])
    np.testing.assert_allclose(g,numeric,atol=1e-8)
    assert np.linalg.eigvalsh(h).min()>=.01-1e-12
    raw,receipt=newton(x,y,np.eye(9),.01)
    assert receipt['gradient_inf']<1e-8
    for t in feature_maps({'monitor':z[:40],'audit':z[40:]}).values():
        th,_=newton(x@t,y,t.T@t,.01,raw_transport=t)
        np.testing.assert_allclose(x@raw,x@t@th,atol=1e-6)


def test_finite_adam_repeats_and_common_initial_function():
    _,z,y=panel()
    maps=feature_maps({'monitor':z[:40],'audit':z[40:]})
    x=np.stack([augment(z[:40]),augment(z[40:])])
    ys=np.stack([y[:40],y[40:]])
    a,trace,order=finite_adam(x,ys,maps,steps=10)
    b,_,other=finite_adam(x,ys,maps,steps=10)
    assert tuple(maps)==MAPS and order==other
    assert np.array_equal(a[10],b[10]) and np.all(a[0]==0)
    np.testing.assert_allclose(trace[0]['objective'],np.log(2))
    # Same accessible functions need not give the same finite coordinate updates.
    assert np.max(abs(a[10][:,0]-a[10][:,2]))>.001


def test_class_metrics_and_conditionals_keep_all_cells():
    rng=np.random.default_rng(6)
    labels=np.tile(np.repeat([0,1],20),2)
    domains=np.repeat([0,1],40)
    z=rng.normal(size=(80,8))+labels[:,None]*.2
    stats=conditional_geometry(z,labels,domains)
    assert len(stats)==2 and all(v['regularized_fisher']>=0 for v in stats)
    scores=(2*labels-1).astype(float)
    metrics=cell_metrics(scores,labels,domains)
    assert len(metrics)==7 and all(v['accuracy']==100 for v in metrics.values())


def test_native_penalty_changes_prior_and_bank_maps_exclude_queries():
    _,z,y=panel()
    x=augment(z)
    maps=feature_maps({'monitor':z[:40],'audit':z[40:]})
    t=maps['whiten_monitor']
    raw,_=newton(x,y,np.eye(9),.01)
    native,_=newton(x@t,y,np.eye(9),.01)
    assert np.max(abs(x@raw-x@t@native))>.01
    # Rank/centering operations use only supplied banks; query values are absent.
    changed=z.copy(); changed[40:]+=100
    other=feature_maps({'monitor':z[:40],'audit':changed[40:]})
    np.testing.assert_array_equal(maps['whiten_monitor'],other['whiten_monitor'])
