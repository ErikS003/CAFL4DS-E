import numpy as np

from cafl4ds.bank_task_stability import sample_plan
from cafl4ds.center_rotation_control import (bank_sufficient,class_axes,closed_bank_cost,controls,
                                           crossed_plan,finite_grid_variance,symmetric_gap)


def test_complete_cross_and_original_maps():
    maps,blocks=crossed_plan(sample_plan())
    assert len(maps)==1225 and len(blocks)==19
    assert len({(m['rotation_index'],m['center_index']) for m in maps})==1225
    assert sum(m['rotation_index']==m['center_index'] for m in maps)==59
    for block in blocks:
        rows=maps[block['start']:block['stop']]
        assert len(rows)==64 and len({r['rotation_draw'] for r in rows})==len({r['center_draw'] for r in rows})==8


def test_closed_cost_independent_of_direct_residual_computation():
    rng=np.random.default_rng(1); a,b=rng.normal(size=(73,11)),rng.normal(size=(73,11))
    r,_=np.linalg.qr(rng.normal(size=(11,11))); t=rng.normal(size=11)
    direct=np.sum((b@r+t-a)**2)/np.sum((a-a.mean(0))**2)
    np.testing.assert_allclose(closed_bank_cost(r,t,bank_sufficient(a,b)),direct,atol=1e-12)


def test_center_invariance_control_and_preserved_logits():
    result=controls(); assert result['maximum_separation_center_change']<1e-12
    assert result['maximum_total_margin_center_change']==0 and result['preference_shift_magnitude']>.1


def test_variance_exactly_separates_crossed_factors():
    a=np.arange(8,dtype=float)-3.5; b=a/2; x=a[:,None]+b[None,:]+.1*a[:,None]*b[None,:]
    total,orientation,center,interaction=finite_grid_variance(x)
    np.testing.assert_allclose([orientation,center,interaction],[np.var(a),np.var(b),.01*np.var(a)*np.var(b)],atol=1e-12)
    np.testing.assert_allclose(total,orientation+center+interaction,atol=1e-12)


def test_symmetric_gap_matches_two_paths_and_class_axes():
    rng=np.random.default_rng(2); x=rng.normal(size=(3,3,7))
    gap,center,rotation,_=np.moveaxis(symmetric_gap(x,0,1),-1,0)
    np.testing.assert_allclose(gap,center+rotation,atol=1e-12)
    cells=np.array([-2.,2.,-3.,3.]); axes=class_axes(cells)
    np.testing.assert_array_equal(axes[4:],[0,2,0,3])
