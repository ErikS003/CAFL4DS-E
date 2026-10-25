import numpy as np

from cafl4ds.bank_task_stability import accounting_from_means, controls, coverage, margin_profile, sample_plan


def test_nested_balanced_bank_samples():
    plan=sample_plan(); assert len(plan)==59 and sum(p['full'] for p in plan)==3
    for bank in ('monitor','audit','pooled'):
        for seed in range(491001,491009):
            rows=[r for r in plan if r['bank']==bank and r['draw']==seed]
            for a,b in zip(rows,rows[1:]):
                assert set(a['indices'])<set(b['indices'])
            for row in rows:
                assert len(set(row['indices']))==row['n']
                if bank=='pooled':
                    assert sum(i<256 for i in row['indices'])==row['n']//2


def test_coverage_is_orthogonally_invariant_and_detects_unseen_task():
    rng=np.random.default_rng(1); z=rng.normal(size=(64,96)); z[:,-1]=0
    q=rng.normal(size=(3,96)); q[-1]=0; q[-1,-1]=1
    r,_=np.linalg.qr(rng.normal(size=(96,96)))
    a,_=coverage(z,q); b,_=coverage(z@r,q@r)
    np.testing.assert_allclose(a,b,atol=1e-8,rtol=0)
    assert max(a[-1,:3])<1e-10
    assert np.all((a>=-1e-12)&(a<=1+1e-12))


def test_class_means_accounting_matches_individual_average():
    rng=np.random.default_rng(2); a=rng.normal(size=(120,9)); b=rng.normal(size=(120,9))
    ha,hb=rng.normal(size=(5,10)),rng.normal(size=(5,10))
    r,_=np.linalg.qr(rng.normal(size=(9,9))); t=rng.normal(size=9)
    individual=accounting_from_means(a,b,ha,hb,r,t)
    means=accounting_from_means(a.reshape(4,30,9).mean(1),b.reshape(4,30,9).mean(1),ha,hb,r,t)
    np.testing.assert_allclose(means,individual.reshape(4,30,5,3).mean(1),atol=1e-12)


def test_synthetic_identification_and_low_energy_controls():
    result=controls()
    assert result['null_gauge_max_total_change']==0
    assert result['null_gauge_mean_absolute_feature_term']>.9
    assert result['null_gauge_signed_feature_term']<-.9 and result['null_gauge_signed_reader_term']>.9
    assert result['low_energy_accuracy_loss_pp']==50


def test_profile_crossings_reconcile_without_zero_ties():
    a=np.array([-2.,-.1,.2,1.]); b=np.array([1.,-.2,-.1,.5])
    v=margin_profile(a,b)
    assert v['wrong_to_right']==v['right_to_wrong']==1
    assert sum(v['before_histogram'])==sum(v['after_histogram'])==4
    ties=margin_profile(np.array([0.,-.1]),np.array([-.1,0.]),label=0)
    assert ties['wrong_to_right']==ties['right_to_wrong']==1
