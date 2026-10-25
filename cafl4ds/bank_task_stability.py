"""Bank-only alignment, directional coverage and logit-preserving gauge checks."""
from __future__ import annotations

import numpy as np

BANKS = ('monitor', 'audit', 'pooled')
RIDGES = (.001, .01, .1)
TOPK = (16, 32, 64)
DRAWS = tuple(range(491001, 491009))
QUANTILES = (.01, .1, .25, .5, .75, .9, .99)
BINS = (-np.inf, -2, -1, -.5, -.25, -.1, 0, .1, .25, .5, 1, 2, np.inf)


def sample_plan():
    result = []
    for bank in BANKS:
        for seed in DRAWS:
            rng = np.random.default_rng(seed)
            mon, audit = rng.permutation(256), 256+rng.permutation(256)
            sizes = (64,128) if bank != 'pooled' else (64,128,256)
            for n in sizes:
                ids = mon[:n] if bank=='monitor' else audit[:n] if bank=='audit' else np.r_[mon[:n//2],audit[:n//2]]
                result.append({'bank':bank,'n':n,'draw':seed,'indices':ids.tolist(),'full':False})
        ids = np.arange(256) if bank=='monitor' else np.arange(256,512) if bank=='audit' else np.arange(512)
        result.append({'bank':bank,'n':len(ids),'draw':0,'indices':ids.tolist(),'full':True})
    assert len(result)==59
    return result


def spectral(bank):
    z = np.asarray(bank, float); mean = z.mean(0); x = z-mean
    eig, v = np.linalg.eigh(x.T@x/len(x)); order = np.argsort(eig)[::-1]
    eig, v = eig[order],v[:,order]
    if eig[0]<=0 or eig.min() < -1e-10*eig[0]:
        raise ValueError('Invalid/collapsed covariance')
    eig = np.maximum(eig,0)
    return mean,eig,v


def coverage(bank, directions):
    """No labels/queries accepted: directions supplied by support or saved readers.

    Smooth coverage uses covariance eigenvalue/(eigenvalue+alpha*trace/d).
    Also report directional energy in fixed leading PC subspaces, truncated at
    numerical rank so arbitrary null-space eigenvectors do not count as coverage.
    """
    _,eig,v = spectral(bank); q = np.asarray(directions,float)
    energy = (q@v)**2; norm2 = np.sum(q*q,1)
    if np.any(norm2<=1e-24):
        raise ValueError('Nonzero task directions required')
    energy /= norm2[:,None]; tau = eig.mean()
    smooth = np.stack([energy@(eig/(eig+a*tau)) for a in RIDGES],axis=1)
    rank = int(np.sum(eig>eig[0]*1e-10))
    top = np.stack([energy[:,:min(k,rank)].sum(1) for k in TOPK],axis=1)
    bank_top = np.array([eig[:k].sum()/eig.sum() for k in TOPK])
    return np.column_stack([smooth,top]), {'numerical_rank':rank,'top_energy':bank_top.tolist(),'tau':float(tau)}


def contrast_directions(features,labels,domains,pools):
    """Class contrast and fixed-ridge within-class Fisher direction, SUPPORT only."""
    directions=[]; metadata=[]
    for p in range(2):
        for o in range(6):
            ids = pools[p,o]; z=np.asarray(features[ids],float); y=labels[ids]; d=domains[ids]
            for domain in (0,1):
                a,b = z[(d==domain)&(y==0)], z[(d==domain)&(y==1)]
                delta = b.mean(0)-a.mean(0)
                residual = np.r_[a-a.mean(0),b-b.mean(0)]
                cov=residual.T@residual/len(residual); tau=np.trace(cov)/z.shape[1]
                if tau<=1e-20:
                    raise ValueError('Collapsed support covariance')
                fisher=np.linalg.solve(cov+.01*tau*np.eye(z.shape[1]),delta)
                for kind,vector in [('contrast',delta),('fisher',fisher)]:
                    directions.append(vector); metadata.append({'population':p,'outer':o,'domain':domain,'kind':kind,'lambda':None})
    return np.stack(directions),metadata


def orthogonal_fit(before,after):
    ma,mb = before.mean(0),after.mean(0)
    u,_,vt=np.linalg.svd((after-mb).T@(before-ma),full_matrices=False)
    r=u@vt; t=ma-mb@r
    return r,t


def fit_error(before,after,r,t):
    return float(np.sum((after@r+t-before)**2)/np.sum((before-before.mean(0))**2))


def tail_gauge(rotation,bank_before,k,variant,seed):
    """Bank-only arbitrary completion challenge, NOT an encoder intervention."""
    _,_,v=spectral(bank_before); tail=rotation.shape[0]-k
    if not 0<tail<rotation.shape[0]:
        raise ValueError('Proper nonempty tail required')
    if variant=='flip':
        q=-np.eye(tail)
    elif variant=='rotate':
        q,rr=np.linalg.qr(np.random.default_rng(seed).normal(size=(tail,tail)))
        q=q@np.diag(np.where(np.diag(rr)<0,-1.,1.))
    else:
        raise ValueError(variant)
    block=np.eye(len(rotation)); block[k:,k:]=q
    return rotation@(v@block@v.T)


def accounting_from_means(before_means,after_means,head_before,head_after,r,t):
    """Class mean logits suffice for exact LINEAR mean-margin accounting."""
    a=np.asarray(before_means,float); b=np.asarray(after_means,float)@r+t
    ha,hb=np.asarray(head_before,float),np.asarray(head_after,float)
    transported=hb.copy(); transported[:,:-1]=hb[:,:-1]@r
    transported[:,-1]=hb[:,-1]-transported[:,:-1]@t
    feature=(b-a)@((ha[:,:-1]+transported[:,:-1])/2).T
    reader=((a+b)/2)@(transported[:,:-1]-ha[:,:-1]).T+transported[:,-1]-ha[:,-1]
    total=after_means@hb[:,:-1].T+hb[:,-1]-before_means@ha[:,:-1].T-ha[:,-1]
    np.testing.assert_allclose(feature+reader,total,atol=1e-10,rtol=0)
    return np.stack([total,feature,reader],axis=-1)


def margin_profile(before,after,label=1):
    before,after=map(np.asarray,(before,after))
    if before.shape!=after.shape or before.ndim!=1:
        raise ValueError('Paired individual margins required')
    ca=(before>0)|((before==0)&(label==0)); cb=(after>0)|((after==0)&(label==0))
    return {'before_quantiles':np.quantile(before,QUANTILES).tolist(),
            'after_quantiles':np.quantile(after,QUANTILES).tolist(),
            'before_histogram':np.histogram(before,BINS)[0].tolist(),
            'after_histogram':np.histogram(after,BINS)[0].tolist(),
            'before_near_zero':float(np.mean(abs(before)<=.1)),
            'after_near_zero':float(np.mean(abs(after)<=.1)),
            'wrong_to_right':int(np.sum(~ca&cb)),
            'right_to_wrong':int(np.sum(ca&~cb))}


def raw_rank(z):
    s=np.linalg.svd(z,compute_uv=False); p=s/s.sum()
    return float(np.exp(-np.sum(p[p>0]*np.log(p[p>0]))))


def controls(seed=194049):
    rng=np.random.default_rng(seed); bank=rng.normal(size=(512,96)); query=rng.normal(size=(128,96))
    r,_=np.linalg.qr(rng.normal(size=(96,96))); t=rng.normal(size=96)
    head=rng.normal(size=(1,97)); hb=head.copy(); hb[:,:-1]=head[:,:-1]@r; hb[:,-1]=head[:,-1]-hb[:,:-1]@t
    fitted,shift=orthogonal_fit(bank,bank@r+t)
    rotation=accounting_from_means(query,query@r+t,head,hb,fitted,shift)
    assert np.max(abs(rotation))<1e-10
    # Exact bank-null gauge: same encoders/readers, same bank fit, different attribution.
    hidden=bank.copy(); hidden[:,-2:]=0
    labels=np.tile([0,1],64); q=np.zeros((128,96)); q[:,-2:]=(2*labels[:,None]-1)*.02
    h=np.zeros((1,97)); h[0,-2]=50
    gauge=np.eye(96); gauge[-2:,-2:]=[[0,-1],[1,0]]
    null=accounting_from_means(q,q,h,h,gauge,np.zeros(96))
    np.testing.assert_allclose(hidden@gauge,hidden,atol=0,rtol=0)
    assert np.max(abs(null[:,:,0]))==0 and np.mean(abs(null[:,:,1]))>.9
    # Low-energy actual task deletion, held reader: near-zero rank change, 50pp loss.
    low=bank.copy(); low[:,:16]*=100; low[:,-2:]*=.001
    deleted=low.copy(); deleted[:,-2:]=0
    rank_change=(raw_rank(deleted)-raw_rank(low))/raw_rank(low)
    assert abs(rank_change)<.001
    accuracy_before=100*np.mean((q@h[0,:-1]>0)==labels)
    accuracy_after=100*np.mean(np.zeros(128,dtype=int)==labels)
    assert accuracy_before-accuracy_after==50
    return {'rotation_translation_max_term':float(np.max(abs(rotation))),
            'null_bank_fit_error':fit_error(hidden,hidden,gauge,np.zeros(96)),
            'null_gauge_mean_absolute_feature_term':float(np.mean(abs(null[:,:,1]))),
            'null_gauge_signed_feature_term':float(np.mean(null[:,:,1]*(2*labels[:,None]-1))),
            'null_gauge_signed_reader_term':float(np.mean(null[:,:,2]*(2*labels[:,None]-1))),
            'null_gauge_max_total_change':float(np.max(abs(null[:,:,0]))),
            'low_energy_relative_rank_change':rank_change,'low_energy_accuracy_loss_pp':accuracy_before-accuracy_after,
            'hidden_direction_coverage':float(coverage(hidden,h[:,:-1])[0][0,1]),
            'synthetic_only':True}
