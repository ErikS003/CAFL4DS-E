"""Cross bank rotations and centers without changing encoders or reader logits."""
from __future__ import annotations

import numpy as np

BANKS=('monitor','audit','pooled')
AXES=('day_city','day_highway','night_city','night_highway','day_separation','day_preference','night_separation','night_preference')


def crossed_plan(samples):
    """Metadata-only design:9 full maps +1152 two-size maps +64 pooled256 maps."""
    full={b:next(i for i,s in enumerate(samples) if s['bank']==b and s['full']) for b in BANKS}
    maps=[]; blocks=[]
    def add(ri,ci,family,n,rb,cb,rd,cd):
        maps.append({'rotation_index':ri,'center_index':ci,'family':family,'n':n,
                     'rotation_bank':rb,'center_bank':cb,'rotation_draw':rd,'center_draw':cd})
    for rb in BANKS:
        for cb in BANKS:
            add(full[rb],full[cb],'full',0,rb,cb,0,0)
    for n in (64,128):
        for rb in BANKS:
            for cb in BANKS:
                start=len(maps)
                rotations=[i for i,s in enumerate(samples) if s['bank']==rb and s['n']==n and not s['full']]
                centers=[i for i,s in enumerate(samples) if s['bank']==cb and s['n']==n and not s['full']]
                assert len(rotations)==len(centers)==8
                for ri in rotations:
                    for ci in centers:
                        add(ri,ci,f'n{n}',n,rb,cb,samples[ri]['draw'],samples[ci]['draw'])
                blocks.append({'family':f'n{n}','n':n,'rotation_bank':rb,'center_bank':cb,'start':start,'stop':len(maps)})
    ids=[i for i,s in enumerate(samples) if s['bank']=='pooled' and s['n']==256 and not s['full']]
    assert len(ids)==8; start=len(maps)
    for ri in ids:
        for ci in ids:
            add(ri,ci,'pooled256',256,'pooled','pooled',samples[ri]['draw'],samples[ci]['draw'])
    blocks.append({'family':'pooled256','n':256,'rotation_bank':'pooled','center_bank':'pooled','start':start,'stop':len(maps)})
    assert len(maps)==1225 and len(blocks)==19
    assert len([m for m in maps if m['rotation_index']==m['center_index']])==59
    return maps,blocks


def center_translation(rotation,before_center,after_center):
    return np.asarray(before_center)-np.asarray(after_center)@rotation


def bank_sufficient(before,after):
    ma,mb=before.mean(0),after.mean(0); a,b=before-ma,after-mb
    return ma,mb,float(np.sum(a*a)/len(a)),float(np.sum(b*b)/len(b)),b.T@a/len(a)


def closed_bank_cost(rotation,translation,summary):
    ma,mb,va,vb,cross=summary
    shift=mb@rotation+translation-ma
    return float((va+vb-2*np.sum(rotation*cross)+shift@shift)/va)


def class_axes(cells):
    """Last axis: four signed cell margins. Separation cancels translations."""
    x=np.asarray(cells)
    if x.shape[-1]!=4:
        raise ValueError('Four ordered day/night city/highway cells required')
    return np.concatenate([x,np.stack([(x[...,0]+x[...,1])/2,(x[...,1]-x[...,0])/2,
                                       (x[...,2]+x[...,3])/2,(x[...,3]-x[...,2])/2],axis=-1)],axis=-1)


def finite_grid_variance(grid):
    """Exact balanced finite-grid ANOVA; descriptive, not a population variance fit."""
    x=np.asarray(grid,float)
    if x.shape[:2]!=(8,8):
        raise ValueError('Complete8 rotation x8 centering grid required')
    mean=x.mean((0,1)); row=x.mean(1)-mean; col=x.mean(0)-mean
    residual=x-mean-row[:,None]-col[None,:]
    total=np.mean((x-mean)**2,axis=(0,1))
    ori=np.mean(row**2,axis=0); cen=np.mean(col**2,axis=0); interaction=np.mean(residual**2,axis=(0,1))
    np.testing.assert_allclose(total,ori+cen+interaction,atol=1e-12,rtol=1e-9)
    return np.stack([total,ori,cen,interaction],axis=-1)


def symmetric_gap(full,a,b):
    """Two-path symmetric CENTER and ROTATION accounting of a diagonal bank gap."""
    x=np.asarray(full)
    center=((x[a,b]-x[a,a])+(x[b,b]-x[b,a]))/2
    rotation=((x[b,a]-x[a,a])+(x[b,b]-x[a,b]))/2
    gap=x[b,b]-x[a,a]
    interaction=x[b,b]-x[b,a]-x[a,b]+x[a,a]
    np.testing.assert_allclose(center+rotation,gap,atol=1e-10,rtol=0)
    return np.stack([gap,center,rotation,interaction],axis=-1)


def controls(seed=194050):
    from cafl4ds.bank_task_stability import accounting_from_means
    rng=np.random.default_rng(seed); a=rng.normal(size=(8,12)); b=rng.normal(size=(8,12))
    ha,hb=rng.normal(size=(3,13)),rng.normal(size=(3,13)); r,_=np.linalg.qr(rng.normal(size=(12,12)))
    t=rng.normal(size=12); delta=rng.normal(size=12)
    base=accounting_from_means(a,b,ha,hb,r,t); changed=accounting_from_means(a,b,ha,hb,r,t+delta)
    expected=delta@((ha[:,:-1]+hb[:,:-1]@r)/2).T
    np.testing.assert_allclose(changed[:,:,1]-base[:,:,1],np.broadcast_to(expected,(8,3)),atol=1e-12)
    sign=np.tile([-1,1,-1,1],2)
    signed=(changed[:,:,1]-base[:,:,1])*sign[:,None]
    axes=class_axes(signed.reshape(2,4,3).transpose(0,2,1))
    assert np.max(abs(axes[:,:,[4,6]]))<1e-12
    assert np.max(abs(changed[:,:,0]-base[:,:,0]))==0
    # A centered, deliberately crossed grid isolates both factors and interaction.
    u=np.arange(8,dtype=float)-3.5; v=u/2
    grid=u[:,None]+v[None,:]+.1*u[:,None]*v[None,:]
    variance=finite_grid_variance(grid)
    return {'synthetic_only':True,'maximum_center_shift_identity_error':float(np.max(abs(signed-sign[:,None]*expected))),
            'maximum_separation_center_change':float(np.max(abs(axes[:,:,[4,6]]))),
            'maximum_total_margin_center_change':float(np.max(abs(changed[:,:,0]-base[:,:,0]))),
            'preference_shift_magnitude':float(np.mean(abs(axes[:,:,[5,7]]))),
            'finite_grid_variance':variance.tolist()}
