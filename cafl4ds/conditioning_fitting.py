"""Affine-equivalent logistic objectives, finite optimization and class diagnostics."""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from scipy.special import expit

MAPS = ('raw', 'center_monitor', 'scale_monitor', 'whiten_monitor',
        'center_audit', 'scale_audit', 'whiten_audit')
LANDMARKS = (0, 10, 100, 1000)
COMPONENTS = ('balanced', 'day', 'night', 'day_city', 'day_highway', 'night_city', 'night_highway')


def augment(z):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError('Finite feature matrix required')
    return np.column_stack((z, np.ones(len(z))))


def feature_maps(banks, ridge=0.01):
    """x_m = x_aug T; a mapped head theta represents raw head beta=T theta."""
    d = banks['monitor'].shape[1]
    result = {'raw':np.eye(d+1)}
    for bank in ('monitor', 'audit'):
        z = np.asarray(banks[bank], dtype=np.float64)
        mean = z.mean(0)
        x = z-mean
        covariance = x.T@x/len(x)
        eig, v = np.linalg.eigh(covariance)
        rms2 = np.trace(covariance)/d
        if rms2 <= 1e-20 or eig.min() < -1e-10*max(1, eig.max()):
            raise ValueError('Invalid/collapsed bank covariance')
        operators = {'center':np.eye(d), 'scale':np.eye(d)/np.sqrt(rms2),
                     'whiten':(v/np.sqrt(np.maximum(eig, 0)+ridge*rms2))@v.T}
        for name, a in operators.items():
            t = np.eye(d+1)
            t[:d,:d] = a
            t[d,:d] = -mean@a
            assert np.isfinite(t).all() and np.linalg.matrix_rank(t) == d+1
            result[f'{name}_{bank}'] = t
    return {name:result[name] for name in MAPS}


def objective(theta, x, y, penalty, lam):
    score = x@theta
    p = expit(score)
    loss = np.mean(np.logaddexp(0, score)-y*score)+lam*(theta@penalty@theta)/2
    gradient = x.T@(p-y)/len(y)+lam*penalty@theta
    hessian = (x.T*(p*(1-p)))@x/len(y)+lam*penalty
    return float(loss), gradient, hessian


def newton(x, y, penalty, lam, raw_transport=None, tolerance=1e-8, max_steps=100):
    """Deterministic damped Newton; stopping uses raw gradient for common objective."""
    theta = np.zeros(x.shape[1])
    for step in range(max_steps+1):
        value, g, h = objective(theta, x, y, penalty, lam)
        residual = g if raw_transport is None else np.linalg.solve(raw_transport.T, g)
        norm = float(np.max(abs(residual)))
        if norm <= tolerance:
            return theta, {'iterations':step, 'objective':value, 'gradient_inf':norm}
        if step == max_steps:
            raise RuntimeError(f'Newton did not converge: {norm}')
        direction = np.linalg.solve(h, g)
        descent = float(g@direction)
        if not descent > 0:
            raise RuntimeError('Newton direction is not descent')
        rate = 1.0
        for _ in range(60):
            candidate = theta-rate*direction
            trial = objective(candidate, x, y, penalty, lam)[0]
            if trial <= value-1e-4*rate*descent+1e-14:
                theta = candidate
                break
            rate *= 0.5
        else:
            raise RuntimeError('Newton line search failed')


def finite_adam(x, y, transforms, lam=0.01, steps=1000, lr=0.001):
    """Batch 12 support draws x 7 maps, with per-head clipping and common raw L2.

    Binary logits (0,s) are deliberate; this is not the old two-logit joint clip.
    All theta start at zero, so initial functions and mini-batches are identical.
    """
    names = tuple(transforms)
    ts = np.stack([transforms[n] for n in names])
    # x[draw,50,d+1], mapped[draw,map,50,d+1]. Float64 CPU only.
    mapped = np.einsum('bsd,mde->bmse', x, ts)
    b, m, n, d = mapped.shape
    data = torch.from_numpy(mapped.reshape(b*m,n,d))
    target = torch.from_numpy(np.repeat(y[:,None,:],m,axis=1).reshape(b*m,n))
    t = torch.from_numpy(np.tile(ts,(b,1,1)))
    theta = torch.zeros((b*m,d),dtype=torch.float64,requires_grad=True)
    optimizer = torch.optim.Adam([theta],lr=lr,weight_decay=0)
    results, trace, order = {}, [], hashlib.sha256()
    # Private generator, same sequence for all draws/maps, no global RNG mutation.
    generator = torch.Generator().manual_seed(194045)
    result_steps = sorted({v for v in LANDMARKS if v <= steps}|{steps})
    for step in range(steps+1):
        if step in result_steps:
            with torch.no_grad():
                beta = torch.bmm(t,theta[:,:,None]).squeeze(-1)
                score = torch.bmm(data,theta[:,:,None]).squeeze(-1)
                ce = torch.nn.functional.binary_cross_entropy_with_logits(score,target,reduction='none').mean(1)
                loss = ce+lam*beta.square().sum(1)/2
                residual = (data.transpose(1,2)@(score.sigmoid()-target)[:,:,None]).squeeze(-1)/n
                residual += lam*torch.bmm(t.transpose(1,2),beta[:,:,None]).squeeze(-1)
                raw_g = torch.linalg.solve(t.transpose(1,2),residual[:,:,None]).squeeze(-1)
                results[step] = theta.detach().numpy().reshape(b,m,d).copy()
                trace.append({'step':step, 'objective':loss.numpy().reshape(b,m).copy(),
                              'raw_gradient_inf':raw_g.abs().amax(1).numpy().reshape(b,m).copy()})
        if step == steps:
            break
        index = torch.randperm(n,generator=generator)[:min(32,n)]
        order.update(index.numpy().tobytes())
        optimizer.zero_grad()
        score = torch.bmm(data[:,index],theta[:,:,None]).squeeze(-1)
        beta = torch.bmm(t,theta[:,:,None]).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(score,target[:,index],reduction='none').mean(1)
        loss = loss+lam*beta.square().sum(1)/2
        loss.sum().backward()
        with torch.no_grad():
            norm = theta.grad.norm(dim=1,keepdim=True)
            theta.grad.mul_(torch.clamp(1/(norm+1e-6),max=1))
        optimizer.step()
    assert all(np.isfinite(v).all() for v in results.values())
    return results, trace, order.hexdigest()


def cell_metrics(scores, labels, domains):
    scores, labels, domains = map(np.asarray,(scores,labels,domains))
    cells = []
    for domain in (0,1):
        for label in (0,1):
            keep = (domains==domain)&(labels==label)
            if not keep.any():
                raise ValueError('Missing domain/class cell')
            s = scores[keep]
            margin = (2*label-1)*s
            cells.append({'accuracy':100*float(np.mean((s>0)==label)),
                          'ce':float(np.mean(np.logaddexp(0,-margin))),
                          'margin_mean':float(margin.mean()),
                          'margin_q10':float(np.quantile(margin,0.1)),
                          'margin_q50':float(np.quantile(margin,0.5)),
                          'margin_q90':float(np.quantile(margin,0.9))})
    def avg(ids):
        return {k:float(np.mean([cells[i][k] for i in ids])) for k in cells[0]}
    # Pooled quantiles are deliberately not inferred: these are mean cell quantiles.
    return dict(zip(COMPONENTS,[avg(range(4)),avg((0,1)),avg((2,3)),*cells],strict=True))


def conditional_geometry(support, labels, domains):
    rows = []
    for domain in (0,1):
        z = support[domains==domain]
        y = labels[domains==domain]
        means = [z[y==c].mean(0) for c in (0,1)]
        residual = np.concatenate([z[y==c]-means[c] for c in (0,1)])
        cov = residual.T@residual/len(z)
        eig = np.maximum(np.linalg.eigvalsh(cov),0)
        regularizer = 0.01*max(float(np.trace(cov)/z.shape[1]),1e-20)
        delta = means[1]-means[0]
        fisher = float(delta@np.linalg.solve(cov+regularizer*np.eye(z.shape[1]),delta))
        p = eig/eig.sum() if eig.sum()>0 else eig
        rows.append({'domain':domain, 'mean_separation':float(np.linalg.norm(delta)),
                     'within_trace':float(np.trace(cov)), 'regularizer':regularizer,
                     'regularized_fisher':fisher,
                     'within_energy_rank':float(np.exp(-np.sum(p[p>0]*np.log(p[p>0])))),
                     'top_energy_fraction':float(p[-1])})
    return rows
