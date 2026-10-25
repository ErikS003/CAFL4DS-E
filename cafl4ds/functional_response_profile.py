"""Matched, fixed-reader local response measurements; no future utility input."""
from __future__ import annotations

import numpy as np

from cafl4ds.functional_health_profile import profile
from cafl4ds.preaction_benefit import design, ridge

MODELS = ('mean', 'cheap', 'static', 'response_only', 'response_added', 'static_response_added', 'cka_response_added')


def response(before, after, labels, domains):
    a, b = np.asarray(before, float), np.asarray(after, float)
    y, d = np.asarray(labels), np.asarray(domains)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(b).all():
        raise ValueError('Aligned finite score functions required')
    # Validation and spread floor use the same contract as study51.
    profile(a, y, d)
    result = []
    for dom in (0, 1):
        cells = [(d == dom) & (y == c) for c in (0, 1)]
        sigma = max(1e-8, np.sqrt(np.mean([a[v].var() for v in cells])))
        new_sigma = max(1e-8, np.sqrt(np.mean([b[v].var() for v in cells])))
        delta = [np.mean(b[v]-a[v]) for v in cells]
        mass = np.mean([np.mean(abs(b[v]) <= .25*sigma)-np.mean(abs(a[v]) <= .25*sigma) for v in cells])
        result += [(delta[1]-delta[0])/(2*sigma), sum(delta)/(2*sigma), np.log(new_sigma/sigma), mass]
    return np.array(result)


def designs(signals, diets, static, delta):
    n = len(diets); diet = np.asarray(diets, float)[:, None]
    h, r = np.asarray(static, float), np.asarray(delta, float)
    if h.shape != (n, 8) or r.shape != (n, 8) or not np.isfinite(h).all() or not np.isfinite(r).all():
        raise ValueError('Aligned eight-axis static/response profiles required')
    cheap = design(signals, diets, 'cheap'); cka = design(signals, diets, 'health_added')
    hs = np.c_[h[:, 4:], h[:, 4:]*diet]; rs = np.c_[r[:, 4:], r[:, 4:]*diet]
    return {'mean': np.zeros((n, 0)), 'cheap': cheap, 'static': np.c_[cheap, hs],
            'response_only': np.c_[diet, rs], 'response_added': np.c_[cheap, rs],
            'static_response_added': np.c_[cheap, hs, rs], 'cka_response_added': np.c_[cka, rs]}


def forecast(signals, diets, groups, static, delta, targets):
    xs = designs(signals, diets, static, delta); y = np.asarray(targets, float)
    groups = np.asarray(groups); predictions = {m: np.empty_like(y) for m in MODELS}
    folds = []; maximum = 0.
    for group in np.unique(groups, axis=0):
        tr = np.flatnonzero(np.all(groups != group, axis=1)); te = np.flatnonzero(np.all(groups == group, axis=1))
        assert len(tr) == 24 and len(te) == 2
        assert all(not set(groups[tr, k]) & set(groups[te, k]) for k in range(3))
        for m, x in xs.items():
            p = ridge(x[tr], y[tr], x[te], 10.); predictions[m][te] = p
            mu, sd = x[tr].mean(0), x[tr].std(0); sd = np.where(sd > 1e-12, sd, 1.)
            a, b = (x[tr]-mu)/sd, (x[te]-mu)/sd
            ym = y[tr].mean(0)
            beta = np.linalg.lstsq(np.r_[a, np.sqrt(10)*np.eye(x.shape[1])],
                np.r_[y[tr]-ym, np.zeros((x.shape[1], y.shape[1]))], rcond=None)[0]
            error = float(np.max(abs(p-(ym+b@beta)))); maximum = max(maximum, error); assert error < 1e-9
        folds.append({'group': group.tolist(), 'train': tr.tolist(), 'test': te.tolist()})
    return predictions, folds, maximum


def screen(y, predictions, groups):
    rows = {}
    new = (predictions['static_response_added']-y)**2
    for baseline in ('cheap', 'static'):
        old = (predictions[baseline]-y)**2; change = old-new
        fraction = float(change.mean()/old.mean())
        seeds = [float(change[groups[:, 0] == s].mean()) for s in np.unique(groups[:, 0])]
        sources = [float(change[groups[:, 1] == s].mean()) for s in np.unique(groups[:, 1])]
        rows[baseline] = {'MSE_reduction_fraction': fraction, 'improvement_per_seed': seeds,
                         'improvement_per_source': sources, 'passed': bool(fraction >= .1 and min(seeds) > 0 and min(sources) > 0)}
    return {'baselines': rows, 'development_screen_passed': all(v['passed'] for v in rows.values()),
            'no_controller_qualified': True}
