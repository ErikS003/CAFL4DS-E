"""Pre-action, support-cross-fitted functional measurements and fixed forecasts."""
from __future__ import annotations

import numpy as np

from cafl4ds.preaction_benefit import design as legacy_design, ridge

FEATURES = tuple(f'{d}_{m}' for d in ('day', 'night') for m in
                 ('separation', 'preference', 'log_spread', 'boundary_mass'))
MODELS = ('mean', 'diet', 'cheap', 'cka_added', 'functional_only',
          'functional_added', 'cka_functional_added', 'day_night_added')
COMPONENTS = ('balanced', 'day', 'night', 'day_city', 'day_highway', 'night_city', 'night_highway')


def profile(scores, labels, domains):
    """Four interpretable descriptors per domain; no intervention targets accepted.

    Scores are highway-versus-city logits. Within-class population variances
    receive equal weight. Positive separation is useful; positive preference
    shifts both class means toward highway. Spread floor and boundary fixed.
    """
    s, y, d = np.asarray(scores, float), np.asarray(labels), np.asarray(domains)
    if s.ndim != 1 or not s.shape == y.shape == d.shape or not np.isfinite(s).all():
        raise ValueError('Aligned finite scores and metadata required')
    result = []
    for domain in (0, 1):
        cells = [s[(d == domain) & (y == c)] for c in (0, 1)]
        if any(len(c) < 2 for c in cells):
            raise ValueError('At least two scores per class/domain required')
        mu0, mu1 = [c.mean() for c in cells]
        spread = max(1e-8, float(np.sqrt(np.mean([c.var() for c in cells]))))
        boundary = np.mean([np.mean(abs(c) <= .25*spread) for c in cells])
        result.extend(((mu1-mu0)/(2*spread), (mu1+mu0)/(2*spread), np.log(spread), boundary))
    return np.array(result)


def designs(signals, diets, profiles):
    h = np.asarray(profiles, float)
    diet = np.asarray(diets, float)[:, None]
    if h.shape != (len(diet), 8) or not np.isfinite(h).all():
        raise ValueError('Eight finite pre-action measurements required')
    cheap = legacy_design(signals, diets, 'cheap')
    cka = legacy_design(signals, diets, 'health_added')
    night = np.c_[h[:, 4:], h[:, 4:]*diet]
    both = np.c_[h, h*diet]
    return {'mean': legacy_design(signals, diets, 'mean'), 'diet': diet,
            'cheap': cheap, 'cka_added': cka, 'functional_only': np.c_[diet, night],
            'functional_added': np.c_[cheap, night], 'cka_functional_added': np.c_[cka, night],
            'day_night_added': np.c_[cheap, both]}


def forecast(signals, diets, groups, profiles, targets, audit=True):
    """24 training rows/2 test rows, excluding seed AND source AND context."""
    xs = designs(signals, diets, profiles)
    groups, targets = np.asarray(groups), np.asarray(targets, float)
    result = {m: np.empty_like(targets) for m in MODELS}
    folds, maximum_error = [], 0.
    for group in np.unique(groups, axis=0):
        tr = np.flatnonzero(np.all(groups != group, axis=1))
        te = np.flatnonzero(np.all(groups == group, axis=1))
        assert len(tr) == 24 and len(te) == 2
        assert all(not set(groups[tr, k]) & set(groups[te, k]) for k in range(3))
        for m, x in xs.items():
            pred = ridge(x[tr], targets[tr], x[te], 10.)
            result[m][te] = pred
            if audit:
                mu, sd = x[tr].mean(0), x[tr].std(0)
                sd = np.where(sd > 1e-12, sd, 1.)
                a, b = (x[tr]-mu)/sd, (x[te]-mu)/sd
                y, ym = targets[tr], targets[tr].mean(0)
                coefficient = np.linalg.lstsq(np.r_[a, np.sqrt(10)*np.eye(x.shape[1])],
                    np.r_[y-ym, np.zeros((x.shape[1], targets.shape[1]))], rcond=None)[0]
                error = float(np.max(abs(pred-(ym+b@coefficient))))
                maximum_error = max(error, maximum_error)
                assert error <= 1e-9
        folds.append({'group': group.tolist(), 'train': tr.tolist(), 'test': te.tolist()})
    return result, folds, maximum_error


def screen(target, predictions, groups):
    base = (predictions['cheap']-target)**2
    added = (predictions['functional_added']-target)**2
    improvement = base-added
    seeds = [float(improvement[groups[:, 0] == s].mean()) for s in np.unique(groups[:, 0])]
    sources = [float(improvement[groups[:, 1] == s].mean()) for s in np.unique(groups[:, 1])]
    fraction = float(improvement.mean()/base.mean())
    return {'MSE_reduction_fraction': fraction, 'MSE_improvement_per_seed': seeds,
            'MSE_improvement_per_source': sources,
            'development_screen_passed': bool(fraction >= .1 and min(seeds) > 0 and min(sources) > 0),
            'no_alarm_or_controller_qualified': True}
