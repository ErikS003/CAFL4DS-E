"""Matched disjoint calibration/evaluation banks and crossed functional readers."""
from __future__ import annotations

import numpy as np

from cafl4ds.functional_response_profile import designs
from cafl4ds.preaction_benefit import ridge

SEEDS = (195551, 195552, 195553, 195554, 195555, 195556)
LAMBDAS = (.01, .001)
HORIZONS = (1, 8, 32, 128, 384)
MODELS = ('mean', 'cheap', 'static', 'static_response_added')
CONDITIONS = ('same_reader_same_bank', 'same_reader_other_bank', 'other_reader_same_bank', 'other_reader_other_bank')
SWITCHES = ((0, 0), (0, 1), (1, 0), (1, 1))
SCALES = ('evaluation', 'calibration')


def allocations(labels, domains):
    y, d = np.asarray(labels)[:200], np.asarray(domains)[:200]
    calib = np.empty((6, 2, 64), int); evaluation = np.empty((6, 2, 32), int); unused = np.empty((6, 8), int)
    for draw, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        cells = [rng.permutation(np.flatnonzero((d == dom) & (y == cls))) for dom in (0, 1) for cls in (0, 1)]
        if any(len(ids) != 50 for ids in cells):
            raise ValueError('Exactly50 validation images per class/domain required')
        for bank in range(2):
            calib[draw, bank] = np.concatenate([ids[16*bank:16*(bank+1)] for ids in cells])
            evaluation[draw, bank] = np.concatenate([ids[32+8*bank:40+8*bank] for ids in cells])
        unused[draw] = np.concatenate([ids[48:] for ids in cells])
        all_ids = np.r_[calib[draw].ravel(), evaluation[draw].ravel(), unused[draw]]
        assert len(np.unique(all_ids)) == 200 and np.array_equal(np.sort(all_ids), np.arange(200))
    return calib, evaluation, unused


def sigma(scores, labels, domains):
    s, y, d = np.asarray(scores, float), np.asarray(labels), np.asarray(domains)
    out = []
    for dom in (0, 1):
        cells = [s[(d == dom) & (y == c)] for c in (0, 1)]
        if any(len(v) < 2 for v in cells):
            raise ValueError('At least two scores per class/domain required')
        out.append(max(1e-8, float(np.sqrt(sum(v.var() for v in cells)/2))))
    return np.array(out)


def descriptors(before, after, labels, domains, calibration_sigma):
    """Two scale conventions, same fixed baseline head and baseline thresholds.

    after [...,n]; result [...,scale,8], static[scale,8]. Spread RESPONSE always
    log(after evaluation SD/baseline evaluation SD); static log SD unchanged.
    Calibration scale changes only separation/preference denominators and the
    boundary threshold. It is fixed for both evaluation banks of that reader.
    """
    a, b, y, d = np.asarray(before, float), np.asarray(after, float), np.asarray(labels), np.asarray(domains)
    if a.ndim != 1 or b.shape[-1] != len(a) or not np.isfinite(b).all():
        raise ValueError('Aligned finite score arrays required')
    es = sigma(a, y, d); cs = np.asarray(calibration_sigma, float)
    if cs.shape != (2,) or not np.isfinite(cs).all() or np.any(cs < 1e-8):
        raise ValueError('Fixed positive two-domain calibration scales required')
    result = np.empty((*b.shape[:-1], 2, 8)); static = np.empty((2, 8))
    for dom in (0, 1):
        cells = [(d == dom) & (y == c) for c in (0, 1)]
        delta = [np.mean(b[..., mask]-a[mask], axis=-1) for mask in cells]
        mu = [a[mask].mean() for mask in cells]
        new_sd = np.maximum(1e-8, np.sqrt(sum(b[..., mask].var(axis=-1) for mask in cells)/2))
        for scale, unit in enumerate((es[dom], cs[dom])):
            boundary = np.mean([np.mean(abs(a[mask]) <= .25*unit) for mask in cells])
            mass = sum(np.mean(abs(b[..., mask]) <= .25*unit, axis=-1)-np.mean(abs(a[mask]) <= .25*unit) for mask in cells)/2
            result[..., scale, 4*dom:4*dom+4] = np.stack([(delta[1]-delta[0])/(2*unit), sum(delta)/(2*unit), np.log(new_sd/es[dom]), mass], axis=-1)
            static[scale, 4*dom:4*dom+4] = [(mu[1]-mu[0])/(2*unit), sum(mu)/(2*unit), np.log(es[dom]), boundary]
    return result, static


def crossed_variance(values):
    """Finite2x2 reader×evaluation ANOVA, ending [...,reader,bank,axis]."""
    x = np.asarray(values, float)
    if x.shape[-3:-1] != (2, 2) or not np.isfinite(x).all():
        raise ValueError('Finite reader×bank2x2 grid required')
    mu = x.mean((-3, -2), keepdims=True)
    reader = x.mean(-2, keepdims=True)-mu; bank = x.mean(-3, keepdims=True)-mu
    interaction = x-mu-reader-bank
    out = np.stack([((x-mu)**2).mean((-3, -2)), (reader**2).mean((-3, -2)), (bank**2).mean((-3, -2)),
                    (interaction**2).mean((-3, -2))], -1)
    np.testing.assert_allclose(out[..., 0], out[..., 1:].sum(-1), rtol=1e-10, atol=1e-18)
    return out


def forecast(signals, diets, groups, static, local, target):
    xs = designs(signals, diets, static, local); y = np.asarray(target, float)
    predictions = np.empty((72, 4, 5, 8)); maximum = 0.; folds = []
    for group in np.unique(groups, axis=0):
        tr = np.flatnonzero(np.all(groups != group, axis=1)); te = np.flatnonzero(np.all(groups == group, axis=1))
        assert len(tr) == 24 and len(te) == 2 and all(not set(groups[tr, k]) & set(groups[te, k]) for k in range(3))
        for j, name in enumerate(MODELS):
            x = xs[name]; p = ridge(x[tr], y[tr].reshape(24, 40), x[te], 10.)
            mu, sd = x[tr].mean(0), x[tr].std(0); sd = np.where(sd > 1e-12, sd, 1.)
            a, b = (x[tr]-mu)/sd, (x[te]-mu)/sd; ym = y[tr].reshape(24, 40).mean(0)
            beta = np.linalg.lstsq(np.r_[a, np.sqrt(10)*np.eye(x.shape[1])],
                np.r_[y[tr].reshape(24, 40)-ym, np.zeros((x.shape[1], 40))], rcond=None)[0]
            error = float(np.max(abs(p-(ym+b@beta)))); maximum = max(maximum, error); assert error < 1e-9
            predictions[te, j] = p.reshape(2, 5, 8)
        folds.append({'group': group.tolist(), 'train': tr.tolist(), 'test': te.tolist()})
    return predictions, folds, maximum


def screen_errors(errors, groups):
    """Per-state errors already averaged AFTER squaring over draws/orientations."""
    e = np.asarray(errors, float)
    if e.shape != (72, 4):
        raise ValueError('State×four-model squared errors required')
    rows = {}
    for j, baseline in ((1, 'cheap'), (2, 'static')):
        gain = e[:, j]-e[:, 3]; denominator = float(e[:, j].mean())
        fraction = float(gain.mean()/denominator) if denominator > 0 else 0.
        seeds = [float(gain[groups[:, 0] == s].mean()) for s in np.unique(groups[:, 0])]
        sources = [float(gain[groups[:, 1] == s].mean()) for s in np.unique(groups[:, 1])]
        rows[baseline] = {'MSE_reduction_fraction': fraction, 'comparison_defined': denominator > 0,
            'improvement_per_seed': seeds, 'improvement_per_source': sources,
            'passed': bool(denominator > 0 and fraction >= .1 and min(seeds) > 0 and min(sources) > 0)}
    return {'baselines': rows, 'development_screen_passed': all(r['passed'] for r in rows.values()), 'no_controller_qualified': True}


def grid_forecasts(local, static, future, signals, groups, diets, progress=None):
    pred = np.empty((6, 2, 2, 2, 2, 4, 72, 4, 5, 8)); errors = np.empty_like(pred)
    correlations = np.empty((6, 2, 2, 2, 2, 4, 5, 8)); maximum = 0.
    for draw, k, scale, reader, bank, condition in np.ndindex(6, 2, 2, 2, 2, 4):
        dr, de = SWITCHES[condition]; tr, tb = reader ^ dr, bank ^ de
        h, r, y = static[:, draw, k, reader, bank, scale], local[:, draw, k, reader, bank, scale], future[:, draw, k, tr, tb, scale]
        pp, folds, error = forecast(signals, diets, groups, h, r, y); maximum = max(maximum, error)
        key = (draw, k, scale, reader, bank, condition); pred[key] = pp; errors[key] = (pp-y[:, None])**2
        for t, axis in np.ndindex(5, 8):
            a, b = r[:, axis], y[:, t, axis]
            correlations[(*key, t, axis)] = np.corrcoef(a, b)[0, 1] if a.std() > 1e-15 and b.std() > 1e-15 else np.nan
        if condition == 3 and reader == bank == 1 and progress is not None:
            progress((draw*4+k*2+scale+1)*16)
    state_errors = errors.mean((0, 3, 4))
    for condition in range(1, 4):
        np.testing.assert_allclose(state_errors[:, :, condition, :, 1], state_errors[:, :, 0, :, 1], atol=1e-12, rtol=1e-12)
    return pred, errors, correlations, state_errors, maximum, folds
