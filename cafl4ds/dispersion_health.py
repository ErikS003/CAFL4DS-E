"""Task-direction dispersion, global scale controls and strictly nested forecasts."""
from __future__ import annotations
import numpy as np
from cafl4ds.preaction_benefit import design, ridge

COMPONENTS = ('global', 'reader_total', 'reader_excess', 'matched_random_excess', 'isotropic_random_excess')
ACCESS = ('fixed_reader_accuracy_pp', 'fixed_reader_CE', 'fixed_reader_signed_margin')
MODELS = ('mean', 'cheap_fixed10', 'cheap', 'static', 'global', 'random', 'reader_total', 'reader_excess')
PENALTIES = (1., 10., 100., 1000., 10000.)
HORIZONS = (1, 8, 32, 128, 384)


def variances(features, directions, labels, domains):
    """Equal-class within-domain covariance trace and directional variance."""
    x = np.asarray(features, float); q = np.asarray(directions, float)
    trace = []; projected = []
    for dom in (0, 1):
        tc = []; pc = []
        for cls in (0, 1):
            z = x[..., (domains == dom) & (labels == cls), :]
            z = z-z.mean(-2, keepdims=True)
            tc.append(np.sum(z*z, axis=-1).mean(-1))
            pc.append(np.mean((z@q.T)**2, axis=-2))
        trace.append((tc[0]+tc[1])/2); projected.append((pc[0]+pc[1])/2)
    return np.stack(trace, -1), np.stack(projected, -2)


def directions(before, heads, calibration, labels, domains, draw, group):
    """Two readers, four matched randoms each, eight independent isotropic controls.

    Candidate pool and selection rule fixed before eval. Match unit-reader
    calibration within-class variance; record imperfect matches, never tune.
    """
    readers = heads[:, :-1]/np.linalg.norm(heads[:, :-1], axis=1, keepdims=True)
    matched = []; gaps = []; selected = []
    for half in (0, 1):
        rng = np.random.default_rng(195701+100*draw+10*group+half)
        candidates = rng.normal(size=(256, before.shape[-1])); candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        ids = calibration[half]
        _, vv = variances(before[ids], np.r_[readers[half:half+1], candidates], labels[ids], domains[ids])
        # Average log variance across domains; no evaluation image is touched.
        target = np.log(np.maximum(vv[:, 0], 1e-16)).mean()
        gap = np.abs(np.log(np.maximum(vv[:, 1:], 1e-16)).mean(0)-target)
        ix = np.argsort(gap, kind='stable')[:4]
        matched.extend(candidates[ix]); gaps.append(gap[ix]); selected.append(ix)
    rng = np.random.default_rng(195799+100*draw+10*group)
    isotropic = rng.normal(size=(8, before.shape[-1])); isotropic /= np.linalg.norm(isotropic, axis=1, keepdims=True)
    return np.r_[readers, matched, isotropic], np.array(gaps), np.array(selected)


def measure(before, after, q, heads, labels, domains):
    """after[view,image,feature] -> components[view,domain,5],static[domain,2]."""
    a, b = np.asarray(before, float), np.asarray(after, float)
    tr0, v0 = variances(a, q, labels, domains); tr1, v1 = variances(b, q, labels, domains)
    floors = int(np.sum(tr0 < 1e-16)+np.sum(v0 < 1e-16)+np.sum(tr1 < 1e-16)+np.sum(v1 < 1e-16))
    g = .5*np.log(np.maximum(tr1, 1e-16)/np.maximum(tr0, 1e-16))
    response = .5*np.log(np.maximum(v1, 1e-16)/np.maximum(v0, 1e-16))
    reader = response[..., :2].mean(-1); random = response[..., 2:10].mean(-1); iso = response[..., 10:].mean(-1)
    components = np.stack([g, reader, reader-g, random-g, iso-g], -1)
    static = np.stack([.5*np.log(np.maximum(tr0, 1e-16)), .5*np.log(np.maximum(v0[..., :2], 1e-16)).mean(-1)], -1)
    # Accessibility means metrics from two fixed heads, rather than an averaged classifier.
    score0 = a@heads[:, :-1].T+heads[:, -1]
    score1 = b@heads[:, :-1].T+heads[:, -1]
    accessibility = np.empty((len(b), 2, 3)); base_access = np.empty((2, 3)); extras = np.empty((len(b), 2, 3))
    for dom in (0, 1):
        ix = domains == dom; y = labels[ix, None]
        s0, s1 = score0[ix], score1[:, ix]
        def metrics(s):
            return np.stack([100*((s > 0) == y).mean((-2, -1)),
                (np.logaddexp(0, s)-y*s).mean((-2, -1)), ((2*y-1)*s).mean((-2, -1))], -1)
        base_access[dom] = metrics(s0); accessibility[:, dom] = metrics(s1)-base_access[dom]
        # Keep separation/common-score preference/boundary as distinct diagnostic coordinates.
        masks = [(domains == dom) & (labels == cls) for cls in (0, 1)]
        sd = np.maximum(1e-8, np.sqrt(sum(score0[v].var(0) for v in masks)/2))
        delta = [(score1[:, v]-score0[v]).mean(1) for v in masks]
        boundary = sum((abs(score1[:, v]) <= .25*sd).mean(1)-(abs(score0[v]) <= .25*sd).mean(0) for v in masks)/2
        extras[:, dom] = np.stack([((delta[1]-delta[0])/(2*sd)).mean(-1), ((delta[1]+delta[0])/(2*sd)).mean(-1), boundary.mean(-1)], -1)
    assert np.isfinite(components).all() and np.isfinite(accessibility).all()
    np.testing.assert_allclose(components[..., 1], components[..., 0]+components[..., 2], atol=1e-14, rtol=1e-12)
    return components, static, accessibility, base_access, extras, floors


def designs(signals, diets, static, local):
    cheap = design(signals, diets, 'cheap'); h = static[:, 1]; r = local[:, 1]
    base = np.c_[cheap, h]; glob = np.c_[base, r[:, 0]]
    return {'mean': np.zeros((len(diets), 0)), 'cheap_fixed10': cheap, 'cheap': cheap, 'static': base,
        'global': glob, 'random': np.c_[glob, r[:, 3]], 'reader_total': np.c_[base, r[:, 1]], 'reader_excess': np.c_[glob, r[:, 2]]}


def checked_fit(x, y, z, penalty, recorder, key):
    recorder('started', key)
    p = ridge(x, y, z, penalty)
    mu, sd = x.mean(0), x.std(0); sd = np.where(sd > 1e-12, sd, 1.)
    a, b = (x-mu)/sd, (z-mu)/sd; ym = y.mean(0)
    beta = np.linalg.lstsq(np.r_[a, np.sqrt(penalty)*np.eye(x.shape[1])], np.r_[y-ym, np.zeros((x.shape[1], y.shape[1]))], rcond=None)[0]
    error = float(np.max(abs(p-(ym+b@beta)))); assert error < 1e-8
    recorder('completed', {**key, 'independent_LS_error': error})
    return p, error


def nested_forecast(signals, diets, groups, static, local, targets, recorder, prior):
    xs = designs(signals, diets, static, local); predictions = np.empty((72, 8, 14))
    choices = np.empty((36, 8)); losses = np.full((36, 8, 5), np.nan); folds = []; maximum = 0.
    for outer, group in enumerate(np.unique(groups, axis=0)):
        tr = np.flatnonzero(np.all(groups != group, axis=1)); te = np.flatnonzero(np.all(groups == group, axis=1))
        assert len(tr) == 24 and len(te) == 2
        inner = []
        for held in np.unique(groups[tr], axis=0):
            it = tr[np.all(groups[tr] != held, axis=1)]; iv = tr[np.all(groups[tr] == held, axis=1)]
            assert len(it) == 4 and len(iv) == 2 and not set(it) & set(te) and not set(iv) & set(te)
            inner.append({'train': it.tolist(), 'valid': iv.tolist()})
        assert len(inner) == 12
        for j, model in enumerate(MODELS):
            x = xs[model]
            if model in ('mean', 'cheap_fixed10'):
                penalty = 10.
            else:
                scores = []
                for li, lam in enumerate(PENALTIES):
                    ss = []
                    for fold, item in enumerate(inner):
                        it, iv = item['train'], item['valid']
                        # Tuning uses ONLY primary utility within outer training, never future geometry.
                        pp, error = checked_fit(x[it], targets[it, 13:14], x[iv], lam, recorder,
                            {'stage': 'inner', 'prior': prior, 'outer': outer, 'model': model, 'penalty': lam, 'inner': fold})
                        maximum = max(maximum, error); ss.extend(((pp[:, 0]-targets[iv, 13])**2).tolist())
                    scores.append(np.mean(ss))
                losses[outer, j] = scores
                # Largest penalty wins numerical ties; fixed before results.
                best = np.flatnonzero(np.array(scores) <= min(scores)+1e-12*max(1, abs(min(scores))))[-1]
                penalty = PENALTIES[best]
            choices[outer, j] = penalty
            pp, error = checked_fit(x[tr], targets[tr], x[te], penalty, recorder, {'stage': 'outer_utility', 'prior': prior, 'outer': outer, 'model': model, 'penalty': penalty})
            predictions[te, j] = pp; maximum = max(maximum, error)
        folds.append({'group': group.tolist(), 'train': tr.tolist(), 'test': te.tolist(), 'inner': inner})
    return predictions, choices, losses, folds, maximum


def secondary_forecast(signals, diets, groups, static, local, targets, penalties, recorder, setting):
    xs = designs(signals, diets, static, local); prediction = np.empty((72, 8, 80)); maximum = 0.
    for outer, group in enumerate(np.unique(groups, axis=0)):
        tr = np.flatnonzero(np.all(groups != group, axis=1)); te = np.flatnonzero(np.all(groups == group, axis=1))
        for j, model in enumerate(MODELS):
            pp, error = checked_fit(xs[model][tr], targets[tr], xs[model][te], penalties[outer, j], recorder,
                {'stage': 'outer_secondary', 'setting': setting, 'outer': outer, 'model': model, 'penalty': float(penalties[outer, j])})
            prediction[te, j] = pp; maximum = max(maximum, error)
    return prediction, maximum


def screen(errors, groups):
    """Primary reader-excess versus fixed, predeclared controls."""
    result = {}
    for j, name in ((0, 'mean'), (2, 'cheap'), (3, 'static'), (4, 'global'), (5, 'random')):
        gain = errors[:, j]-errors[:, 7]; old = errors[:, j].mean()
        fraction = float(gain.mean()/old) if old > 0 else None
        seeds = [float(gain[groups[:, 0] == v].mean()) for v in np.unique(groups[:, 0])]
        sources = [float(gain[groups[:, 1] == v].mean()) for v in np.unique(groups[:, 1])]
        threshold = 0 if name == 'mean' else .1
        result[name] = {'MSE_reduction_fraction': fraction, 'improvement_per_seed': seeds, 'improvement_per_source': sources,
            'passed': bool(fraction is not None and (fraction > 0 if name == 'mean' else fraction >= threshold) and min(seeds) > 0 and min(sources) > 0)}
    return {'baselines': result, 'development_screen_passed': all(v['passed'] for v in result.values()), 'no_controller_qualified': True}


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-15 and b.std() > 1e-15 else None


def diagnostics(local, future, accessibility):
    rows = []; variance = []
    for k, t, dom, source, target in np.ndindex(2, 5, 2, 5, 8):
        values = []
        for draw, group, bank in np.ndindex(6, 2, 2):
            x = local[:, draw, k, group, bank, dom, source]
            y = future[:, draw, k, 1-group, 1-bank, t, dom, target] if target < 5 else accessibility[:, draw, k, 1-group, 1-bank, t, dom, target-5]
            r = correlation(x, y)
            if r is not None:
                values.append(r)
        rows.append({'prior': k, 'horizon': HORIZONS[t], 'domain': dom, 'source': COMPONENTS[source],
            'target': (COMPONENTS+ACCESS)[target], 'q25_median_q75': np.quantile(values, [.25, .5, .75]).tolist() if values else [None]*3,
            'finite_orientation_correlations': len(values)})
    for k, dom in np.ndindex(2, 2):
        for time in range(-1, 5):
            x = local[:, :, k, :, :, dom] if time == -1 else future[:, :, k, :, :, time, dom]
            g, total, excess = x[..., 0], x[..., 1], x[..., 2]
            # Center across states independently inside each fixed measurement setting.
            g, total, excess = [v-v.mean(0, keepdims=True) for v in (g, total, excess)]
            vg, vt, ve, twice_cov = [float(np.mean(v)) for v in (g*g, total*total, excess*excess, 2*g*excess)]
            assert abs(vt-vg-ve-twice_cov) < 1e-10*max(1, vt)
            variance.append({'prior': k, 'domain': dom, 'horizon': 'local' if time == -1 else HORIZONS[time],
                'global_variance': vg, 'reader_variance': vt, 'excess_variance': ve, 'twice_covariance': twice_cov})
    return {'cross_reader_cross_image_correlations': rows, 'signed_variance_identity': variance}
