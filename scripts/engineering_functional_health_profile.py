"""Synthetic-only fit/profile/forecast/report validation before scientific freeze."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from cafl4ds.accessibility_replication import fit
from cafl4ds.conditioning_fitting import augment, objective
from cafl4ds.functional_health_profile import FEATURES, MODELS, forecast, profile, screen
from cafl4ds.support_regularization import stratified_folds
from scripts.report_functional_health_profile import controls, make_report
from scripts.run_functional_health_profile import OUT, csv_write, digest, write


def main():
    out = OUT/'engineering'; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(195151)
    y = np.tile(np.repeat([0, 1], 32), 2); d = np.repeat([0, 1], 64)
    z = rng.normal(size=(128, 8)); z[:, 0] += .5*(2*y-1)
    fold = stratified_folds(y, d, 195100, folds=4); scores = np.empty((2, 128)); checks = []
    for k, lam in enumerate((.01, .001)):
        for f in range(4):
            tr, te = fold != f, fold == f
            beta, receipt = fit(z[tr], y[tr], lam); scores[k, te] = augment(z[te])@beta
            checks.append(receipt)
            if k == f == 0:
                again, _ = fit(z[tr], y[tr], lam); np.testing.assert_array_equal(beta, again)
                x = augment(z[tr]); penalty = np.eye(9)
                fun = lambda b: objective(b, x, y[tr], penalty, lam)
                opt = minimize(lambda b: fun(b)[0], np.zeros(9), jac=lambda b: fun(b)[1],
                    hess=lambda b: fun(b)[2], method='trust-exact', options={'gtol': 1e-11, 'maxiter': 100})
                assert np.max(abs(fun(opt.x)[1])) < 1e-8
                assert np.max(abs(augment(z)@(opt.x-beta))) < 1e-6
    control = controls(); assert max(abs(control[0, [0, 2]])) < 1e-12
    assert max(abs(control[1, [0, 1, 3]])) < 1e-12
    h = np.array([[[profile(scores[k]+rng.normal(0, .1, 128)+rng.normal(0, .15), y, d)
                    for k in range(2)] for _ in range(6)] for _ in range(72)])
    full = h.copy(); full[..., [0, 4]] += .12; full[..., [3, 7]] *= .9
    groups = np.array([(s, b, c) for s in (1, 2, 3) for b in (1, 2, 3)
                       for c in (1, 2, 3, 4) for _ in range(2)])
    diets = np.tile([0, 1], 36); signals = rng.normal(size=(72, 16))
    target = rng.normal(0, 7, (72, 2, 7))+h[:, :, 0, 4].mean(1)[:, None, None]*5
    all_pred = []; screens = {}
    panels = [('mean_lambda01', h[:, :, 0].mean(1)), ('mean_lambda001', h[:, :, 1].mean(1))]
    panels += [(f'pool{o}_lambda01', h[:, o, 0]) for o in range(6)]
    for name, hh in panels:
        pred, _, _ = forecast(signals, diets, groups, hh, target.reshape(72, 14))
        pp = np.stack([pred[m].reshape(72, 2, 7) for m in MODELS], 1); all_pred.append(pp)
        screens[name] = screen(target[:, 1, 6], {m: pp[:, j, 1, 6] for j, m in enumerate(MODELS)}, groups)
    all_pred = np.array(all_pred)
    np.savez_compressed(out/'analysis-arrays.npz', profiles=h, full_profiles=full,
        target=target, groups=groups, diets=diets, predictions=all_pred)
    write(out/'summary.json', {'states': 72, 'strict_holdout_folds': 36, 'screens': screens,
        'primary_MSE': {m: float(np.mean((all_pred[0, :, j, 1, 6]-target[:, 1, 6])**2)) for j, m in enumerate(MODELS)}})
    corr = [{'diet': diet, 'feature': name, 'pearson_r': float(np.corrcoef(h[diets == diet, :, 0, j].mean(1), target[diets == diet, 1, 6])[0, 1])}
            for diet in (0, 1) for j, name in enumerate(FEATURES)]
    shadow = []
    for pol in ('always_current', 'always_replay', 'cheap', 'functional_added', 'cka_functional_added'):
        action = np.zeros(72, bool) if pol == 'always_current' else np.ones(72, bool) if pol == 'always_replay' else all_pred[0, :, MODELS.index(pol), 1, 6] > 0
        for k, comp in enumerate(('balanced', 'day', 'night', 'day_city', 'day_highway', 'night_city', 'night_highway')):
            shadow.append({'model': pol, 'component': comp, 'value_relative_current_pp': float(np.mean(target[:, 1, k]*action))})
    csv_write(out/'correlations.csv', corr); csv_write(out/'shadow-values.csv', shadow)
    delivery = make_report(out, out/'synthetic.ipynb', engineering=True)
    write(OUT/'engineering-verification.json', {'passed': True, 'visual_QA_pending': True,
        'synthetic_only': True, 'synthetic_collection_fits': 8, 'synthetic_audit_fits': 2,
        'controls': control.tolist(), 'fit_receipts': checks, 'hashes': delivery['hashes'] | {
            str(p): digest(p) for p in (out/'analysis-arrays.npz', out/'summary.json', out/'correlations.csv', out/'shadow-values.csv')}})
    print('Synthetic fitting, profile, all forecast panels and six figures complete; visual QA pending.')


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
