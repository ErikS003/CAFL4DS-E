"""Freeze, fit support-only profiles, seal them, then evaluate known benefits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from cafl4ds.accessibility_replication import fit
from cafl4ds.conditioning_fitting import augment, objective
from cafl4ds.functional_health_profile import COMPONENTS, FEATURES, MODELS, forecast, profile, screen
from cafl4ds.support_regularization import stratified_folds

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'cafl4ds/configs/functional_health_profile.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8'); tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(hashes):
    for p, h in hashes.items():
        assert digest(p) == h, p


def csv_write(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


CFG = read(CONFIG); OUT = ROOT/CFG['output']; GEO = ROOT/CFG['geometry']
READERS = ROOT/CFG['readers']; OLD = ROOT/CFG['outcomes']
MANIFEST = ROOT/'outputs/early-state-adaptation/20261019/evaluation/data-role-manifest.json'


def reserve():
    assert shutil.disk_usage(OUT).free > CFG['minimum_free_mib']*2**20
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) < CFG['max_output_mib']*2**20


def status(stage, **kw):
    value = {'stage': stage, 'pid': os.getpid(), 'unix': time.time(), **kw}
    write(OUT/'status.json', value); print(json.dumps(value), flush=True)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True); reserve(); assert not (OUT/'input_lock.json').exists()
    eng = read(OUT/'engineering-verification.json'); assert eng['passed'] and not eng['visual_QA_pending']; check(eng['hashes'])
    cases = read(OLD/'plan.json')['cases']; jobs = read(READERS/'plan.json')['jobs'][27:]
    assert len(cases) == len(jobs) == 72
    for c, j in zip(cases, jobs, strict=True):
        assert j['point'] == f"s43/{c['id']}/fork"
        assert all(c[k] == j[k] for k in ('seed', 'source', 'context', 'pre_diet'))
    with np.load(READERS/'design.npz') as z:
        pools, labels, domains = z['pools'][0], z['labels'], z['domains']
    assert pools.shape == (6, 128) and pools.max() < 200
    records = read(MANIFEST)['records']; assert len(records) == 2248
    assert all(r['role'] == 'support_pool' and Path(r['path']).parent.name == 'val' for r in records[:200])
    assert all(r['role'] == 'fresh_query' and Path(r['path']).parent.name == 'train' for r in records[200:])
    assert not {r['clip_id'] for r in records[:200]} & {r['clip_id'] for r in records[200:]}
    assert all(r['label'] == int(labels[i]) and r['domain'] == int(domains[i]) for i, r in enumerate(records))
    folds = np.array([stratified_folds(labels[ix], domains[ix], s, folds=4)
                     for ix, s in zip(pools, CFG['fold_seeds'], strict=True)])
    for o, ix in enumerate(pools):
        for f in range(4):
            assert all(np.sum((folds[o] == f) & (labels[ix] == c) & (domains[ix] == d)) == 8
                       for d in (0, 1) for c in (0, 1))
    write(OUT/'plan.json', {'cases': cases, 'jobs': jobs, 'features': FEATURES, 'models': MODELS})
    np.savez_compressed(OUT/'design.npz', pools=pools, labels=labels, domains=domains, folds=folds)
    sources = [CONFIG, Path(__file__), ROOT/'cafl4ds/functional_health_profile.py',
        ROOT/'cafl4ds/accessibility_replication.py', ROOT/'cafl4ds/conditioning_fitting.py',
        ROOT/'cafl4ds/support_regularization.py', ROOT/'cafl4ds/preaction_benefit.py',
        ROOT/'scripts/report_functional_health_profile.py', ROOT/'scripts/engineering_functional_health_profile.py',
        ROOT/'scripts/verify_functional_health_profile.py',
        ROOT/'tests/unit/test_functional_health_profile.py', ROOT/'docs/project-plan/functional-health-profile-frozen.md', ROOT/'uv.lock']
    paths = sources+[MANIFEST, READERS/'design.npz', READERS/'plan.json', READERS/'fitting-ledger.json',
        READERS/'input_lock.json', OLD/'plan.json', OLD/'analysis-arrays.npz', OLD/'verification.json',
        OLD/'final-delivery-verification.json']+[OUT/n for n in ('plan.json', 'design.npz', 'tests.log', 'quality.log', 'engineering-verification.json')]
    prior = read(READERS/'input_lock.json')['hashes']; ledger = read(READERS/'fitting-ledger.json'); assert ledger['passed']
    assert digest(READERS/'design.npz') == prior[str(READERS/'design.npz')]
    oldhashes = read(OLD/'verification.json')['hashes']
    assert digest(OLD/'analysis-arrays.npz') == oldhashes[str(OLD/'analysis-arrays.npz')]
    for j in jobs:
        f = GEO/'features'/j['point']/'features.npz'; receipt = f.parent/'complete.json'
        assert digest(f) == read(receipt)['sha256']; paths += [f, receipt]
        h = READERS/'points'/j['point']/'heads.npz'
        assert digest(h) == ledger['hashes'][str(h)]; paths += [h]
    for p in sources:
        dest = OUT/'source_snapshot'/p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
    write(OUT/'input_lock.json', {'hashes': {str(p): digest(p) for p in set(paths)}, 'config': CFG,
        'before_new_real_fitting_or_profiles': True, 'frozen_unix': time.time(),
        'budget': {'collection': 3456, 'repeat': 72, 'independent': 72, 'total': 3600, 'SSL': 0, 'encoder_forwards': 0},
        'environment': {'python': sys.version, 'packages': {n: version(n) for n in ('numpy', 'scipy', 'torch', 'matplotlib', 'threadpoolctl')},
                        'dtype': 'float64', 'device': 'CPU', 'BLAS_threads': 4}})
    status('frozen', cases=72, total_fits=3600)


def profiles():
    lock = read(OUT/'input_lock.json'); check(lock['hashes']); reserve()
    plan = read(OUT/'plan.json'); hashes = {}; all_h = []; all_full = []; all_fold = []
    with np.load(OUT/'design.npz') as z:
        pools, labels, domains, folds = [z[k] for k in ('pools', 'labels', 'domains', 'folds')]
    journal = OUT/'fit-journal.jsonl'; assert not journal.exists(), 'Preserve interrupted fits; no silent retries'
    counts = {'collection': 0, 'repeat': 0, 'independent': 0}; receipts = []
    max_repeat = max_independent = 0.
    with journal.open('x', encoding='utf-8') as jf:
        def record(value):
            jf.write(json.dumps({'unix': time.time(), **value})+'\n'); jf.flush()
        for i, j in enumerate(plan['jobs']):
            reserve()
            with np.load(GEO/'features'/j['point']/'features.npz') as z:
                # Slice support immediately; no old query features enter profile construction.
                support = z['eval'][:200].astype(float)
            heads = np.empty((6, 2, 4, 97)); scores = np.empty((6, 2, 128))
            h = np.empty((6, 2, 8)); fold_h = np.empty((6, 2, 4, 8)); raw_boundary = np.empty((6, 2, 2))
            for o, ix in enumerate(pools):
                x, y, d = augment(support[ix]), labels[ix], domains[ix]
                for k, lam in enumerate(CFG['lambdas']):
                    for f in range(4):
                        tr, te = folds[o] != f, folds[o] == f
                        key = {'case': i, 'outer': o, 'lambda': lam, 'fold': f}
                        record({'phase': 'started', 'kind': 'collection', **key})
                        beta, receipt = fit(support[ix][tr], y[tr], lam); counts['collection'] += 1
                        record({'phase': 'complete', 'kind': 'collection', **key, **receipt})
                        receipts.append({**key, **receipt}); heads[o, k, f] = beta
                        scores[o, k, te] = x[te]@beta
                        fold_h[o, k, f] = profile(scores[o, k, te], y[te], d[te])
                        if o == k == f == 0:
                            record({'phase': 'started', 'kind': 'repeat', **key})
                            again, _ = fit(support[ix][tr], y[tr], lam); counts['repeat'] += 1
                            error = float(np.max(abs(again-beta))); max_repeat = max(max_repeat, error); assert error <= 1e-12
                            record({'phase': 'complete', 'kind': 'repeat', **key, 'coefficient_error': error})
                            record({'phase': 'started', 'kind': 'independent', **key})
                            xx, yy, penalty = x[tr], y[tr], np.eye(97)
                            def obj(b):
                                return objective(b, xx, yy, penalty, lam)
                            opt = minimize(lambda b: obj(b)[0], np.zeros(97), jac=lambda b: obj(b)[1],
                                hess=lambda b: obj(b)[2], method='trust-exact', options={'gtol': 1e-11, 'maxiter': 100})
                            counts['independent'] += 1
                            g = float(np.max(abs(obj(opt.x)[1])))
                            error = float(np.max(abs(augment(support)@(opt.x-beta))))
                            max_independent = max(max_independent, error); assert g <= 1e-8 and error <= 1e-6
                            record({'phase': 'complete', 'kind': 'independent', **key, 'gradient_inf': g, 'reservoir_score_error': error})
                    h[o, k] = profile(scores[o, k], y, d)
                    raw_boundary[o, k] = [np.mean(abs(scores[o, k, d == dom]) <= .1) for dom in (0, 1)]
            with np.load(READERS/'points'/j['point']/'heads.npz') as z:
                full_heads = z['heads'][0, :, 3]
            full = np.array([[profile(augment(support[ix])@full_heads[o, k], labels[ix], domains[ix])
                              for k in range(2)] for o, ix in enumerate(pools)])
            path = OUT/'profiles'/f'{i:02d}.npz'; path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, heads=heads, oof_scores=scores, profiles=h,
                                fold_profiles=fold_h, full_profiles=full, raw_boundary_mass=raw_boundary)
            hashes[str(path)] = digest(path); all_h.append(h); all_full.append(full); all_fold.append(fold_h)
            status('support-only-profiles', states_complete=i+1, total=72, fits=sum(counts.values()))
    assert counts == {'collection': 3456, 'repeat': 72, 'independent': 72}
    path = OUT/'profile-arrays.npz'
    np.savez_compressed(path, profiles=all_h, full_profiles=all_full, fold_profiles=all_fold)
    hashes[str(path)] = digest(path); hashes[str(journal)] = digest(journal)
    write(OUT/'fit-receipts.json', {'receipts': receipts}); hashes[str(OUT/'fit-receipts.json')] = digest(OUT/'fit-receipts.json')
    check(lock['hashes'])
    write(OUT/'profile-ledger.json', {'passed': True, 'hashes': hashes, 'counts': counts, 'finished_unix': time.time(),
        'all_profiles_sealed_before_target_loading': True, 'maximum_repeat_coefficient_error': max_repeat,
        'maximum_independent_reservoir_score_error': max_independent})


def evaluate():
    lock = read(OUT/'input_lock.json'); check(lock['hashes'])
    ledger = read(OUT/'profile-ledger.json'); assert ledger['passed']; check(ledger['hashes'])
    marker = OUT/'target-start.json'; assert not marker.exists(), 'Preserve first target evaluation'
    write(marker, {'unix': time.time(), 'profile_ledger_sha256': digest(OUT/'profile-ledger.json')})
    assert read(marker)['unix'] > ledger['finished_unix']
    with np.load(OUT/'profile-arrays.npz') as z:
        h, full, fold_h = [z[k] for k in ('profiles', 'full_profiles', 'fold_profiles')]
    with np.load(OLD/'analysis-arrays.npz') as z:
        target, signals, groups, diets = [z[k] for k in ('target', 'signals', 'groups', 'diets')]
        correct = z['correct']
        cells = np.stack([100*correct[..., c*512:(c+1)*512].mean(-1) for c in range(4)], -1)
        comps = np.concatenate([cells.mean(-1, keepdims=True), cells[..., :2].mean(-1, keepdims=True),
                                cells[..., 2:].mean(-1, keepdims=True), cells], -1)
        reconstructed = (comps[:, :, 2]-comps[:, :, 1]).mean(2)
        np.testing.assert_array_equal(target, reconstructed)
        oldcheap, oldcka = z['prediction_cheap'], z['prediction_health_added']
    cases = read(OUT/'plan.json')['cases']
    np.testing.assert_array_equal(groups, [[c[k] for k in ('seed', 'source', 'context')] for c in cases])
    np.testing.assert_array_equal(diets, [int(c['pre_diet'] == 'balanced') for c in cases])
    panels = [('mean_lambda01', h[:, :, 0].mean(1)), ('mean_lambda001', h[:, :, 1].mean(1))]
    panels += [(f'pool{o}_lambda01', h[:, o, 0]) for o in range(6)]
    all_pred = []; error_rows = []; screens = {}; max_error = 0.; shadow = []
    subsets = [('all', 0, np.ones(72, bool))]
    for axis, name in enumerate(('seed', 'source', 'context')):
        subsets += [(name, int(v), groups[:, axis] == v) for v in np.unique(groups[:, axis])]
    subsets += [('diet', int(v), diets == v) for v in (0, 1)]
    for panel, hh in panels:
        pred, folds, error = forecast(signals, diets, groups, hh, target.reshape(72, 14))
        max_error = max(error, max_error); pred = {m: p.reshape(72, 2, 7) for m, p in pred.items()}
        np.testing.assert_allclose(pred['cheap'], oldcheap, atol=1e-10, rtol=0)
        np.testing.assert_allclose(pred['cka_added'], oldcka, atol=1e-10, rtol=0)
        all_pred.append(np.stack([pred[m] for m in MODELS], 1))
        screens[panel] = screen(target[:, 1, 6], {m: p[:, 1, 6] for m, p in pred.items()}, groups)
        for m in MODELS:
            for p, proc in enumerate(('joint', 'fixed')):
                for k, comp in enumerate(COMPONENTS):
                    for subset, value, keep in subsets:
                        error_rows.append({'panel': panel, 'model': m, 'procedure': proc, 'component': comp,
                            'subset': subset, 'value': value, 'n_states': int(keep.sum()),
                            'MSE': float(np.mean((pred[m][keep, p, k]-target[keep, p, k])**2))})
        if panel == 'mean_lambda01':
            for m in ('always_current', 'always_replay', 'cheap', 'functional_added', 'cka_functional_added'):
                action = np.zeros(72, bool) if m == 'always_current' else np.ones(72, bool) if m == 'always_replay' else pred[m][:, 1, 6] > 0
                for k, comp in enumerate(COMPONENTS):
                    shadow.append({'model': m, 'component': comp, 'replay_states': int(action.sum()),
                                   'value_relative_current_pp': float(np.mean(target[:, 1, k]*action))})
    correlations = []
    for diet in (0, 1):
        keep = diets == diet
        for k, name in enumerate(FEATURES):
            correlations.append({'diet': diet, 'feature': name, 'n_states': int(keep.sum()),
                'pearson_r': float(np.corrcoef(h[keep, :, 0, k].mean(1), target[keep, 1, 6])[0, 1])})
    arrays = OUT/'analysis-arrays.npz'
    np.savez_compressed(arrays, profiles=h, full_profiles=full, fold_profiles=fold_h,
        target=target, groups=groups, diets=diets, predictions=all_pred)
    csv_write(OUT/'errors.csv', error_rows); csv_write(OUT/'shadow-values.csv', shadow); csv_write(OUT/'correlations.csv', correlations)
    write(OUT/'folds.json', {'folds': folds, 'all_three_axes_excluded': True})
    primary_errors = [r for r in error_rows if r['panel'] == 'mean_lambda01' and r['procedure'] == 'fixed'
                      and r['component'] == 'night_highway' and r['subset'] == 'all']
    summary = {'states': 72, 'profiles': 864, 'collection_fits': 3456, 'total_reader_fits': 3600,
        'new_SSL': 0, 'new_encoder_forwards': 0, 'strict_holdout_folds': 36, 'forecast_panels': 8,
        'primary_MSE': {r['model']: r['MSE'] for r in primary_errors}, 'screens': screens,
        'maximum_independent_ridge_error': max_error, 'old_cheap_and_CKA_forecasts_reproduced': True,
        'primary_heldout_reader_training_images': 96, 'profile_support_images': 128,
        'outer_pool_SD_mean_by_feature': h[:, :, 0].std(1).mean(0).tolist(),
        'OOF_minus_full_mean_by_feature': (h[:, :, 0]-full[:, :, 0]).mean((0, 1)).tolist(),
        'known_cohort_retrospective_development_only': True, 'no_alarm_or_controller_qualified': True}
    write(OUT/'summary.json', summary); check(lock['hashes']); check(ledger['hashes'])
    paths = [arrays]+[OUT/n for n in ('errors.csv', 'shadow-values.csv', 'correlations.csv', 'folds.json', 'summary.json', 'target-start.json')]
    write(OUT/'evaluation-verification.json', {'passed': True, 'hashes': {str(p): digest(p) for p in paths},
        'original_targets_exactly_reconstructed': True, 'legacy_baselines_reproduced': True,
        'forecast_ridge_fits': 8*36*8, 'all_forecasts_independently_verified': True})
    status('numerically-complete', total_reader_fits=3600)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('freeze', 'profiles', 'evaluate', 'run'))
    stage = parser.parse_args().stage
    with threadpool_limits(limits=4):
        if stage == 'run':
            profiles(); evaluate()
            from scripts.report_functional_health_profile import make_report
            write(OUT/'delivery-verification.json', make_report(OUT, ROOT/CFG['notebook']))
            status('complete-awaiting-visual-QA')
        else:
            {'freeze': freeze, 'profiles': profiles, 'evaluate': evaluate}[stage]()


if __name__ == '__main__':
    main()
