"""Exact known-state reconstruction, bounded model-copy probes, sealed responses."""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from cafl4ds.content_step_intervention import displacement, set_encoder_step
from cafl4ds.domain_health_calibration import schedule
from cafl4ds.functional_response_profile import MODELS, forecast, response, screen
from cafl4ds.health_calibration import embeddings
from cafl4ds.natural_health_rate import OrderedFrames
from cafl4ds.reservoir_action_replication import capture, restore
from cafl4ds.update_filtering import optimize_current
from scripts.run_adaptation_experiment import _method
from scripts.run_functional_health_profile import ROOT, check, csv_write, digest, read, write
from scripts.run_paired_update_response import fingerprint
from scripts.run_retention_anchor_signal import setup

CONFIG = ROOT/'cafl4ds/configs/functional_response_profile.json'; CFG = read(CONFIG)
OUT = ROOT/CFG['output']; PREV = ROOT/CFG['previous']; OLD = ROOT/CFG['outcomes']; GEO = ROOT/CFG['geometry']


def reserve():
    assert shutil.disk_usage(OUT).free > CFG['minimum_free_mib']*2**20
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) < CFG['max_output_mib']*2**20


def status(stage, **kw):
    v = {'stage': stage, 'pid': os.getpid(), 'unix': time.time(), **kw}; write(OUT/'status.json', v); print(v, flush=True)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True); reserve(); assert not (OUT/'input_lock.json').exists()
    eng = read(OUT/'engineering-verification.json'); assert eng['passed'] and not eng['visual_QA_pending']; check(eng['hashes'])
    check(read(PREV/'final-delivery-verification.json')['hashes']); check(read(PREV/'input_lock.json')['hashes'])
    oldlock = read(OLD/'input_lock.json'); plan = read(OLD/'plan.json')
    support_images = torch.load(ROOT/CFG['images'], weights_only=True, map_location='cpu')[:200].clone()
    torch.save(support_images, OUT/'support-images.pt')
    write(OUT/'plan.json', {'cases': plan['cases'], 'contexts': plan['contexts'], 'model_config': oldlock['model_config'],
                          'models': MODELS, 'magnitudes': CFG['magnitudes']})
    paths = [CONFIG, OUT/'plan.json', PREV/'input_lock.json', PREV/'final-delivery-verification.json',
        PREV/'design.npz', PREV/'analysis-arrays.npz', PREV/'profile-ledger.json', OLD/'input_lock.json',
        OLD/'analysis-arrays.npz', ROOT/CFG['images'], OUT/'support-images.pt', ROOT/'uv.lock', Path(oldlock['model_config']['well'])]
    paths += [ROOT/n for n in ('scripts/run_functional_response_profile.py', 'scripts/report_functional_response_profile.py',
        'scripts/engineering_functional_response_profile.py', 'scripts/verify_functional_response_profile.py',
        'cafl4ds/functional_response_profile.py', 'tests/unit/test_functional_response_profile.py',
        'docs/project-plan/functional-response-profile-frozen.md')]
    paths += [OUT/n for n in ('tests.log', 'quality.log', 'engineering-verification.json')]
    for c in plan['cases']:
        for p in (OLD/'histories'/c['history']/'state.pt', OLD/'cases'/c['id']/'record.json', OLD/'cases'/c['id']/'trace.npz'):
            assert digest(p) == read(OLD/'verification.json')['hashes'][str(p)]; paths.append(p)
        paths += [PREV/'profiles'/f'{plan["cases"].index(c):02d}.npz', GEO/'features'/f"s43/{c['id']}/fork/features.npz"]
        paths += [Path(c['base_encoder']), OLD/'branches'/c['id']/'fork/encoder-xor.npz']
    for context in plan['contexts']:
        for r in context['clips']:
            p = Path(r['frames_path']); assert digest(p) == r['frames_sha256']; paths.append(p)
    paths += [Path(m.__file__) for n, m in sys.modules.items() if n.startswith(('scripts.', 'cafl4ds.'))
              and (getattr(m, '__file__', '') or '').endswith('.py')]
    hashes = {str(p.resolve()): digest(p) for p in set(paths)}
    inherited = oldlock['hashes'] | read(PREV/'input_lock.json')['hashes']
    for p, h in hashes.items():
        if p in inherited:
            assert inherited[p] == h, p
    for p in set(paths):
        if p.suffix in ('.py', '.json', '.md') and not p.is_relative_to(ROOT/'outputs'):
            dest = OUT/'source_snapshot'/p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
    write(OUT/'input_lock.json', {'hashes': hashes, 'config': CFG, 'frozen_unix': time.time(),
        'before_real_reconstruction_probes_or_inference': True, 'device': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'TF32': False, 'threads': 4})
    status('frozen', SSL_budget=9504, encoder_passes=720, inputs=len(hashes))


def collect():
    lock = read(OUT/'input_lock.json'); check(lock['hashes']); reserve(); plan = read(OUT/'plan.json')
    method = OmegaConf.create(plan['model_config']); template = _method(method).cuda()
    images = torch.load(OUT/'support-images.pt', weights_only=True, map_location='cpu'); assert len(images) == 200
    with np.load(PREV/'design.npz') as z:
        pools, labels, domains, folds = [z[k] for k in ('pools', 'labels', 'domains', 'folds')]
    wp = OUT/'work.json'
    work = read(wp) if wp.exists() else {'SSL_started': 0, 'SSL_completed': 0, 'passes_started': 0, 'passes_completed': 0, 'in_progress': None}
    assert work['in_progress'] is None and work['SSL_started'] == work['SSL_completed'] and work['passes_started'] == work['passes_completed'], 'Preserve interrupted work'
    hashes = {}; all_response = []
    def counted_update(model, optimizer, batch):
        assert work['SSL_started'] < CFG['maximum_SSL']; work['SSL_started'] += 1; write(wp, work)
        metrics = optimize_current(model, optimizer, torch.from_numpy(batch).permute(0, 3, 1, 2).float().cuda()/255, True)
        work['SSL_completed'] += 1; write(wp, work); return metrics
    def infer(model):
        first = None
        for repeat in (False, True):
            assert work['passes_started'] < 720; work['passes_started'] += 1; write(wp, work)
            value = embeddings(model, images)
            if repeat:
                np.testing.assert_array_equal(value, first)
            else:
                first = value
            work['passes_completed'] += 1; write(wp, work)
        return first
    for i, c in enumerate(plan['cases']):
        directory = OUT/'states'/f'{i:02d}'; done = directory/'complete.json'
        if done.exists():
            receipt = read(done); check(receipt['hashes']); hashes.update(receipt['hashes']); hashes[str(done)] = digest(done)
            with np.load(directory/'responses.npz') as z:
                all_response.append(z['response'])
            continue
        directory.mkdir(parents=True, exist_ok=True); reserve(); work['in_progress'] = i; write(wp, work)
        start_ssl, start_pass = work['SSL_completed'], work['passes_completed']
        saved = torch.load(OLD/'histories'/c['history']/'state.pt', weights_only=True, map_location='cpu')
        model, optimizer = restore(template, method.optim, saved['learner'])
        torch.manual_seed(c['pre_seed']); torch.cuda.manual_seed_all(c['pre_seed'])
        stream = OrderedFrames(plan['contexts'][c['context']-1]['clips'])
        pool = saved['memory']['images'].numpy(); ids = saved['memory']['ids']; positions = {v: j for j, v in enumerate(ids)}
        diet = schedule(ids, c['pre_diet'], 512)
        with np.load(OLD/'cases'/c['id']/'trace.npz') as z:
            old_metrics, old_hashes = z['metrics'], z['hashes']
        params = [p for p in model.encoder.parameters() if p.requires_grad]
        for step in range(128):
            now = stream.batch(step); picked = diet[step]
            mixed = now if c['pre_diet'] == 'current' else np.concatenate([now[picked['current']], pool[[positions[v] for v in picked['past']]]])
            before = [p.detach().clone() for p in params]
            metrics = counted_update(model, optimizer, mixed); norm = displacement(params, before)
            row = [*metrics, norm, norm, norm, 1.]
            np.testing.assert_array_equal(row, old_metrics[step])
            actual_hash = [hashlib.sha256(now.tobytes()).hexdigest(), hashlib.sha256(mixed.tobytes()).hexdigest(), fingerprint(capture(model, optimizer))]
            np.testing.assert_array_equal(np.array(actual_hash, dtype='S64'), old_hashes[step])
        fork = capture(model, optimizer); record = read(OLD/'cases'/c['id']/'record.json')
        assert fingerprint({'learner': fork, 'memory': saved['memory']}) == record['fork_fingerprint']
        base_features = infer(model)
        with np.load(GEO/'features'/f"s43/{c['id']}/fork/features.npz") as z:
            np.testing.assert_array_equal(base_features, z['eval'][:200])
        target_norm = float(old_metrics[-16:, 3].mean()); assert target_norm == record['norm_target']
        now = stream.batch(CFG['probe_observed_batch']); assert hashlib.sha256(now.tobytes()).hexdigest().encode() == old_hashes[127, 0]
        picked = schedule(ids, 'balanced', 512)[127]
        batches = [now, np.concatenate([now[picked['current']], pool[[positions[v] for v in picked['past']]]])]
        features = np.empty((2, 2, 200, 96), dtype=np.float32); raw_deltas = []; proposal_rows = []; dose_rows = []
        for action, batch in enumerate(batches):
            proposal = None
            for repeat in (False, True):
                local, opt = restore(template, method.optim, fork)
                torch.manual_seed(CFG['probe_seed_offset']+i); torch.cuda.manual_seed_all(CFG['probe_seed_offset']+i)
                parameters = [p for p in local.encoder.parameters() if p.requires_grad]
                before = [p.detach().clone() for p in parameters]
                metrics = counted_update(local, opt, batch)
                fp = fingerprint(capture(local, opt))
                if repeat:
                    assert fp == proposal_rows[-1]['full_fingerprint']; np.testing.assert_array_equal(metrics, proposal_rows[-1]['metrics'])
                    del local, opt
                else:
                    proposal = (local, opt, parameters, before)
                    proposal_rows.append({'action': action, 'metrics': list(metrics), 'full_fingerprint': fp,
                        'encoder_fingerprint': fingerprint(local.encoder.state_dict()), 'batch_sha256': hashlib.sha256(batch.tobytes()).hexdigest()})
            local, opt, parameters, before = proposal
            proposed = [p.detach().clone() for p in parameters]
            delta = torch.cat([(p.double()-b.double()).flatten() for p, b in zip(proposed, before, strict=True)]).cpu().numpy()
            raw_deltas.append(delta)
            for m, scale in enumerate(CFG['magnitudes']):
                with torch.no_grad():
                    for p, value in zip(parameters, proposed, strict=True):
                        p.copy_(value)
                raw, actual, factor = set_encoder_step(parameters, before, scale*target_norm)
                realized = torch.cat([(p.detach().double()-b.double()).flatten() for p, b in zip(parameters, before, strict=True)]).cpu().numpy()
                relative_error = float(np.linalg.norm(realized-factor*delta)/actual)
                assert relative_error < 1e-3
                dose_rows.append({'action': action, 'magnitude': scale, 'raw_norm': raw, 'actual_norm': actual,
                    'target_norm': scale*target_norm, 'factor': factor, 'relative_rounding_error': relative_error})
                features[action, m] = infer(local)
            del local, opt, proposal
        vectors = np.array(raw_deltas); cosine = float(vectors[0]@vectors[1]/(np.linalg.norm(vectors[0])*np.linalg.norm(vectors[1])))
        with np.load(PREV/'profiles'/f'{i:02d}.npz') as z:
            heads, old_scores = z['heads'], z['oof_scores']
        measured = np.empty((2, 2, 6, 2, 8)); score_error = 0.
        for o, ix in enumerate(pools):
            y, d = labels[ix], domains[ix]
            for k in range(2):
                before_score = np.empty(128); after_score = np.empty((2, 2, 128))
                for f in range(4):
                    te = folds[o] == f; beta = heads[o, k, f]
                    before_score[te] = base_features[ix[te]].astype(float)@beta[:-1]+beta[-1]
                    after_score[:, :, te] = features[:, :, ix[te]].astype(float)@beta[:-1]+beta[-1]
                score_error = max(score_error, float(np.max(abs(before_score-old_scores[o, k])))); assert score_error < 1e-12
                for action in range(2):
                    for m in range(2):
                        measured[action, m, o, k] = response(before_score, after_score[action, m], y, d)
        np.savez_compressed(directory/'features.npz', before=base_features, after=features)
        np.savez_compressed(directory/'directions.npz', raw_deltas=vectors)
        np.savez_compressed(directory/'responses.npz', response=measured)
        write(directory/'audit.json', {'passed': True, 'all128_steps_and_fork_fingerprint_exact': True,
            'zero_features_match_cached_and_all_passes_repeat': True, 'case': c['id'],
            'physical_SSL': work['SSL_completed']-start_ssl, 'encoder_passes': work['passes_completed']-start_pass,
            'proposal_repeats_exact': True, 'proposals': proposal_rows, 'doses': dose_rows, 'causal_norm': target_norm,
            'parameter_names_shapes': [(n, list(p.shape)) for n, p in model.encoder.named_parameters() if p.requires_grad],
            'replay_current_direction_cosine': cosine, 'maximum_baseline_score_error': score_error})
        paths = [directory/n for n in ('features.npz', 'directions.npz', 'responses.npz', 'audit.json')]
        local_hashes = {str(p): digest(p) for p in paths}; write(done, {'passed': True, 'hashes': local_hashes})
        hashes.update(local_hashes); hashes[str(done)] = digest(done); all_response.append(measured)
        work['in_progress'] = None; write(wp, work)
        status('collecting-responses', states_complete=i+1, total=72, SSL=work['SSL_completed'], passes=work['passes_completed'])
        del model, optimizer, fork, saved; torch.cuda.empty_cache()
    assert work['SSL_completed'] == 9504 and work['passes_completed'] == 720
    path = OUT/'response-arrays.npz'; np.savez_compressed(path, response=all_response); hashes[str(path)] = digest(path)
    audit_rows = []
    for i in range(72):
        audit = read(OUT/'states'/f'{i:02d}'/'audit.json')
        audit_rows.append({'case': i, 'cosine': audit['replay_current_direction_cosine'],
            'maximum_rounding_error': max(r['relative_rounding_error'] for r in audit['doses']),
            'all_checks_passed': audit['passed']})
    write(OUT/'audit-summary.json', {'states': audit_rows}); hashes[str(OUT/'audit-summary.json')] = digest(OUT/'audit-summary.json')
    hashes[str(wp)] = digest(wp); check(lock['hashes'])
    write(OUT/'response-ledger.json', {'passed': True, 'hashes': hashes, 'finished_unix': time.time(),
        'all_responses_sealed_before_target_loading': True, 'physical_SSL': 9504, 'encoder_passes': 720, 'image_forwards': 144000})
    status('responses-sealed', states_complete=72)


def evaluate():
    lock = read(OUT/'input_lock.json'); check(lock['hashes']); ledger = read(OUT/'response-ledger.json'); check(ledger['hashes'])
    assert not (OUT/'target-start.json').exists(); write(OUT/'target-start.json', {'unix': time.time()})
    with np.load(OUT/'response-arrays.npz') as z:
        action = z['response']
    delta = action[:, 1]-action[:, 0]  # state,dose,pool,prior,8
    with np.load(PREV/'analysis-arrays.npz') as z:
        h, target, groups, diets, old_predictions = [z[k] for k in ('profiles', 'target', 'groups', 'diets', 'predictions')]
    with np.load(OLD/'analysis-arrays.npz') as z:
        signals = z['signals']; np.testing.assert_array_equal(target, z['target'])
    panels = [(f'mean_dose{m}_prior{k}', delta[:, m, :, k].mean(1), h[:, :, k].mean(1)) for m, k in ((1, 0), (0, 0), (1, 1), (0, 1))]
    panels += [(f'pool{o}_dose1_prior0', delta[:, 1, o, 0], h[:, o, 0]) for o in range(6)]
    all_pred = []; screens = {}; errors = []; maximum = 0.; shadows = []; correlations = []
    subsets = [('all', 0, np.ones(72, bool))]
    for axis, name in enumerate(('seed', 'source', 'context')):
        subsets += [(name, int(v), groups[:, axis] == v) for v in np.unique(groups[:, axis])]
    subsets += [('diet', int(v), diets == v) for v in (0, 1)]
    for panel, rr, static in panels:
        pred, folds, error = forecast(signals, diets, groups, static, rr, target.reshape(72, 14))
        maximum = max(maximum, error); pp = np.stack([pred[m].reshape(72, 2, 7) for m in MODELS], 1); all_pred.append(pp)
        np.testing.assert_allclose(pp[:, 1], old_predictions[0, :, 2], atol=1e-10, rtol=0)
        if panel == 'mean_dose1_prior0':
            np.testing.assert_allclose(pp[:, 2], old_predictions[0, :, 5], atol=1e-10, rtol=0)
        screens[panel] = screen(target[:, 1, 6], {m: pp[:, j, 1, 6] for j, m in enumerate(MODELS)}, groups)
        for j, model in enumerate(MODELS):
            for p in range(2):
                for k in range(7):
                    for subset, value, keep in subsets:
                        errors.append({'panel': panel, 'model': model, 'procedure': p, 'component': k, 'subset': subset,
                            'value': value, 'MSE': float(np.mean((pp[keep, j, p, k]-target[keep, p, k])**2))})
        if panel == 'mean_dose1_prior0':
            for model in ('always_current', 'always_replay', 'cheap', 'static', 'static_response_added'):
                take = np.zeros(72, bool) if model == 'always_current' else np.ones(72, bool) if model == 'always_replay' else pp[:, MODELS.index(model), 1, 6] > 0
                for k in range(7):
                    shadows.append({'model': model, 'component': k, 'replay_states': int(take.sum()), 'value_pp': float(np.mean(target[:, 1, k]*take))})
            for diet in (0, 1):
                keep = diets == diet
                for k in range(8):
                    correlations.append({'diet': diet, 'axis': k, 'r': float(np.corrcoef(rr[keep, k], target[keep, 1, 6])[0, 1])})
    np.savez_compressed(OUT/'analysis-arrays.npz', response=action, contrast=delta, profiles=h, target=target,
        groups=groups, diets=diets, predictions=all_pred)
    csv_write(OUT/'errors.csv', errors); csv_write(OUT/'shadow-values.csv', shadows); csv_write(OUT/'correlations.csv', correlations)
    write(OUT/'folds.json', {'folds': folds}); pp = all_pred[0]
    write(OUT/'summary.json', {'states': 72, 'physical_SSL': 9504, 'reconstruction_SSL': 9216, 'proposal_SSL': 144,
        'repeat_SSL': 144, 'reader_fits': 0, 'forecast_ridge_fits': 2520, 'encoder_passes': 720, 'image_forwards': 144000,
        'primary_MSE': {m: float(np.mean((pp[:, j, 1, 6]-target[:, 1, 6])**2)) for j, m in enumerate(MODELS)},
        'screens': screens, 'maximum_independent_ridge_error': maximum, 'no_controller_qualified': True,
        'known_cohort_retrospective_development_only': True})
    paths = [OUT/n for n in ('analysis-arrays.npz', 'summary.json', 'errors.csv', 'shadow-values.csv', 'correlations.csv', 'folds.json', 'target-start.json')]
    check(lock['hashes']); check(ledger['hashes'])
    write(OUT/'evaluation-verification.json', {'passed': True, 'hashes': {str(p): digest(p) for p in paths},
        'all_2520_ridge_fits_independently_verified': True, 'cheap_and_primary_static_forecasts_reproduced': True})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('freeze', 'collect', 'evaluate', 'run'))
    stage = parser.parse_args().stage; setup()
    with threadpool_limits(limits=4):
        if stage == 'run':
            collect(); evaluate()
            from scripts.report_functional_response_profile import make_report
            write(OUT/'delivery-verification.json', make_report(OUT, ROOT/CFG['notebook']))
            status('numerically-complete-awaiting-QA')
        else:
            {'freeze': freeze, 'collect': collect, 'evaluate': evaluate}[stage]()


if __name__ == '__main__':
    main()
