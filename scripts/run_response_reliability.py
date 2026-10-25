"""Repeated pre-action probes sealed before original384-update benefit evaluation."""
from __future__ import annotations

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

import scripts.run_functional_response_profile as base
from cafl4ds.functional_response_profile import MODELS, forecast, response, screen
from cafl4ds.response_reliability import BATCHES, MASK_OFFSETS, grid_variance, panels
from scripts.continue_functional_response_profile_padded import padded_embeddings

ROOT = base.ROOT; CONFIG = ROOT/'cafl4ds/configs/response_reliability.json'; CFG = base.read(CONFIG)
OUT = ROOT/CFG['output']; PRIOR = ROOT/CFG['previous']; PROFILE = ROOT/CFG['profiles']; OLD = ROOT/CFG['outcomes']; GEO = ROOT/CFG['geometry']
read, write, digest, check, csv_write = base.read, base.write, base.digest, base.check, base.csv_write


def reserve():
    assert shutil.disk_usage(OUT).free > CFG['minimum_free_mib']*2**20, 'Preserve1536MiB reserve'
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) < CFG['max_output_mib']*2**20


def status(stage, **kw):
    v = {'stage': stage, 'pid': __import__('os').getpid(), 'unix': time.time(), **kw}; write(OUT/'status.json', v); print(v, flush=True)


def measure(before, after, heads, pools, labels, domains, folds):
    result = np.empty((3, 3, 2, 6, 2, 8))
    for o, ids in enumerate(pools):
        y, d = labels[ids], domains[ids]
        for k in range(2):
            a = np.empty(128); b = np.empty((3, 3, 2, 128))
            for f in range(4):
                te = folds[o] == f; beta = heads[o, k, f]
                a[te] = before[ids[te]].astype(float)@beta[:-1]+beta[-1]
                b[..., te] = after[..., ids[te], :].astype(float)@beta[:-1]+beta[-1]
            for batch, mask, action in np.ndindex(3, 3, 2):
                result[batch, mask, action, o, k] = response(a, b[batch, mask, action], y, d)
    return result


def freeze():
    OUT.mkdir(parents=True, exist_ok=True); reserve(); assert not (OUT/'input_lock.json').exists()
    eng = read(OUT/'engineering-verification.json'); assert eng['passed'] and not eng['visual_QA_pending']; check(eng['hashes'])
    prior = read(PRIOR/'input_lock.json'); check(prior['hashes']); check(read(PRIOR/'padding-amendment-lock.json')['hashes'])
    check(read(PRIOR/'final-delivery-verification.json')['hashes'])
    plan = read(PRIOR/'plan.json'); write(OUT/'plan.json', {**plan, 'batches': BATCHES, 'mask_offsets': MASK_OFFSETS})
    paths = [CONFIG, OUT/'plan.json', PRIOR/'input_lock.json', PRIOR/'padding-amendment-lock.json', PRIOR/'final-delivery-verification.json',
        PRIOR/'support-images.pt', PRIOR/'analysis-arrays.npz', PROFILE/'analysis-arrays.npz', PROFILE/'design.npz', OLD/'analysis-arrays.npz', ROOT/'uv.lock']
    paths += [ROOT/n for n in ('scripts/run_response_reliability.py', 'scripts/report_response_reliability.py',
        'scripts/engineering_response_reliability.py', 'scripts/verify_response_reliability.py', 'cafl4ds/response_reliability.py',
        'tests/unit/test_response_reliability.py', 'docs/project-plan/response-reliability-frozen.md')]
    paths += [OUT/n for n in ('tests.log', 'quality.log', 'engineering-verification.json')]
    paths += [Path(p) for p in prior['hashes']]
    for i, c in enumerate(plan['cases']):
        paths += [PRIOR/'states'/f'{i:02d}'/n for n in ('features.npz', 'responses.npz', 'audit.json')]
        paths += [PROFILE/'profiles'/f'{i:02d}.npz', OLD/'cases'/c['id']/'record.json', OLD/'cases'/c['id']/'trace.npz',
                  OLD/'histories'/c['history']/'state.pt', Path(c['base_encoder']), OLD/'branches'/c['id']/'fork/encoder-xor.npz']
    paths += [Path(m.__file__) for n, m in sys.modules.items() if n.startswith(('cafl4ds.', 'scripts.'))
              and (getattr(m, '__file__', '') or '').endswith('.py')]
    hashes = {str(p.resolve()): digest(p) for p in set(paths)}
    inherited = prior['hashes'] | read(PROFILE/'input_lock.json')['hashes'] | read(PRIOR/'response-ledger.json')['hashes']
    for p, h in hashes.items():
        if p in inherited:
            assert inherited[p] == h, p
    for p in set(paths):
        if p.suffix in ('.py', '.md', '.json') and not p.is_relative_to(ROOT/'outputs'):
            dest = OUT/'source_snapshot'/p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
    assert shutil.disk_usage(OUT).free > (CFG['minimum_free_mib']+CFG['max_output_mib'])*2**20
    write(OUT/'input_lock.json', {'hashes': hashes, 'config': CFG, 'frozen_unix': time.time(),
        'before_real_reconstruction_probes_or_new_responses': True, 'device': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'TF32': False, 'threads': 4})
    status('frozen', states=72, SSL_budget=10656, passes=2736, inputs=len(hashes))


def collect():
    lock = read(OUT/'input_lock.json'); check(lock['hashes']); reserve(); plan = read(OUT/'plan.json')
    method = OmegaConf.create(plan['model_config']); template = base._method(method).cuda()
    images = torch.load(PRIOR/'support-images.pt', weights_only=True, map_location='cpu'); assert len(images) == 200
    with np.load(PROFILE/'design.npz') as z:
        pools, labels, domains, folds = [z[k] for k in ('pools', 'labels', 'domains', 'folds')]
    wp = OUT/'work.json'; work = read(wp) if wp.exists() else {'SSL_started': 0, 'SSL_completed': 0, 'passes_started': 0, 'passes_completed': 0, 'in_progress': None}
    assert work['in_progress'] is None and work['SSL_started'] == work['SSL_completed'] and work['passes_started'] == work['passes_completed'], 'Preserve interrupted work'
    hashes = {}; all_response = []
    def update(model, optimizer, batch):
        assert work['SSL_started'] < CFG['maximum_SSL']; work['SSL_started'] += 1; write(wp, work)
        value = base.optimize_current(model, optimizer, torch.from_numpy(batch).permute(0, 3, 1, 2).float().cuda()/255, True)
        work['SSL_completed'] += 1; write(wp, work); return value
    def infer(model):
        first = None
        for repeat in (False, True):
            assert work['passes_started'] < CFG['encoder_passes']; work['passes_started'] += 1; write(wp, work)
            value = padded_embeddings(model, images)
            if repeat:
                np.testing.assert_array_equal(value, first)
            else:
                first = value
            work['passes_completed'] += 1; write(wp, work)
        return first
    for i, c in enumerate(plan['cases']):
        directory = OUT/'states'/f'{i:02d}'; done = directory/'complete.json'
        if done.exists():
            r = read(done); check(r['hashes']); hashes.update(r['hashes']); hashes[str(done)] = digest(done)
            with np.load(directory/'responses.npz') as z:
                all_response.append(z['response'])
            continue
        reserve(); directory.mkdir(parents=True, exist_ok=True); work['in_progress'] = i; write(wp, work)
        start_ssl, start_pass = work['SSL_completed'], work['passes_completed']
        saved = torch.load(OLD/'histories'/c['history']/'state.pt', weights_only=True, map_location='cpu')
        model, optimizer = base.restore(template, method.optim, saved['learner'])
        torch.manual_seed(c['pre_seed']); torch.cuda.manual_seed_all(c['pre_seed'])
        stream = base.OrderedFrames(plan['contexts'][c['context']-1]['clips'])
        pool = saved['memory']['images'].numpy(); ids = saved['memory']['ids']; positions = {v: j for j, v in enumerate(ids)}
        diet = base.schedule(ids, c['pre_diet'], 512)
        with np.load(OLD/'cases'/c['id']/'trace.npz') as z:
            old_metrics, old_hashes = z['metrics'], z['hashes']
        params = [p for p in model.encoder.parameters() if p.requires_grad]
        for step in range(128):
            now = stream.batch(step); picked = diet[step]
            mixed = now if c['pre_diet'] == 'current' else np.concatenate([now[picked['current']], pool[[positions[v] for v in picked['past']]]])
            before = [p.detach().clone() for p in params]; value = update(model, optimizer, mixed); norm = base.displacement(params, before)
            np.testing.assert_array_equal([*value, norm, norm, norm, 1.], old_metrics[step])
            actual = [hashlib.sha256(now.tobytes()).hexdigest(), hashlib.sha256(mixed.tobytes()).hexdigest(), base.fingerprint(base.capture(model, optimizer))]
            np.testing.assert_array_equal(np.array(actual, dtype='S64'), old_hashes[step])
        fork = base.capture(model, optimizer); record = read(OLD/'cases'/c['id']/'record.json')
        assert base.fingerprint({'learner': fork, 'memory': saved['memory']}) == record['fork_fingerprint']
        baseline = infer(model)
        with np.load(PRIOR/'states'/f'{i:02d}'/'features.npz') as z:
            np.testing.assert_array_equal(baseline, z['before']); old_anchor_features = z['after'][:, 1]
        old_anchor_audit = read(PRIOR/'states'/f'{i:02d}'/'audit.json')
        causal_norm = float(old_metrics[-16:, 3].mean()); assert causal_norm == record['norm_target']
        features = np.empty((3, 3, 2, 200, 96), dtype=np.float32); proposals = []; audit_vectors = []; cosines = np.empty((3, 3))
        replay_schedule = base.schedule(ids, 'balanced', 512)
        for b, batch_index in enumerate(BATCHES):
            now = stream.batch(batch_index); assert hashlib.sha256(now.tobytes()).hexdigest().encode() == old_hashes[batch_index, 0]
            picked = replay_schedule[batch_index]
            batches = [now, np.concatenate([now[picked['current']], pool[[positions[v] for v in picked['past']]]])]
            for m, offset in enumerate(MASK_OFFSETS):
                pair = []
                for action, batch in enumerate(batches):
                    local, opt = base.restore(template, method.optim, fork)
                    torch.manual_seed(offset+i); torch.cuda.manual_seed_all(offset+i)
                    parameters = [p for p in local.encoder.parameters() if p.requires_grad]; before = [p.detach().clone() for p in parameters]
                    metrics = update(local, opt, batch); fp = base.fingerprint(base.capture(local, opt)); encoder_fp = base.fingerprint(local.encoder.state_dict())
                    vector = torch.cat([(p.detach().double()-a.double()).flatten() for p, a in zip(parameters, before, strict=True)]).cpu().numpy()
                    pair.append(vector)
                    if b == m == 1:
                        again, again_opt = base.restore(template, method.optim, fork)
                        torch.manual_seed(offset+i); torch.cuda.manual_seed_all(offset+i)
                        repeat_metrics = update(again, again_opt, batch)
                        assert base.fingerprint(base.capture(again, again_opt)) == fp; np.testing.assert_array_equal(metrics, repeat_metrics)
                        audit_vectors.append(vector); del again, again_opt
                    if (b, m) == (2, 0):
                        assert fp == old_anchor_audit['proposals'][action]['full_fingerprint']
                    raw, actual, factor = base.set_encoder_step(parameters, before, CFG['magnitude']*causal_norm)
                    realized = torch.cat([(p.detach().double()-a.double()).flatten() for p, a in zip(parameters, before, strict=True)]).cpu().numpy()
                    error = float(np.linalg.norm(realized-factor*vector)/actual); assert error < 1e-3
                    features[b, m, action] = infer(local)
                    proposals.append({'batch': b, 'mask': m, 'action': action, 'mask_seed': offset+i, 'metrics': list(metrics),
                        'batch_sha256': hashlib.sha256(batch.tobytes()).hexdigest(), 'full_fingerprint': fp, 'encoder_fingerprint': encoder_fp,
                        'raw_norm': raw, 'actual_norm': actual, 'target_norm': CFG['magnitude']*causal_norm, 'factor': factor,
                        'relative_rounding_error': error, 'raw_delta_sha256': hashlib.sha256(vector.tobytes()).hexdigest()})
                    del local, opt
                cosines[b, m] = pair[0]@pair[1]/(np.linalg.norm(pair[0])*np.linalg.norm(pair[1]))
        np.testing.assert_array_equal(features[2, 0], old_anchor_features)
        with np.load(PROFILE/'profiles'/f'{i:02d}.npz') as z:
            heads = z['heads']
        measured = measure(baseline, features, heads, pools, labels, domains, folds)
        with np.load(PRIOR/'states'/f'{i:02d}'/'responses.npz') as z:
            np.testing.assert_array_equal(measured[2, 0], z['response'][:, 1])
        np.savez_compressed(directory/'features.npz', before=baseline, after=features)
        np.savez_compressed(directory/'audit-directions.npz', raw_deltas=audit_vectors)
        np.savez_compressed(directory/'responses.npz', response=measured)
        write(directory/'audit.json', {'passed': True, 'all128_updates_and_complete_fork_exact': True,
            'all_features_repeat_exactly': True, 'anchor_features_responses_and_proposal_fingerprints_exact': True,
            'selected_pair_repeats_exact': True, 'case': c['id'], 'physical_SSL': work['SSL_completed']-start_ssl,
            'encoder_passes': work['passes_completed']-start_pass, 'proposals': proposals, 'cosines': cosines.tolist(),
            'parameter_names_shapes': [(n, list(p.shape)) for n, p in model.encoder.named_parameters() if p.requires_grad]})
        paths = [directory/n for n in ('features.npz', 'audit-directions.npz', 'responses.npz', 'audit.json')]
        hh = {str(p): digest(p) for p in paths}; write(done, {'passed': True, 'hashes': hh}); hashes.update(hh); hashes[str(done)] = digest(done)
        all_response.append(measured); work['in_progress'] = None; write(wp, work)
        status('collecting', states_complete=i+1, total=72, SSL=work['SSL_completed'], passes=work['passes_completed'])
        del model, optimizer, fork, saved; torch.cuda.empty_cache()
    assert work['SSL_completed'] == 10656 and work['passes_completed'] == 2736
    np.savez_compressed(OUT/'response-arrays.npz', response=all_response)
    hashes[str(OUT/'response-arrays.npz')] = digest(OUT/'response-arrays.npz'); hashes[str(wp)] = digest(wp)
    check(lock['hashes']); write(OUT/'response-ledger.json', {'passed': True, 'hashes': hashes, 'finished_unix': time.time(),
        'all_responses_sealed_before_target_loading': True, 'physical_SSL': 10656, 'encoder_passes': 2736, 'image_forwards': 612864})
    status('responses-sealed', states_complete=72)


def evaluate():
    check(read(OUT/'input_lock.json')['hashes']); ledger = read(OUT/'response-ledger.json'); check(ledger['hashes'])
    assert not (OUT/'target-start.json').exists(); write(OUT/'target-start.json', {'unix': time.time()})
    with np.load(OUT/'response-arrays.npz') as z:
        action = z['response']
    delta = action[:, :, :, 1]-action[:, :, :, 0]
    with np.load(PROFILE/'analysis-arrays.npz') as z:
        h, target, groups, diets = [z[k] for k in ('profiles', 'target', 'groups', 'diets')]
    with np.load(OLD/'analysis-arrays.npz') as z:
        signals, by_draw = z['signals'], z['benefit']; np.testing.assert_array_equal(target, z['target'])
    np.testing.assert_array_equal(target, by_draw.mean(2))
    with np.load(PRIOR/'analysis-arrays.npz') as z:
        old_pred = z['predictions'][0]
    settings = panels(delta, h); all_pred = []; screens = {}; errors = []; shadows = []; maximum = 0.
    subsets = [('all', 0, np.ones(72, bool))]
    for axis, name in enumerate(('seed', 'source', 'context')):
        subsets += [(name, int(v), groups[:, axis] == v) for v in np.unique(groups[:, axis])]
    subsets += [('diet', int(v), diets == v) for v in (0, 1)]
    for name, rr, static in settings:
        pred, folds, error = forecast(signals, diets, groups, static, rr, target.reshape(72, 14)); maximum = max(maximum, error)
        pp = np.stack([pred[m].reshape(72, 2, 7) for m in MODELS], 1); all_pred.append(pp)
        np.testing.assert_allclose(pp[:, 1], old_pred[:, 1], atol=1e-10, rtol=0)
        if name == 'anchor':
            np.testing.assert_allclose(pp, old_pred, atol=1e-10, rtol=0)
        screens[name] = screen(target[:, 1, 6], {m: pp[:, j, 1, 6] for j, m in enumerate(MODELS)}, groups)
        for j, model in enumerate(MODELS):
            for p in range(2):
                for k in range(7):
                    for subset, value, keep in subsets:
                        errors.append({'panel': name, 'model': model, 'procedure': p, 'component': k, 'subset': subset, 'value': value,
                            'MSE': float(np.mean((pp[keep, j, p, k]-target[keep, p, k])**2))})
        if name in ('all9', 'anchor'):
            for model in ('cheap', 'static', 'static_response_added'):
                take = pp[:, MODELS.index(model), 1, 6] > 0
                for k in range(7):
                    shadows.append({'panel': name, 'model': model, 'component': k, 'replay_states': int(take.sum()), 'value_pp': float(np.mean(target[:, 1, k]*take))})
    pp, anchor = all_pred[0], all_pred[1]; y = target[:, 1, 6]
    mse = {m: float(np.mean((pp[:, j, 1, 6]-y)**2)) for j, m in enumerate(MODELS)}
    anchor_mse = float(np.mean((anchor[:, 5, 1, 6]-y)**2))
    deletion = [n for n, _, _ in settings if n.startswith('drop_')]
    deletion_pass = all(screens[n]['baselines'][b]['MSE_reduction_fraction'] > 0 for n in deletion for b in ('cheap', 'static'))
    variance = grid_variance(delta[..., 0, :].mean(3))
    draw_error = np.array([np.mean((pp[:, :, 1, 6]-by_draw[:, 1, d, 6, None])**2, axis=0) for d in range(12)])
    np.savez_compressed(OUT/'analysis-arrays.npz', response=action, contrast=delta, profiles=h, target=target, benefit_by_draw=by_draw,
        groups=groups, diets=diets, predictions=all_pred, grid_variance=variance, draw_MSE=draw_error)
    csv_write(OUT/'errors.csv', errors); csv_write(OUT/'shadow-values.csv', shadows); write(OUT/'folds.json', {'folds': folds})
    write(OUT/'summary.json', {'states': 72, 'probes_per_state': 9, 'future_updates': 384, 'physical_SSL': 10656,
        'encoder_passes': 2736, 'image_forwards': 612864, 'reader_fits': 0, 'forecast_ridge_fits': 7308,
        'panels': [n for n, _, _ in settings], 'primary_MSE': mse, 'anchor_primary_MSE': anchor_mse,
        'screens': screens, 'primary_beats_anchor': bool(mse['static_response_added'] < anchor_mse),
        'all6_deletions_beat_both_baselines_pooled': bool(deletion_pass),
        'averaging_reliability_screen_passed': bool(screens['all9']['development_screen_passed'] and mse['static_response_added'] < anchor_mse and deletion_pass),
        'maximum_independent_ridge_error': maximum, 'all_anchor_forecasts_reproduced': True,
        'known_cohort_retrospective_development_only': True, 'no_controller_qualified': True})
    paths = [OUT/n for n in ('analysis-arrays.npz', 'summary.json', 'errors.csv', 'shadow-values.csv', 'folds.json', 'target-start.json')]
    check(read(OUT/'input_lock.json')['hashes']); check(ledger['hashes'])
    write(OUT/'evaluation-verification.json', {'passed': True, 'hashes': {str(p): digest(p) for p in paths},
        'all7308_ridge_fits_independently_verified': True, 'original_targets_and_anchor_reproduced': True})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('freeze', 'collect', 'evaluate', 'run'))
    stage = parser.parse_args().stage; base.setup()
    with threadpool_limits(limits=4):
        if stage == 'run':
            collect(); evaluate()
            from scripts.report_response_reliability import make_report
            write(OUT/'delivery-verification.json', make_report(OUT, ROOT/CFG['notebook']))
            status('numerically-complete-awaiting-QA')
        else:
            {'freeze': freeze, 'collect': collect, 'evaluate': evaluate}[stage]()


if __name__ == '__main__':
    main()
