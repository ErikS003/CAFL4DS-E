"""Independent literal response, selected encoder, ANOVA and count checks."""
from __future__ import annotations

import math

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from cafl4ds.content_step_intervention import load_xor
from scripts.run_response_reliability import CFG, OLD, OUT, PRIOR, PROFILE, base, check, digest, read, write
from scripts.verify_functional_response_profile import literal


def main():
    check(read(OUT/'input_lock.json')['hashes']); ledger = read(OUT/'response-ledger.json'); check(ledger['hashes'])
    check(read(OUT/'evaluation-verification.json')['hashes']); cases = read(OUT/'plan.json')['cases']
    with np.load(PROFILE/'design.npz') as z:
        pools, labels, domains, folds = [z[k] for k in ('pools', 'labels', 'domains', 'folds')]
    max_response = max_norm = max_variance = 0.; checked = 0
    for i, c in enumerate(cases):
        directory = OUT/'states'/f'{i:02d}'; audit = read(directory/'audit.json')
        assert audit['physical_SSL'] == 148 and audit['encoder_passes'] == 38
        encoder = load_xor(OLD/'branches'/c['id']/'fork/encoder-xor.npz', torch.load(c['base_encoder'], weights_only=True, map_location='cpu'))
        names = audit['parameter_names_shapes']; flat = np.concatenate([encoder[n].numpy().flatten() for n, _ in names]).astype(float)
        with np.load(directory/'audit-directions.npz') as z:
            vectors = z['raw_deltas']
        for action, vector in enumerate(vectors):
            r = next(r for r in audit['proposals'] if r['batch'] == r['mask'] == 1 and r['action'] == action)
            state = {n: t.clone() for n, t in encoder.items()}; offset = 0
            for n, shape in names:
                size = math.prod(shape); state[n] = torch.from_numpy((flat[offset:offset+size]+vector[offset:offset+size]).astype(np.float32).reshape(shape)); offset += size
            assert offset == len(vector) and base.fingerprint(state) == r['encoder_fingerprint']
            assert abs(np.linalg.norm(vector)-r['raw_norm']) < 1e-10
            realized = (flat+r['factor']*vector).astype(np.float32).astype(float)-flat
            error = abs(float(np.linalg.norm(realized))-r['actual_norm']); max_norm = max(max_norm, error)
        for r in audit['proposals']:
            assert abs(r['actual_norm']-r['target_norm']) <= max(1e-9, 2e-5*r['target_norm'])
        with np.load(directory/'features.npz') as z:
            before, after = z['before'], z['after']
        with np.load(PRIOR/'states'/f'{i:02d}'/'features.npz') as z:
            np.testing.assert_array_equal(before, z['before']); np.testing.assert_array_equal(after[2, 0], z['after'][:, 1])
        with np.load(directory/'responses.npz') as z:
            stored = z['response']
        with np.load(PROFILE/'profiles'/f'{i:02d}.npz') as z:
            heads = z['heads']
        for o, ids in enumerate(pools):
            y, d = labels[ids], domains[ids]
            for k in range(2):
                a = np.empty(128); b = np.empty((3, 3, 2, 128))
                for f in range(4):
                    te = folds[o] == f; beta = heads[o, k, f]
                    a[te] = before[ids[te]].astype(float)@beta[:-1]+beta[-1]
                    b[..., te] = after[..., ids[te], :].astype(float)@beta[:-1]+beta[-1]
                for batch, mask, action in np.ndindex(3, 3, 2):
                    error = float(np.max(abs(literal(a, b[batch, mask, action], y, d)-stored[batch, mask, action, o, k])))
                    max_response = max(max_response, error); checked += 1
    with np.load(OUT/'analysis-arrays.npz') as z:
        grid = z['contrast'][..., 0, :].mean(3); variance = z['grid_variance']
        by_draw, target, predictions, draw_mse = z['benefit_by_draw'], z['target'], z['predictions'], z['draw_MSE']
    np.testing.assert_array_equal(by_draw.mean(2), target)
    for i in range(72):
        for k in range(8):
            values = grid[i, :, :, k]; row, col = values.mean(1), values.mean(0); flat_values = values.ravel()
            total = np.mean((flat_values[:, None]-flat_values[None, :])**2)/2
            batch = np.mean((row[:, None]-row[None, :])**2)/2; mask = np.mean((col[:, None]-col[None, :])**2)/2
            direct = [total, batch, mask, total-batch-mask]
            max_variance = max(max_variance, float(np.max(abs(variance[i, k]-direct))))
    for draw in range(12):
        independent = np.mean((predictions[0, :, :, 1, 6]-by_draw[:, 1, draw, 6, None])**2, axis=0)
        np.testing.assert_array_equal(independent, draw_mse[draw])
    assert checked == 15552 and max_response < 1e-10 and max_norm < 1e-10 and max_variance < 1e-18
    work = read(OUT/'work.json'); assert work['SSL_started'] == work['SSL_completed'] == CFG['maximum_SSL']
    assert work['passes_started'] == work['passes_completed'] == CFG['encoder_passes']
    assert ledger['finished_unix'] < read(OUT/'target-start.json')['unix']
    check(read(OUT/'input_lock.json')['hashes']); check(ledger['hashes'])
    write(OUT/'independent-verification.json', {'passed': True, 'response_vectors_checked': checked,
        'maximum_literal_response_error': max_response, 'maximum_selected_actual_norm_error': max_norm,
        'maximum_independent_grid_variance_error': max_variance, 'selected_raw_encoders_reconstructed': 144,
        'all72_anchor_features_exact': True, 'all_readout_MSEs_and_original_targets_reconstructed': True,
        'physical_SSL': 10656, 'encoder_passes': 2736, 'image_forwards': 612864,
        'hashes': {str(OUT/'work.json'): digest(OUT/'work.json')}})
    print('Independent response, selected-encoder, finite-grid, target/readout and physical-work audits passed.')


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
