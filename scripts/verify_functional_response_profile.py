"""Independent score-response arithmetic and saved parameter-direction audit."""
from __future__ import annotations

import math

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from cafl4ds.content_step_intervention import load_xor
from scripts.run_functional_response_profile import CFG, OLD, OUT, PREV, check, digest, fingerprint, read, write


def literal(a, b, y, d):
    result = []
    for dom in (0, 1):
        aa = [[float(s) for s, c, v in zip(a, y, d, strict=True) if v == dom and c == cls] for cls in (0, 1)]
        bb = [[float(s) for s, c, v in zip(b, y, d, strict=True) if v == dom and c == cls] for cls in (0, 1)]
        mean = lambda v: math.fsum(v)/len(v)
        variance = lambda v: mean([(s-mean(v))**2 for s in v])
        sa, sb = [max(1e-8, math.sqrt(sum(variance(v) for v in cells)/2)) for cells in (aa, bb)]
        delta = [mean(new)-mean(old) for old, new in zip(aa, bb, strict=True)]
        mass = sum((sum(abs(s) <= .25*sa for s in new)-sum(abs(s) <= .25*sa for s in old))/len(old)
                   for old, new in zip(aa, bb, strict=True))/2
        result += [(delta[1]-delta[0])/(2*sa), sum(delta)/(2*sa), math.log(sb/sa), mass]
    return np.array(result)


def main():
    check(read(OUT/'input_lock.json')['hashes']); ledger = read(OUT/'response-ledger.json'); check(ledger['hashes'])
    cases = read(OUT/'plan.json')['cases']; max_response = max_norm = 0.; checked = 0
    with np.load(PREV/'design.npz') as z:
        pools, labels, domains, folds = [z[k] for k in ('pools', 'labels', 'domains', 'folds')]
    for i, c in enumerate(cases):
        directory = OUT/'states'/f'{i:02d}'; audit = read(directory/'audit.json')
        assert audit['physical_SSL'] == 132 and audit['encoder_passes'] == 10
        base = load_xor(OLD/'branches'/c['id']/'fork/encoder-xor.npz', torch.load(c['base_encoder'], weights_only=True, map_location='cpu'))
        names = audit['parameter_names_shapes']; flat = np.concatenate([base[n].numpy().flatten() for n, _ in names]).astype(float)
        with np.load(directory/'directions.npz') as z:
            vectors = z['raw_deltas']
        for action, v in enumerate(vectors):
            state = {n: t.clone() for n, t in base.items()}; offset = 0
            for n, shape in names:
                size = math.prod(shape); state[n] = torch.from_numpy((flat[offset:offset+size]+v[offset:offset+size]).astype(np.float32).reshape(shape)); offset += size
            assert offset == len(v) and fingerprint(state) == audit['proposals'][action]['encoder_fingerprint']
            raw = float(np.linalg.norm(v))
            for dose in [r for r in audit['doses'] if r['action'] == action]:
                assert abs(raw-dose['raw_norm']) < 1e-10
                realized = (flat+dose['factor']*v).astype(np.float32).astype(float)-flat
                norm = float(np.linalg.norm(realized)); max_norm = max(max_norm, abs(norm-dose['actual_norm']))
                assert abs(norm-dose['target_norm']) <= max(1e-9, 2e-5*dose['target_norm'])
        with np.load(directory/'features.npz') as z:
            before, after = z['before'], z['after']
        with np.load(directory/'responses.npz') as z:
            stored = z['response']
        with np.load(PREV/'profiles'/f'{i:02d}.npz') as z:
            heads = z['heads']
        for o, ids in enumerate(pools):
            y, d = labels[ids], domains[ids]
            for k in range(2):
                a = np.empty(128); b = np.empty((2, 2, 128))
                for f in range(4):
                    te = folds[o] == f; beta = heads[o, k, f]
                    a[te] = before[ids[te]].astype(float)@beta[:-1]+beta[-1]
                    b[:, :, te] = after[:, :, ids[te]].astype(float)@beta[:-1]+beta[-1]
                for action in range(2):
                    for dose in range(2):
                        error = float(np.max(abs(literal(a, b[action, dose], y, d)-stored[action, dose, o, k])))
                        max_response = max(max_response, error); checked += 1
    assert checked == 3456 and max_response < 1e-10 and max_norm < 1e-10
    work = read(OUT/'work.json'); assert work['SSL_started'] == work['SSL_completed'] == CFG['maximum_SSL']
    assert work['passes_started'] == work['passes_completed'] == CFG['encoder_passes']
    assert ledger['finished_unix'] < read(OUT/'target-start.json')['unix']
    check(read(OUT/'input_lock.json')['hashes']); check(ledger['hashes'])
    write(OUT/'independent-verification.json', {'passed': True, 'response_vectors_checked': checked,
        'maximum_literal_response_error': max_response, 'maximum_numpy_actual_norm_error': max_norm,
        'all144_raw_proposal_encoders_reconstructed_exactly': True, 'physical_SSL': 9504, 'encoder_passes': 720,
        'all_frozen_inputs_unchanged': True, 'hashes': {str(OUT/'work.json'): digest(OUT/'work.json')}})
    print('All response arithmetic, proposal encoders, actual dose norms and physical-work counts independently verified.')


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
