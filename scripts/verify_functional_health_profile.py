"""Independent literal profile arithmetic, OOF reconstruction and fit audit."""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from scripts.run_functional_health_profile import CFG, GEO, OUT, check, digest, read, write


def literal_profile(scores, labels, domains):
    result = []
    for domain in (0, 1):
        cell = [[float(s) for s, y, d in zip(scores, labels, domains, strict=True) if d == domain and y == c] for c in (0, 1)]
        means = [math.fsum(v)/len(v) for v in cell]
        variances = [math.fsum((s-mu)**2 for s in v)/len(v) for v, mu in zip(cell, means, strict=True)]
        sd = max(1e-8, math.sqrt(math.fsum(variances)/2))
        mass = sum(sum(abs(s) <= .25*sd for s in v)/len(v) for v in cell)/2
        result += [(means[1]-means[0])/(2*sd), sum(means)/(2*sd), math.log(sd), mass]
    return np.array(result)


def main():
    lock = read(OUT/'input_lock.json'); check(lock['hashes'])
    ledger = read(OUT/'profile-ledger.json'); check(ledger['hashes'])
    check(read(OUT/'evaluation-verification.json')['hashes'])
    jobs = read(OUT/'plan.json')['jobs']
    with np.load(OUT/'design.npz') as z:
        pools, labels, domains, folds = [z[k] for k in ('pools', 'labels', 'domains', 'folds')]
    max_score = max_profile = max_gradient = max_fold = 0.; checked = 0
    for i, j in enumerate(jobs):
        with np.load(GEO/'features'/j['point']/'features.npz') as z:
            support = z['eval'][:200].astype(float)
        with np.load(OUT/'profiles'/f'{i:02d}.npz') as z:
            heads, scores, h, fh = [z[k] for k in ('heads', 'oof_scores', 'profiles', 'fold_profiles')]
        for o, ids in enumerate(pools):
            x = np.column_stack((support[ids], np.ones(128))); y, d = labels[ids], domains[ids]
            for k, lam in enumerate(CFG['lambdas']):
                rebuilt = np.empty(128)
                for f in range(4):
                    te = folds[o] == f; tr = ~te; beta = heads[o, k, f]
                    # Independent raw logistic gradient at the saved solution.
                    gradient = x[tr].T@(expit(x[tr]@beta)-y[tr])/tr.sum()+lam*beta
                    max_gradient = max(max_gradient, float(np.max(abs(gradient))))
                    rebuilt[te] = x[te]@beta
                    max_fold = max(max_fold, float(np.max(abs(literal_profile(rebuilt[te], y[te], d[te])-fh[o, k, f]))))
                    checked += 1
                max_score = max(max_score, float(np.max(abs(rebuilt-scores[o, k]))))
                max_profile = max(max_profile, float(np.max(abs(literal_profile(rebuilt, y, d)-h[o, k]))))
    assert checked == 3456 and max_gradient <= 1.01e-10
    assert max(max_score, max_profile, max_fold) <= 1e-10
    journal = [json.loads(line) for line in (OUT/'fit-journal.jsonl').read_text().splitlines()]
    starts = [r for r in journal if r['phase'] == 'started']; done = [r for r in journal if r['phase'] == 'complete']
    keys = lambda r: tuple(r[k] for k in ('kind', 'case', 'outer', 'lambda', 'fold'))
    assert len(starts) == len(done) == 3600 and len({keys(r) for r in starts}) == 3600
    assert [keys(r) for r in starts] == [keys(r) for r in done]
    check(lock['hashes']); check(ledger['hashes'])
    write(OUT/'independent-verification.json', {'passed': True, 'all_collection_heads_gradient_checked': checked,
        'maximum_gradient_inf': max_gradient, 'maximum_OOF_score_reconstruction_error': max_score,
        'maximum_literal_profile_error': max_profile, 'maximum_literal_fold_profile_error': max_fold,
        'physical_reader_fits_journaled': 3600, 'all_frozen_inputs_unchanged': True,
        'hashes': {str(OUT/'fit-journal.jsonl'): digest(OUT/'fit-journal.jsonl')}})
    print('Independent OOF, literal profile, all 3456 gradient and all 3600 fit-journal checks passed.')


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
