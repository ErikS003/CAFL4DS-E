"""Nested, budget-matched reader/evaluation precision diagnostics."""
from __future__ import annotations
import numpy as np
from cafl4ds.reader_bank_cross import allocations, descriptors, sigma, crossed_variance

METHODS = ('single32', 'single64', 'descriptor_mean2x32', 'score_mean2x32')
METHOD_IDS = ((0, 1), (2,), (3,), (4,))
EVAL_SLOTS = ((0, 1), (2,))
EVAL_SIZES = (16, 32)
FACTORS = ('reader', 'evaluation', 'both')
LAMBDAS = (.01, .001)


def nested_roles(labels, domains):
    calibration, evaluation, unused = allocations(labels, domains)
    sub = calibration.reshape(6, 2, 4, 16).reshape(6, 2, 4, 2, 8).transpose(0, 1, 3, 2, 4).reshape(6, 2, 2, 32)
    small = evaluation.reshape(6, 2, 4, 8).reshape(6, 2, 4, 2, 4).transpose(0, 1, 3, 2, 4).reshape(6, 2, 2, 16)
    return calibration, sub, evaluation, small, unused


def measure(before, views, heads, calibration, sub, evaluation, small, labels, domains):
    """heads[draw,prior,calibration group,4 score heads,coefficient].

    Head order:32a/32b/64/mean32. Method3 is an average of already normalized
    descriptors; method4 is raw-score averaging followed by a single descriptor.
    Eval slots16a/16b/32. Tail:group,eval bank,scale,view,axis.
    """
    count = len(views)
    local = np.empty((6, 2, 3, 5, 2, 2, 2, count, 8))
    static = np.empty((6, 2, 3, 5, 2, 2, 2, 8))
    accuracy = np.empty((6, 2, 3, 4, 2, 2, 4))
    cs = np.empty((6, 2, 2, 4, 2))
    floors = 0
    for draw, k, group, h in np.ndindex(6, 2, 2, 4):
        beta = heads[draw, k, group, h]
        score = before.astype(float)@beta[:-1]+beta[-1]
        scored = views.astype(float)@beta[:-1]+beta[-1]
        ids = sub[draw, group, h] if h < 2 else calibration[draw, group]
        cs[draw, k, group, h] = sigma(score[ids], labels[ids], domains[ids])
        floors += int(np.sum(cs[draw, k, group, h] == 1e-8))
        method = (0, 1, 2, 4)[h]
        for slot, bank in np.ndindex(3, 2):
            ix = evaluation[draw, bank] if slot == 2 else small[draw, bank, slot]
            measured, baseline = descriptors(score[ix], scored[:, ix], labels[ix], domains[ix], cs[draw, k, group, h])
            local[draw, k, slot, method, group, bank] = measured.transpose(1, 0, 2)
            static[draw, k, slot, method, group, bank] = baseline
            accuracy[draw, k, slot, h, group, bank] = [np.mean((score[ix][(domains[ix] == d) & (labels[ix] == c)] > 0) == c) for d in (0, 1) for c in (0, 1)]
            floors += int(np.sum(sigma(score[ix], labels[ix], domains[ix]) == 1e-8))
    local[:, :, :, 3] = (local[:, :, :, 0]+local[:, :, :, 1])/2
    static[:, :, :, 3] = (static[:, :, :, 0]+static[:, :, :, 1])/2
    return local, static, accuracy, cs, floors


def paired_arrays(grid, factor):
    # grid[state,draw,group,bank]; each factor has exactly two symmetric pairs.
    if factor == 0:
        return grid[:, :, 0, :], grid[:, :, 1, :]
    if factor == 1:
        return grid[:, :, :, 0], grid[:, :, :, 1]
    return np.stack([grid[:, :, 0, 0], grid[:, :, 0, 1]], -1), np.stack([grid[:, :, 1, 1], grid[:, :, 1, 0]], -1)


def agreement(pairs):
    """Mean paired moments across fixed repetitions, not pooled pseudo-states.

    CCC=2cov/(varA+varB+(meanA-meanB)^2), equal to1-MSE/denominator.
    No population ICC/uncertainty claim; no z-scoring or alignment on eval data.
    """
    numerator = denominator = variance = energy = covariance = 0.; correlations = []
    for a, b in pairs:
        assert a.shape == b.shape and a.ndim == 3
        am, bm = a.mean(0), b.mean(0); av, bv = a.var(0), b.var(0)
        co = ((a-am)*(b-bm)).mean(0)
        numerator += float(((a-b)**2).mean())
        denominator += float((av+bv+(am-bm)**2).mean())
        variance += float((av+bv).mean()/2)
        covariance += float(co.mean())
        energy += float((a*a+b*b).mean()/2)
        valid = av*bv > 1e-30
        correlations.extend((co[valid]/np.sqrt(av[valid]*bv[valid])).tolist())
    n = len(pairs); assert n
    numerator /= n; denominator /= n; covariance /= n
    defined = denominator > 1e-30
    ccc = 2*covariance/denominator if defined else None
    if defined:
        assert abs(ccc-(1-numerator/denominator)) < 1e-10
    return {'paired_MSE': numerator, 'paired_RMS': float(np.sqrt(numerator)), 'concordance': ccc,
        'normalized_disagreement': numerator/denominator if defined else None,
        'between_state_SD': float(np.sqrt(variance/n)), 'RMS_amplitude': float(np.sqrt(energy/n)),
        'median_Pearson_r': float(np.median(correlations)) if correlations else None,
        'moment_denominator': denominator, 'states': a.shape[0], 'fixed_paired_measurements': int(sum(x.shape[1]*x.shape[2] for x, _ in pairs))}


def summary_row(values, k, size, method, scale, axis, factor, keep=None, draw=None):
    keep = np.arange(len(values)) if keep is None else np.flatnonzero(keep)
    pairs = []; components = []
    for slot in EVAL_SLOTS[size]:
        for m in METHOD_IDS[method]:
            grid = values[keep, :, k, slot, m, :, :, scale, axis]
            if draw is not None:
                grid = grid[:, draw:draw+1]
            pairs.append(paired_arrays(grid, factor))
            components.append(crossed_variance(grid[..., None]).mean((0, 1))[0])
    component = np.mean(components, axis=0)
    return {**agreement(pairs), 'finite_grid_total_reader_eval_interaction': component.tolist(),
        'finite_grid_fraction': (component[1:]/component[0]).tolist() if component[0] > 0 else [0., 0., 0.]}


def analyze(local, static, groups, diets):
    rows = []; partitions = []; strata = []
    for kind, values in (('local', local), ('static', static)):
        for k, size, method, scale, axis, factor in np.ndindex(2, 2, 4, 2, 8, 3):
            key = {'kind': kind, 'prior': k, 'evaluation_n': EVAL_SIZES[size], 'method': METHODS[method], 'scale': ('evaluation', 'calibration')[scale], 'axis': axis, 'factor': FACTORS[factor]}
            rows.append({**key, **summary_row(values, k, size, method, scale, axis, factor)})
            if kind == 'local' and k == scale == 0 and axis >= 4:
                for draw in range(6):
                    partitions.append({**key, 'partition': draw, **summary_row(values, k, size, method, scale, axis, factor, draw=draw)})
    subsets = [('all', 0, np.ones(72, bool))]
    for g, name in enumerate(('seed', 'source', 'context')):
        subsets += [(name, int(v), groups[:, g] == v) for v in np.unique(groups[:, g])]
    subsets += [('diet', int(d), diets == d) for d in (0, 1)]
    subsets += [('context_diet', f'{c}/{d}', (groups[:, 2] == c) & (diets == d)) for c in range(1, 5) for d in (0, 1)]
    for method, size in np.ndindex(4, 2):
        for name, value, keep in subsets:
            strata.append({'subset': name, 'value': value, 'method': METHODS[method], 'evaluation_n': EVAL_SIZES[size],
                **summary_row(local, 0, size, method, 0, 4, 0, keep=keep)})
    primary = [r for r in rows if r['kind'] == 'local' and r['prior'] == 0 and r['evaluation_n'] == 32 and r['scale'] == 'evaluation' and r['axis'] == 4 and r['factor'] == 'reader']
    a, b = primary[2], primary[1]
    effect = {'descriptor_mean_minus_single64_concordance': a['concordance']-b['concordance'],
        'descriptor_mean_to_single64_between_state_SD_ratio': a['between_state_SD']/b['between_state_SD'],
        'descriptor_mean_to_single64_paired_MSE_ratio': a['paired_MSE']/b['paired_MSE'],
        'descriptor_mean_to_single64_normalized_disagreement_ratio': a['normalized_disagreement']/b['normalized_disagreement']}
    return {'rows': rows, 'partitions': partitions, 'strata': strata, 'primary_rows': primary, 'primary_effect': effect}
