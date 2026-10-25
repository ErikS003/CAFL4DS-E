"""Six fixed functional-profile figures and a reconstructable notebook."""
from __future__ import annotations

import base64
import contextlib
import csv
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cafl4ds.functional_health_profile import FEATURES, MODELS, profile
from scripts.run_functional_health_profile import CFG, OUT, ROOT, check, digest, read, write


def rows(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def controls():
    y = np.tile(np.repeat([0, 1], 32), 2); d = np.repeat([0, 1], 64)
    scores = np.tile(np.r_[np.linspace(-2, .4, 32), np.linspace(-.4, 2, 32)], 2)
    base = profile(scores, y, d)
    return np.array([profile(scores+1, y, d)-base, profile(3*scores, y, d)-base])[:, 4:]


def make_report(out, notebook, engineering=False):
    out, notebook = Path(out), Path(notebook)
    if not engineering:
        check(read(out/'input_lock.json')['hashes']); check(read(out/'profile-ledger.json')['hashes'])
        check(read(out/'evaluation-verification.json')['hashes'])
    with np.load(out/'analysis-arrays.npz') as z:
        h, full, target, groups, diets, predictions = [z[k] for k in
            ('profiles', 'full_profiles', 'target', 'groups', 'diets', 'predictions')]
    summary = read(out/'summary.json'); corr = rows(out/'correlations.csv'); shadow = rows(out/'shadow-values.csv')
    avg = h[:, :, 0].mean(1); fullavg = full[:, :, 0].mean(1); sd = h[:, :, 0].std(1)
    names = ['Separation / within-class SD', 'Preference / within-class SD', 'Log within-class SD', 'Near-boundary mass']
    paths = []; directory = out/'figures'; directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    def save(fig, name):
        fig.suptitle('SYNTHETIC ENGINEERING ONLY — NOT RESEARCH RESULTS' if engineering else
                     'Known-cohort retrospective development / labeled pre-action support', fontsize=11)
        p = directory/(name+'.png'); fig.savefig(p, dpi=130, bbox_inches='tight', facecolor='white')
        plt.close(fig); paths.append(p)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained'); axes[0].axis('off')
    axes[0].text(.5, .5, '72 pre-action states / all 4 contexts / both diets\n6 overlapping validation support pools\n4-fold OOF: 96 fitting + 32 held-out labels\n4 measurements each for day and night\nProfiles sealed before target loading\nStrict seed AND source AND context exclusion\nOriginal subsequent replay-minus-current benefit\nNo new SSL or encoder inference', ha='center', va='center', fontsize=11)
    c = controls()
    for offset, values, label, color in [(-.18, c[0], 'Common score shift +1', 'C1'), (.18, c[1], 'Positive scale x3', 'C0')]:
        axes[1].bar(np.arange(4)+offset, values, width=.36, label=label, color=color)
    axes[1].set(xticks=range(4), xticklabels=['Separation', 'Preference', 'Log spread', 'Boundary'], ylabel='Synthetic profile change', title='Interpretation controls (synthetic)')
    axes[1].legend(fontsize=8); axes[1].axhline(0, color='gray', linewidth=.7)
    save(fig, '01-design-and-functional-controls')
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), layout='constrained')
    for j in range(4):
        k = j+4; ax = axes[0, j]
        for diet, label in ((0, 'Current'), (1, 'Balanced')):
            keep = diets == diet; ax.scatter(fullavg[keep, k], avg[keep, k], s=17, alpha=.7, label=label)
        low = min(fullavg[:, k].min(), avg[:, k].min()); high = max(fullavg[:, k].max(), avg[:, k].max())
        ax.plot([low, high], [low, high], color='gray', linewidth=1)
        ax.set(xlabel='Full-128 in-sample profile', ylabel='96-fit / 128 OOF profile', title=names[j]); ax.legend(fontsize=7)
        axes[1, j].boxplot([sd[groups[:, 2] == v, k] for v in range(1, 5)], tick_labels=['1', '2', '3', '4'], showfliers=True)
        axes[1, j].set(xlabel='Continuation context', ylabel='SD across six support pools', title='Measurement sensitivity across states')
    fig.supxlabel('Night / lambda 0.01; full-versus-OOF also changes fit size; overlapping pools do not give population uncertainty', fontsize=9)
    save(fig, '02-OOF-full-and-pool-sensitivity')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for j, ax in enumerate(axes.ravel()):
        for diet, label, color in ((0, 'Current pre-diet', 'C0'), (1, 'Balanced pre-diet', 'C1')):
            for context in range(1, 5):
                ids = np.flatnonzero((diets == diet) & (groups[:, 2] == context))
                jitter = np.linspace(-.06, .06, len(ids)); x = context+(.13 if diet else -.13)
                ax.scatter(x+jitter, avg[ids, j+4], s=23, alpha=.65, color=color, label=label if context == 1 else None)
                ax.plot([x-.09, x+.09], [avg[ids, j+4].mean()]*2, color=color, linewidth=3)
        ax.set(xticks=range(1, 5), xlabel='Continuation context (all included)', ylabel=names[j], title=f'Night {names[j].split(" /")[0]}')
        ax.legend(fontsize=8)
    fig.supxlabel('Points are 72 states, each the mean of six OOF pool profiles; horizontal strokes are descriptive means', fontsize=9)
    save(fig, '03-functional-profile-by-context-and-diet')
    labels = ['Mean', 'Diet', 'Cheap', '+ CKA', 'Function\nonly', '+ Function', '+ CKA +\nfunction', '+ Day +\nnight']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    axes[0].bar(range(8), [summary['primary_MSE'][m] for m in MODELS], color=['gray']*5+['C0', 'C2', 'C4'])
    axes[0].set(xticks=range(8), xticklabels=labels, ylabel='Held-out MSE (percentage points squared)', title='Original fixed-reader night-highway benefit')
    y = target[:, 1, 6]; p = predictions[0, :, :, 1, 6]
    improvement = (p[:, 2]-y)**2-(p[:, 5]-y)**2
    vals = []; glabels = []
    for axis, name in ((0, 'Seed'), (1, 'Source')):
        for value in np.unique(groups[:, axis]):
            vals.append(improvement[groups[:, axis] == value].mean()); glabels.append(f'{name}\n{value}')
    axes[1].bar(range(6), vals, color=['C0' if v >= 0 else 'C3' for v in vals]); axes[1].axhline(0, color='gray')
    axes[1].set(xticks=range(6), xticklabels=glabels, ylabel='Cheap MSE minus functional-added MSE', title='Primary stability across each seed and source')
    fig.supxlabel('24 training / 2 test states in each of 36 folds; all preprocessing uses training rows only; fixed ridge 10', fontsize=9)
    save(fig, '04-heldout-forecast-and-group-stability')
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    for diet, label in ((0, 'Current pre-diet'), (1, 'Balanced pre-diet')):
        keep = diets == diet; axes[0].scatter(y[keep], p[keep, 5], label=label, s=27, alpha=.7)
    lim = [float(min(y.min(), p[:, 5].min())), float(max(y.max(), p[:, 5].max()))]
    axes[0].plot(lim, lim, color='gray'); axes[0].axhline(0, color='gray', linewidth=.5)
    axes[0].set(xlabel='Observed replay-minus-current benefit (pp)', ylabel='Strictly held-out forecast (pp)', title='Primary functional-added model'); axes[0].legend()
    panelnames = list(summary['screens']); reductions = [100*summary['screens'][v]['MSE_reduction_fraction'] for v in panelnames]
    axes[1].barh(range(8), reductions, color=['C0', 'C1']+['gray']*6); axes[1].axvline(10, linestyle='--', color='C3', label='Development screen: 10%')
    axes[1].set(yticks=range(8), yticklabels=['Six-pool mean, .01', 'Six-pool mean, .001']+[f'Single pool {i}, .01' for i in range(6)], xlabel='MSE reduction over cheap (%)', title='Predeclared prior / pool sensitivity')
    axes[1].invert_yaxis(); axes[1].legend(fontsize=8)
    fig.supxlabel('Same 72 states in every panel; single pools and priors are sensitivities, not independent replications or model selection', fontsize=9)
    save(fig, '05-benefit-forecast-and-sensitivity')
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), layout='constrained')
    grid = np.array([[float(next(r['pearson_r'] for r in corr if int(r['diet']) == diet and r['feature'] == feature)) for feature in FEATURES] for diet in (0, 1)])
    im = axes[0].imshow(grid, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    axes[0].set(yticks=[0, 1], yticklabels=['Current pre-diet', 'Balanced pre-diet'], xticks=range(8), xticklabels=[f'{d}\n{m}' for d in ('Day', 'Night') for m in ('separation', 'preference', 'log spread', 'boundary')], title='Exploratory within-diet correlations with future night-highway benefit')
    for (r, c), v in np.ndenumerate(grid):
        axes[0].text(c, r, f'{v:+.2f}', ha='center', va='center', color='white' if abs(v) > .6 else 'black')
    fig.colorbar(im, ax=axes[0], label='Pearson r (36 known states / diet)')
    policies = ['always_current', 'always_replay', 'cheap', 'functional_added', 'cka_functional_added']
    comps = ['day_city', 'day_highway', 'night_city', 'night_highway']
    for j, pol in enumerate(policies):
        values = [float(next(r['value_relative_current_pp'] for r in shadow if r['model'] == pol and r['component'] == c)) for c in comps]
        axes[1].bar(np.arange(4)+(j-2)*.15, values, width=.15, label=pol.replace('_', ' '))
    axes[1].axhline(0, color='gray'); axes[1].set(xticks=range(4), xticklabels=[c.replace('_', ' ') for c in comps], ylabel='Shadow value versus always-current (pp)', title='Choose replay when held-out night-highway forecast > 0; original fixed assay')
    axes[1].legend(fontsize=8, ncol=3)
    fig.supxlabel('Associations and shadow decisions are retrospective; labels and old utility are required; no causal interpretation or deployed alarm', fontsize=9)
    save(fig, '06-correlations-and-shadow-class-values')
    text = ('SYNTHETIC ENGINEERING ONLY\n\n' if engineering else '')+'''# Pre-action functional health profile

All72 known study43 fork states, all4 continuation contexts and both pre-diets.
Profiles use validation support only, image-disjoint from the original benefit
query. Six overlapping128-label pools; four-fold OOF predictions from96-label
readers; lambda0.01 primary and0.001 sensitivity. The four domain measurements
are normalized class separation, normalized class preference, log within-class
score spread and scale-standardized near-boundary mass. Fold readers are pooled
for the profile; sampling and calibration sensitivity remain part of the assay.

Profiles and receipts were sealed before loading old intervention targets.
Original study43 fixed-reader night-highway replay-minus-current benefit is
primary, with joint and all7 components retained. This reader assay differs
from the profile's converged regularized reader; utility has not been redefined.
Strict holdouts exclude test seed AND source AND context; training24/test2.
All model designs and ridge10 were fixed, with train-only standardization.
Both priors, all six single-pool panels, per-context/diet errors and class shadow
values are retained. The primary development screen requires10% MSE reduction
over cheap, plus improvement in each seed and source.

Known-cohort retrospective development only. Outcomes have previously been
studied; histories share warm weights/footage and capture sessions are unknown.
Repeated support pools are not independent state replications. Profiles need
task labels and a specified reader/prior. Correlations do not identify causes,
and neither a passing nor failing development screen qualifies a health alarm.
No new SSL, inference, data downloads or actual action decisions were made.
Observed findings are added after independent verification and plot inspection.
'''
    (out/'report.md').write_text(text, encoding='utf-8')
    code = f'''from pathlib import Path
import json, numpy as np
p=Path({str(out)!r})
s=json.loads((p/'summary.json').read_text(encoding='utf-8'))
with np.load(p/'analysis-arrays.npz') as z:
    y=z['target'][:,1,6]
    predictions=z['predictions'][0,:,:,1,6]
    from cafl4ds.functional_health_profile import MODELS
    for j,m in enumerate(MODELS):
        assert abs(float(np.mean((predictions[:,j]-y)**2))-s['primary_MSE'][m])<1e-9
assert s['states']==72 and s['strict_holdout_folds']==36
print(json.dumps(s,indent=2))
'''
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, 'functional-profile-reconstruction', 'exec'), {})
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)},
        {'cell_type': 'code', 'metadata': {}, 'source': code.splitlines(keepends=True), 'execution_count': 1,
         'outputs': [{'output_type': 'stream', 'name': 'stdout', 'text': stream.getvalue().splitlines(keepends=True)}]}]
    for i, pth in enumerate(paths):
        cells += [{'cell_type': 'markdown', 'metadata': {}, 'source': [f'## Figure {i+1}: {pth.stem}\n']},
            {'cell_type': 'code', 'metadata': {}, 'source': [f'from IPython.display import Image, display\ndisplay(Image(filename={str(pth)!r}))\n'], 'execution_count': i+2,
             'outputs': [{'output_type': 'display_data', 'metadata': {}, 'data': {'image/png': base64.b64encode(pth.read_bytes()).decode(), 'text/plain': [pth.name]}}]}]
    write(notebook, {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name': 'Python3', 'language': 'python', 'name': 'python3'}},
        'cells': [{'id': f'functional-profile-{i}', **c} for i, c in enumerate(cells)]})
    for cell, pth in zip(read(notebook)['cells'][3::2], paths, strict=True):
        assert base64.b64decode(cell['outputs'][0]['data']['image/png']) == pth.read_bytes()
    return {'passed': True, 'visual_QA_pending': True, 'saved_reconstruction_executed': True, 'embedded_PNGs_checked': 6,
        'hashes': {str(p): digest(p) for p in paths+[notebook, out/'report.md']}}


if __name__ == '__main__':
    write(OUT/'delivery-verification.json', make_report(OUT, ROOT/CFG['notebook']))
