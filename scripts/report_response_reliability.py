"""Six fixed figures for repeated-probe reliability and original384-step utility."""
from __future__ import annotations

import base64
import contextlib
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cafl4ds.functional_response_profile import MODELS
from scripts.run_response_reliability import check, digest, read, write


def make_report(out, notebook, engineering=False):
    out, notebook = Path(out), Path(notebook)
    if not engineering:
        check(read(out/'input_lock.json')['hashes']); check(read(out/'response-ledger.json')['hashes'])
        check(read(out/'evaluation-verification.json')['hashes'])
    with np.load(out/'analysis-arrays.npz') as z:
        delta, groups, diets, target, pred, variance, draw = [z[k] for k in
            ('contrast', 'groups', 'diets', 'target', 'predictions', 'grid_variance', 'draw_MSE')]
    s = read(out/'summary.json'); labels = s['panels']; mean = delta[..., 0, :].mean((1, 2, 3))
    directory = out/'figures'; directory.mkdir(exist_ok=True); paths = []
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.formatter.limits': (-2, 3), 'axes.formatter.use_mathtext': True})
    def save(fig, name):
        fig.suptitle('SYNTHETIC ENGINEERING ONLY — NOT RESEARCH RESULTS' if engineering else
                     'Repeated pre-action probes / known-cohort retrospective384-update benefit', fontsize=11)
        p = directory/(name+'.png'); fig.savefig(p, dpi=130, bbox_inches='tight', facecolor='white'); plt.close(fig); paths.append(p)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained'); axes[0].axis('off')
    axes[0].text(.5, .5, '72 known fork states / all4 contexts / both diets\n3 already-observed batches x3 mask seeds\nPaired current/replay half-step probes\nFixed OOF readers / six support pools\nAll responses sealed before target loading\nOriginal384-update future benefit\n36 strict seed AND source AND context holdouts\n29 predeclared panels / no tuning', ha='center', va='center', fontsize=11)
    axes[1].axis('off'); axes[1].text(.5, .5, f"SSL work: {s['physical_SSL']:,}\nFeature passes: {s['encoder_passes']:,}\nImage forwards: {s['image_forwards']:,}\nExact128-step reconstruction\nExact proposal audit repeats\nExact study52 anchor reproduction\nNo new readout optimizer fits", ha='center', va='center', fontsize=11)
    save(fig, '01-crossed-probe-design-and-audits')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained'); vv = variance[:, 4:].mean(0)
    fractions = vv[:, 1:]/np.maximum(vv[:, :1], 1e-30); bottom = np.zeros(4)
    for j, name in enumerate(('Batch', 'Mask', 'Interaction')):
        axes[0].bar(range(4), fractions[:, j], bottom=bottom, label=name); bottom += fractions[:, j]
    axes[0].set(xticks=range(4), xticklabels=['Separation', 'Preference', 'Log spread', 'Boundary'], ylabel='Fraction of mean finite-grid variance', title='3x3 variability of six-pool means'); axes[0].legend()
    probe_sd = delta[..., 0, :].mean(3).std((1, 2)); pool_sd = delta[..., 0, :].mean((1, 2)).std(1)
    for j, name in enumerate(('Separation', 'Preference', 'Log spread', 'Boundary')):
        axes[1].scatter(np.maximum(probe_sd[:, j+4], 1e-10), np.maximum(pool_sd[:, j+4], 1e-10), s=18, alpha=.65, label=name)
    axes[1].set(xscale='log', yscale='log', xlabel='SD across9 probes (pool-mean response)', ylabel='SD across6 pools (nine-probe mean)', title='Two sources of measurement sensitivity'); axes[1].legend(fontsize=8)
    fig.supxlabel('Descriptive finite grids and overlapping support pools; not population variance components or confidence intervals', fontsize=9)
    save(fig, '02-batch-mask-and-support-variation')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for j, ax in enumerate(axes.ravel()):
        for diet, label, color in ((0, 'Current pre-diet', 'C0'), (1, 'Balanced pre-diet', 'C1')):
            for c in range(1, 5):
                ids = np.flatnonzero((groups[:, 2] == c) & (diets == diet)); x = c+(.13 if diet else -.13); values = mean[ids, j+4]
                ax.scatter(x+np.linspace(-.06, .06, len(ids)), values, s=22, color=color, alpha=.7, label=label if c == 1 else None)
                ax.plot([x-.09, x+.09], [values.mean()]*2, color=color, linewidth=3)
        ax.axhline(0, color='gray', linewidth=.7); ax.set(xticks=range(1, 5), xlabel='All continuation contexts', ylabel='Replay-minus-current response', title=['Night separation', 'Night preference', 'Night log spread', 'Night boundary mass'][j]); ax.legend(fontsize=8)
    fig.supxlabel('Primary average of9 probes and6 support-pool descriptors / lambda0.01 / fixed half-step magnitude', fontsize=9)
    save(fig, '03-average-response-by-context-and-diet')
    y = target[:, 1, 6]; p = pred[0, :, :, 1, 6]; a = pred[1, :, :, 1, 6]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    ml = ['Mean', 'Cheap', '+ Static', 'Response\nonly', '+ Response', '+ Static +\nresponse', '+ CKA +\nresponse']
    for offset, predictions, name, color in ((-.18, a, 'Single anchor', 'gray'), (.18, p, 'Nine-probe mean', 'C0')):
        axes[0].bar(np.arange(7)+offset, np.mean((predictions-y[:, None])**2, axis=0), width=.36, label=name, color=color)
    axes[0].set(xticks=range(7), xticklabels=ml, ylabel='Strictly held-out MSE (pp squared)', title='Original night-highway replay benefit'); axes[0].legend(fontsize=8)
    for j, base in enumerate(('cheap', 'static')):
        change = (p[:, MODELS.index(base)]-y)**2-(p[:, 5]-y)**2
        values = [change[groups[:, axis] == v].mean() for axis in (0, 1) for v in np.unique(groups[:, axis])]
        axes[1].bar(np.arange(6)+(j-.5)*.36, values, width=.36, label=f'Gain versus {base}')
    axes[1].axhline(0, color='gray'); axes[1].set(xticks=range(6), xticklabels=[f'Seed\n{v}' for v in np.unique(groups[:, 0])]+[f'Source\n{v}' for v in np.unique(groups[:, 1])], ylabel='Baseline MSE minus primary MSE', title='Every seed and source must improve'); axes[1].legend(fontsize=8)
    fig.supxlabel('24 training /2 test states in36 folds; same seven model designs, fixed ridge10, train-only preprocessing', fontsize=9)
    save(fig, '04-longer-term-MSE-and-group-stability')
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), layout='constrained')
    singles = ['anchor']+[v for v in labels if v.startswith('single_')]
    single_mse = [np.mean((pred[labels.index(v), :, 5, 1, 6]-y)**2) for v in singles]
    axes[0].bar(range(9), single_mse, color='gray'); axes[0].axhline(s['primary_MSE']['static_response_added'], color='C0', label='Nine-probe average'); axes[0].axhline(s['primary_MSE']['cheap'], color='C3', linestyle='--', label='Cheap baseline')
    axes[0].set(xticks=range(9), xticklabels=['Anchor']+[str(i) for i in range(1, 9)], ylabel='MSE (pp squared)', title='All9 individual probes retained'); axes[0].legend(fontsize=8)
    use = [v for v in labels if v.startswith('drop_')]+['all9_weak']
    for ax, selected, title in ((axes[1], use, 'Leave-factor-out and prior sensitivity'), (axes[2], [f'pool{i}' for i in range(6)], 'Support-pool sensitivity')):
        for j, base in enumerate(('cheap', 'static')):
            values = [100*s['screens'][v]['baselines'][base]['MSE_reduction_fraction'] for v in selected]
            ax.barh(np.arange(len(selected))+(j-.5)*.35, values, height=.35, label=f'Versus {base}')
        ax.axvline(10, color='C3', linestyle='--'); ax.set(yticks=range(len(selected)), yticklabels=[v.replace('drop_', 'omit ').replace('all9_weak', 'weaker prior') for v in selected], xlabel='MSE reduction (%)', title=title); ax.invert_yaxis(); ax.legend(fontsize=8)
    fig.supxlabel('Sensitivities share the same72 states; none can replace the frozen primary setting or provide independent confirmation', fontsize=9)
    save(fig, '05-single-average-and-deletion-robustness')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    for base in ('cheap', 'static'):
        values = draw[:, MODELS.index(base)]-draw[:, 5]
        axes[0].plot(range(1, 13), values, 'o-', label=f'Gain versus {base}')
    axes[0].axhline(0, color='gray'); axes[0].set(xticks=range(1, 13), xlabel='Original downstream readout draw', ylabel='Baseline MSE minus primary MSE', title='Same forecasts /12 original utility readouts'); axes[0].legend(fontsize=8)
    policies = [('Always replay', np.ones(72, bool)), ('Cheap', p[:, 1] > 0), ('Static', p[:, 2] > 0), ('Single anchor', a[:, 5] > 0), ('Nine-probe mean', p[:, 5] > 0)]
    for j, (name, take) in enumerate(policies):
        values = np.mean(target[:, 1, 3:7]*take[:, None], axis=0)
        axes[1].bar(np.arange(4)+(j-2)*.15, values, width=.15, label=name)
    axes[1].axhline(0, color='gray'); axes[1].set(xticks=range(4), xticklabels=['Day city', 'Day highway', 'Night city', 'Night highway'], ylabel='Value versus always-current (pp)', title='Shadow choices from held-out night-highway forecast'); axes[1].legend(fontsize=8, ncol=2)
    fig.supxlabel('Repeated readouts measure conditional assay sensitivity, not fresh training replication; policies are not deployed', fontsize=9)
    save(fig, '06-readout-sensitivity-and-shadow-class-values')
    text = ('SYNTHETIC ENGINEERING ONLY\n\n' if engineering else '')+'''# Repeated local responses and longer-term intervention benefit

All72 known pre-action states, all4 contexts and both pre-diets. Three recent
already-observed batches cross three mask seeds; paired current/replay model
copies use half of the pre-action actual-step norm. Fixed OOF readers evaluate
held-out validation support. All responses are sealed before original384-update
benefit targets are loaded. The study52 anchor is freshly recomputed exactly.

Primary averages all9 probe descriptors and6 support pools at lambda0.01.
The seven model designs, strict seed AND source AND context exclusions,
24 training/2 test rows, train-only preprocessing and ridge10 are unchanged.
All29 settings and all class/domain targets remain saved without selection.
The development screen requires10% improvement versus cheap AND static plus
every seed/source improving. The additional reliability screen also requires
beating the single anchor and all6 factor-deletion panels beating both baselines.

This is retrospective known-cohort development, not fresh confirmation or a
newly extended horizon. Batch/mask repetitions, overlapping pools and12 utility
readouts are measurement variation, not extra independent training states.
Histories share warm weights/footage and capture sessions remain unknown.
Finite-grid ANOVA is descriptive. More probe compute precludes an equal-compute
controller claim. No causal-health or alarm/controller qualification follows.
Observed findings are appended after independent and visual verification.
'''
    (out/'report.md').write_text(text, encoding='utf-8')
    code = f'''from pathlib import Path
import json,numpy as np
from cafl4ds.functional_response_profile import MODELS
p=Path({str(out)!r});s=json.loads((p/'summary.json').read_text())
with np.load(p/'analysis-arrays.npz') as z:
    y=z['target'][:,1,6];pred=z['predictions'][0,:,:,1,6]
    for j,m in enumerate(MODELS):
        assert abs(float(np.mean((pred[:,j]-y)**2))-s['primary_MSE'][m])<1e-9
assert s['states']==72 and len(s['panels'])==29
print(json.dumps(s,indent=2))
'''
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, 'response-reliability-notebook', 'exec'), {})
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)},
        {'cell_type': 'code', 'metadata': {}, 'source': code.splitlines(keepends=True), 'execution_count': 1,
         'outputs': [{'output_type': 'stream', 'name': 'stdout', 'text': stream.getvalue().splitlines(keepends=True)}]}]
    for i, p in enumerate(paths):
        cells += [{'cell_type': 'markdown', 'metadata': {}, 'source': [f'## Figure {i+1}: {p.stem}\n']},
            {'cell_type': 'code', 'metadata': {}, 'source': [f'from PIL import Image\nfigure_image=Image.open({str(p)!r})\nfigure_image\n'], 'execution_count': i+2,
             'outputs': [{'output_type': 'display_data', 'metadata': {}, 'data': {'image/png': base64.b64encode(p.read_bytes()).decode(), 'text/plain': [p.name]}}]}]
    write(notebook, {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name': 'Python3', 'language': 'python', 'name': 'python3'}}, 'cells': [{'id': f'response-reliability-{i}', **c} for i, c in enumerate(cells)]})
    for cell, p in zip(read(notebook)['cells'][3::2], paths, strict=True):
        assert base64.b64decode(cell['outputs'][0]['data']['image/png']) == p.read_bytes()
    return {'passed': True, 'visual_QA_pending': True, 'embedded_PNGs_checked': 6, 'reconstruction_executed': True,
        'hashes': {str(p): digest(p) for p in paths+[notebook, out/'report.md']}}
