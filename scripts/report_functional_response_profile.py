"""Six predeclared local-response figures and saved-array notebook."""
from __future__ import annotations

import base64
import contextlib
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cafl4ds.functional_response_profile import MODELS, response
from scripts.report_functional_health_profile import rows
from scripts.run_functional_response_profile import check, digest, read, write


def make_report(out, notebook, engineering=False):
    out, notebook = Path(out), Path(notebook)
    if not engineering:
        check(read(out/'input_lock.json')['hashes']); check(read(out/'response-ledger.json')['hashes'])
        check(read(out/'evaluation-verification.json')['hashes'])
    with np.load(out/'analysis-arrays.npz') as z:
        contrast, target, groups, diets, prediction = [z[k] for k in ('contrast', 'target', 'groups', 'diets', 'predictions')]
    primary = contrast[:, 1, :, 0].mean(1); quarter = contrast[:, 0, :, 0].mean(1)
    s = read(out/'summary.json'); audits = read(out/'audit-summary.json')['states']
    corr, shadow = rows(out/'correlations.csv'), rows(out/'shadow-values.csv')
    paths = []; directory = out/'figures'; directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    def save(fig, name):
        fig.suptitle('SYNTHETIC ENGINEERING ONLY — NOT RESEARCH RESULTS' if engineering else
                     'Known-cohort retrospective development / local responses on disposable model copies', fontsize=11)
        p = directory/(name+'.png'); fig.savefig(p, dpi=130, bbox_inches='tight', facecolor='white'); plt.close(fig); paths.append(p)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained'); axes[0].axis('off')
    axes[0].text(.5, .5, '72 exact pre-action learner reconstructions\nCurrent batch127 already observed; historical replay\nPaired inherited-Adam proposals / exact repeats\nQuarter and half of causal encoder-step norm\nFixed OOF readers / held-out labeled support\nZero and perturbed features each read twice\nAll responses sealed before utility loading\nNo future continuation frames in the probe', ha='center', va='center', fontsize=11)
    y = np.tile(np.repeat([0, 1], 16), 2); d = np.repeat([0, 1], 32)
    a = np.tile(np.r_[np.linspace(-2, 0, 16), np.linspace(0, 2, 16)], 2)
    control = np.array([response(a, a, y, d), response(a, a+.1, y, d), response(a, a+.1*(2*y-1), y, d)])[:, 4:]
    for j, label in enumerate(('Zero change', 'Common +0.1 shift', '+0.1 class separation')):
        axes[1].bar(np.arange(4)+(j-1)*.24, control[j], width=.24, label=label)
    axes[1].set(xticks=range(4), xticklabels=['Separation', 'Preference', 'Log spread', 'Boundary'], ylabel='Synthetic response', title='Fixed-reader response interpretation'); axes[1].legend(fontsize=8)
    save(fig, '01-design-and-response-controls')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained'); axes[0, 0].axis('off')
    axes[0, 0].text(.5, .5, f"Physical SSL updates: {s['physical_SSL']:,}\nEncoder passes: {s['encoder_passes']}\nFeature forwards: {s['image_forwards']:,}\n128-step traces and full forks exact\nAll proposal repeats exact\nAll zero and repeated features exact\nNo additional reader fitting", ha='center', va='center', fontsize=11)
    axes[0, 1].scatter(groups[:, 2]+np.linspace(-.08, .08, 72), [v['cosine'] for v in audits], c=diets, cmap='coolwarm', s=20)
    axes[0, 1].set(xticks=range(1, 5), xlabel='Context / color = pre-diet', ylabel='Encoder displacement cosine', title='Current versus replay proposal directions')
    for j, ax in enumerate([axes[0, 2], *axes[1]]):
        ax.scatter(2*quarter[:, j+4], primary[:, j+4], c=diets, cmap='coolwarm', s=18, alpha=.7)
        lo = min((2*quarter[:, j+4]).min(), primary[:, j+4].min()); hi = max((2*quarter[:, j+4]).max(), primary[:, j+4].max())
        ax.plot([lo, hi], [lo, hi], color='gray'); ax.set(xlabel='2 x quarter-step contrast', ylabel='Half-step contrast', title=['Night separation', 'Night preference', 'Night log spread', 'Night boundary mass'][j])
    fig.supxlabel('Same raw Adam direction at two matched norms; nonlinearity and boundary crossings are descriptive, not selection criteria', fontsize=9)
    save(fig, '02-reconstruction-and-dose-stability')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    names = ['Night separation response', 'Night preference response', 'Night log-spread response', 'Night boundary-mass response']
    for j, ax in enumerate(axes.ravel()):
        for diet, label, color in ((0, 'Current pre-diet', 'C0'), (1, 'Balanced pre-diet', 'C1')):
            for c in range(1, 5):
                keep = (diets == diet) & (groups[:, 2] == c); value = primary[keep, j+4]; x = c+(.13 if diet else -.13)
                ax.scatter(x+np.linspace(-.06, .06, len(value)), value, s=22, color=color, alpha=.7, label=label if c == 1 else None)
                ax.plot([x-.09, x+.09], [value.mean()]*2, color=color, linewidth=3)
        ax.axhline(0, color='gray', linewidth=.7); ax.set(xticks=range(1, 5), xlabel='All continuation contexts', ylabel='Replay minus current response', title=names[j]); ax.legend(fontsize=8)
    fig.supxlabel('Primary half-step / lambda0.01 / mean of six support pools; fixed readers and baseline score scale', fontsize=9)
    save(fig, '03-response-by-context-and-diet')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    labels = ['Mean', 'Cheap', '+ Static', 'Response\nonly', '+ Response', '+ Static +\nresponse', '+ CKA +\nresponse']
    axes[0].bar(range(7), [s['primary_MSE'][m] for m in MODELS], color=['gray']*5+['C0', 'C2'])
    axes[0].set(xticks=range(7), xticklabels=labels, ylabel='Strictly held-out MSE (pp squared)', title='Original fixed-reader night-highway benefit')
    yy = target[:, 1, 6]; pp = prediction[0, :, :, 1, 6]
    for b, base in enumerate(('cheap', 'static')):
        change = (pp[:, MODELS.index(base)]-yy)**2-(pp[:, MODELS.index('static_response_added')]-yy)**2
        vals = [change[groups[:, axis] == v].mean() for axis in (0, 1) for v in np.unique(groups[:, axis])]
        axes[1].bar(np.arange(6)+(b-.5)*.36, vals, width=.36, label=f'Gain versus {base}')
    axes[1].axhline(0, color='gray'); axes[1].set(xticks=range(6), xticklabels=[f'Seed\n{v}' for v in np.unique(groups[:, 0])]+[f'Source\n{v}' for v in np.unique(groups[:, 1])], ylabel='Baseline MSE minus primary MSE', title='Both baselines must improve in each group'); axes[1].legend(fontsize=8)
    fig.supxlabel('36 seed AND source AND context exclusions; 24 training /2 test states; fixed ridge10 and train-only preprocessing', fontsize=9)
    save(fig, '04-heldout-MSE-and-group-stability')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
    for diet in (0, 1):
        keep = diets == diet; axes[0].scatter(yy[keep], pp[keep, 5], label='Balanced pre-diet' if diet else 'Current pre-diet', s=25)
    lo = min(yy.min(), pp[:, 5].min()); hi = max(yy.max(), pp[:, 5].max()); axes[0].plot([lo, hi], [lo, hi], color='gray')
    axes[0].set(xlabel='Observed replay benefit (pp)', ylabel='Held-out forecast (pp)', title='Primary static + response model'); axes[0].legend(fontsize=8)
    for b, base in enumerate(('cheap', 'static')):
        vals = [100*v['baselines'][base]['MSE_reduction_fraction'] for v in s['screens'].values()]
        axes[1].barh(np.arange(10)+(b-.5)*.35, vals, height=.35, label=f'Versus {base}')
    axes[1].axvline(10, linestyle='--', color='C3'); axes[1].set(yticks=range(10), yticklabels=['Mean half/.01', 'Mean quarter/.01', 'Mean half/.001', 'Mean quarter/.001']+[f'Pool {i} half/.01' for i in range(6)], xlabel='MSE reduction (%)', title='All predeclared dose/prior/pool sensitivities'); axes[1].invert_yaxis(); axes[1].legend(fontsize=8)
    fig.supxlabel('Repeated settings share states and support images; not independent confirmation or candidates for post-hoc selection', fontsize=9)
    save(fig, '05-forecast-and-dose-prior-pool-sensitivity')
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), layout='constrained')
    grid = np.array([[float(next(v['r'] for v in corr if int(v['diet']) == d and int(v['axis']) == k)) for k in range(8)] for d in (0, 1)])
    im = axes[0].imshow(grid, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    axes[0].set(yticks=[0, 1], yticklabels=['Current pre-diet', 'Balanced pre-diet'], xticks=range(8), xticklabels=[f'{d}\n{a}' for d in ('Day', 'Night') for a in ('separation', 'preference', 'log spread', 'boundary')], title='Exploratory response / future-benefit association (36 known states per diet)')
    for (r, c), value in np.ndenumerate(grid):
        axes[0].text(c, r, f'{value:+.2f}', ha='center', va='center', color='white' if abs(value) > .6 else 'black')
    fig.colorbar(im, ax=axes[0], label='Pearson r')
    for j, pol in enumerate(('always_current', 'always_replay', 'cheap', 'static', 'static_response_added')):
        values = [float(next(v['value_pp'] for v in shadow if v['model'] == pol and int(v['component']) == k)) for k in (3, 4, 5, 6)]
        axes[1].bar(np.arange(4)+(j-2)*.15, values, width=.15, label=pol.replace('_', ' '))
    axes[1].axhline(0, color='gray'); axes[1].set(xticks=range(4), xticklabels=['Day city', 'Day highway', 'Night city', 'Night highway'], ylabel='Value versus always-current (pp)', title='Shadow policy chooses replay on positive held-out primary-target forecast'); axes[1].legend(ncol=3, fontsize=8)
    fig.supxlabel('Local SSL response is not an intervention on an isolated health axis; class tradeoffs and long-horizon transfer remain open', fontsize=9)
    save(fig, '06-associations-and-shadow-class-values')
    text = ('SYNTHETIC ENGINEERING ONLY\n\n' if engineering else '')+'''# Pre-action local functional response profile

All72 known study43 states. Reconstruct the exact complete learner from its
recorded128-update prefix before probing copies. Current batch127 and historical
replay were already available before the action fork; future frames are excluded.
Paired current/replay inherited-Adam proposals repeat exactly. Quarter/half of
the pre-action actual-step norm are evaluated along the same proposal direction.
Zero and perturbed support features each repeat exactly. The fixed study51 OOF
readers measure held-out validation support, image-disjoint from the old query.
No new reader optimization or future continuation training is performed.

Class separation and preference responses use the baseline within-class score
scale; log spread changes and boundary-mass changes retain their definitions.
The replay-minus-current response is sealed before loading old future benefits.
Original fixed-reader night-highway benefit is primary; joint and all7 class/
domain targets remain saved. Strict seed AND source AND context exclusions,
24 training/2 test states, ridge10 and train-only standardization. Primary
static+response must improve10% over BOTH cheap and static, improving each seed
and source. All ten fixed dose/prior/pool panels are reported without selection.

Known-cohort retrospective development only, shared histories/warm weights and
unknown capture sessions. Overlapping support pools are repeated measurements.
A response to one recently seen batch need not predict384 future updates; an
SSL probe changes the model, not an isolated health component. Associations are
exploratory, not causal evidence, fresh confirmation or controller qualification.
Observed findings follow after numerical and visual verification.
'''
    (out/'report.md').write_text(text, encoding='utf-8')
    code = f'''from pathlib import Path
import json, numpy as np
from cafl4ds.functional_response_profile import MODELS
p=Path({str(out)!r});s=json.loads((p/'summary.json').read_text())
with np.load(p/'analysis-arrays.npz') as z:
    y=z['target'][:,1,6];pred=z['predictions'][0,:,:,1,6]
    for j,m in enumerate(MODELS):
        assert abs(float(np.mean((pred[:,j]-y)**2))-s['primary_MSE'][m])<1e-9
assert s['states']==72
print(json.dumps(s,indent=2))
'''
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, 'response-notebook-reconstruction', 'exec'), {})
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)},
        {'cell_type': 'code', 'metadata': {}, 'source': code.splitlines(keepends=True), 'execution_count': 1,
         'outputs': [{'output_type': 'stream', 'name': 'stdout', 'text': stream.getvalue().splitlines(keepends=True)}]}]
    for i, p in enumerate(paths):
        cells += [{'cell_type': 'markdown', 'metadata': {}, 'source': [f'## Figure {i+1}: {p.stem}\n']},
            {'cell_type': 'code', 'metadata': {}, 'source': [f'from PIL import Image\nfigure_image=Image.open({str(p)!r})\nfigure_image\n'], 'execution_count': i+2,
             'outputs': [{'output_type': 'display_data', 'metadata': {}, 'data': {'image/png': base64.b64encode(p.read_bytes()).decode(), 'text/plain': [p.name]}}]}]
    write(notebook, {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name': 'Python3', 'language': 'python', 'name': 'python3'}}, 'cells': [{'id': f'functional-response-{i}', **c} for i, c in enumerate(cells)]})
    for cell, p in zip(read(notebook)['cells'][3::2], paths, strict=True):
        assert base64.b64decode(cell['outputs'][0]['data']['image/png']) == p.read_bytes()
    return {'passed': True, 'visual_QA_pending': True, 'saved_reconstruction_executed': True,
        'embedded_PNGs_checked': 6, 'hashes': {str(p): digest(p) for p in paths+[notebook, out/'report.md']}}
