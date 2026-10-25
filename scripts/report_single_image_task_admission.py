"""Saved offline / online / full comparison, updated at each B checkpoint."""
import base64
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from cafl4ds.single_image_extended_gate import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OFFLINE = ROOT/'outputs/single-image-heuristic/20261024-no-rollback/trajectories/fresh_00/seed_5003/heuristic'


def report(root):
    lock=json.loads((root/'input-lock.json').read_text()); status=json.loads((root/'status.json').read_text())
    branch=root/'trajectories/fresh_00/seed_5003/heuristic'
    common=json.loads((branch.parent/'common/readout.json').read_text())
    records={}; traces={}
    for name,path in [('Offline heuristic (A)',OFFLINE),('Online heuristic (B)',branch)]:
        saved=load_checkpoint(path)
        if saved:
            gen,records[name]=saved
            with np.load(gen/'trace.npz') as z:
                traces[name]={k:z[k].copy() for k in ['accepted','training','rate','J']}
    fullpath=Path(lock['inherited_full_reference'])
    records['Full training']=json.loads(fullpath.read_text())
    with np.load(fullpath.parent/'trace.npz') as z:
        traces['Full training']={k:z[k].copy() for k in ['accepted','training']}
    start=lock['settings']['start']; horizon=max(384,records.get('Online heuristic (B)',{}).get('done',0))
    colors={'Offline heuristic (A)':'#2466b0','Online heuristic (B)':'#d34e2c','Full training':'#458b65'}
    images=[]
    specs=[('performance',3,3,[
        ('linear_accuracy_balanced','Balanced linear accuracy (%)'),('linear_accuracy_day','Day accuracy (%)'),
        ('linear_accuracy_night','Night accuracy (%)'),('knn_accuracy_balanced','Balanced kNN accuracy (%)'),
        ('linear_ce_balanced','Balanced linear CE'),('knn_ce_balanced','Balanced kNN CE'),
        ('linear_ce_day','Day linear CE'),('linear_ce_night','Night linear CE'),('updates','Accepted updates')]),
        ('health',2,3,[('selector_rank','RankMe'),('selector_variance','Variance'),
        ('selector_covariance','Covariance'),('selector_bank_reconstruction','Fixed-bank reconstruction'),
        ('selector_objective','J (0.25 / 0.25)'),('selector_feature_norm','Feature norm')]),
        ('matched_updates',2,3,[('linear_accuracy_balanced','Balanced linear accuracy (%)'),
        ('knn_accuracy_balanced','Balanced kNN accuracy (%)'),('linear_ce_balanced','Balanced linear CE'),
        ('selector_rank','RankMe'),('selector_variance','Variance'),('selector_bank_reconstruction','Reconstruction')])]
    for title,rows,cols,metrics in specs:
        fig,axes=plt.subplots(rows,cols,figsize=(15,rows*3.3),constrained_layout=True)
        matched=title=='matched_updates'
        common_updates=min(int(tr['accepted'][:horizon].sum()) for tr in traces.values())
        for ax,(metric,label) in zip(axes.flat,metrics):
            for name,record in records.items():
                tr=traces[name]; cumulative=np.r_[0,np.cumsum(tr['accepted'])]
                if metric=='updates':
                    end=min(horizon,record.get('done',record.get('offered',0)))
                    ax.plot(np.arange(end+1),cumulative[:end+1],label=name,color=colors[name]);continue
                pairs=[]
                for k,v in record['checkpoints'].items():
                    offered=int(k)-start
                    if offered<0 or offered>horizon or metric not in v['metrics']:continue
                    x=int(cumulative[offered]) if matched else offered
                    if matched and x>common_updates:continue
                    pairs.append((x,v['metrics'][metric]))
                if pairs:
                    pairs.sort(); ax.plot(*np.array(pairs).T,label=name,color=colors[name],linewidth=1.5)
            if metric in common['metrics']:ax.axhline(common['metrics'][metric],color='gray',ls=':',lw=.8)
            ax.set(title=label,xlabel='Accepted updates (common coverage)' if matched else 'Offered continuation images')
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        file=root/f'{title}.png';fig.savefig(file,dpi=120);plt.close(fig)
        images.append((title,base64.b64encode(file.read_bytes()).decode()))
    fig,axes=plt.subplots(3,1,figsize=(13,9),constrained_layout=True)
    for ax,(name,tr) in zip(axes,traces.items()):
        n=min(horizon,records[name].get('done',records[name].get('offered',0)))
        mask=tr['accepted'][:n]; x=np.arange(1,n+1)[mask]
        ax.scatter(x,tr['training'][:n,0][mask],s=2,alpha=.35,color=colors[name])
        ax.set(title=name,xlabel='Offered continuation images',ylabel='Trained-image reconstruction loss');ax.grid(alpha=.2)
    fig.savefig(root/'training_loss.png',dpi=120);plt.close(fig)
    images.append(('training_loss',base64.b64encode((root/'training_loss.png').read_bytes()).decode()))
    text=f'''# Notebook 90: offline versus online versus full

Status: **{status['stage']}**, {status.get('offered',0):,}/57,600 offered, {status.get('accepted',0):,} accepted.

Only B is newly trained. A is the saved Notebook 84 no-rollback heuristic; full is its original saved full-training reference. Same warm parent, stream, augmentation seed 5003 and acceptance RNG seed 835003. J weights 0.25/0.25; fixed bank and existing health/rate rules; no task rollback.

After 128 accepted initialization updates, B multiplies ordinary acceptance probability by 1.2 for predicted immediate development benefit or 0.8 for predicted harm. Bursts and numerical recovery are unchanged. The online predictor learns only from B's accepted outcomes. No C run.

Plots stop at B's current checkpoint horizon. Dotted lines mark the warm parent. Full training consumes more updates; its curve is a reference, not a compute-matched control. Matched-update plots show only common observed coverage and do not remove differences in image selection. These are one-seed paired results, not a multi-seed conclusion.
'''
    (root/'report.md').write_text(text,encoding='utf-8')
    page='<meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Notebook 90: offline / online / full</title><body style="font-family:system-ui;padding:20px"><pre style="white-space:pre-wrap;font-family:inherit">'+text+'</pre>'
    cells=[{'cell_type':'markdown','id':'report','metadata':{},'source':text.splitlines(True)}]
    for name,img in images:
        page+=f'<h2>{name.replace("_"," ")}</h2><img width="100%" src="data:image/png;base64,{img}">'
        cells.append({'cell_type':'markdown','id':name,'metadata':{},'source':[f'![{name}](attachment:{name}.png)'],
                      'attachments':{f'{name}.png':{'image/png':img}}})
    tmp=root/'live.tmp.html';tmp.write_text(page,encoding='utf-8');tmp.replace(root/'live.html')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{},'cells':cells}
    (ROOT/'notebooks/sums/90_single_image_online_admission.ipynb').write_text(json.dumps(nb),encoding='utf-8')
