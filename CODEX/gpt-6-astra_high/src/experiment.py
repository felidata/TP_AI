"""Train sideband-only patch anomaly detectors; select on validation, freeze, test.

Only evaluate() receives ROI counts; score construction never sees ROI labels.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import time

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.ensemble import IsolationForest
from core import ROOT, CONFIG

METHODS = ['persistent_sideband', 'dynamic_robust', 'combined_robust',
           'isolation_forest', 'high_only', 'low_only', 'noise_only']
ELIGIBLE = ['combined_robust', 'isolation_forest', 'dynamic_robust']


def write_csv(path, rows):
    with open(path,'w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def load_records(include_test=False):
    manifest = ROOT/'data'/('manifest.json' if include_test else 'development_manifest.json')
    records=[]
    for file in json.loads(manifest.read_text()):
        if file['split']=='test' and not include_test: continue
        for h in file['hdus']:
            z=np.load(ROOT/'cache'/f"{h['module']}_{h['image_id']}_{h['hdu']}.npz")
            records.append(dict(meta=h,**{k:z[k] for k in z.files if k!='metadata'}))
    return sorted(records,key=lambda r:(r['meta']['image_id'],r['meta']['module'],r['meta']['hdu']))


def aggregate(a, factor):
    h,w=a.shape[:2]
    return a.reshape(h//factor,factor,w//factor,factor,*a.shape[2:]).sum(axis=(1,3))


def make_arrays(records,size):
    f=size//CONFIG['base_tile']
    raw=np.stack([aggregate(r['features'],f) for r in records]).astype(float)
    exposure=np.stack([aggregate(r['exposure'],f) for r in records])
    # Counts scaled to reference patch area, so diagnostics do not encode ROI.
    density=raw*(1024/(size*size))
    train=np.array([r['meta']['split']=='train' for r in records])
    groups=np.array([r['meta']['module']+'_'+str(r['meta']['hdu']) for r in records])
    persistent=np.zeros_like(density[...,:2])
    for g in np.unique(groups):
        persistent[groups==g]=np.mean(density[train & (groups==g),:,:,:2],axis=0)
    neighbor=np.stack([uniform_filter(r[:,:,0],size=3,mode='reflect') for r in density])[...,None]
    x=np.log1p(np.concatenate([density,persistent,neighbor],axis=-1))
    roi=np.zeros(exposure.shape,dtype=int)
    for i,r in enumerate(records):
        sel=(r['charge']>=3.5)&(r['charge']<7.5)
        np.add.at(roi[i],(r['y'][sel]//size,r['x'][sel]//size),1)
    return x,exposure,roi,train,groups


def score_features(x,train):
    """Unsupervised fit only on training diagnostics, globally across modules.

    A floor on robust scale handles sparse zero-inflated diagnostics.
    """
    xt=x[train].reshape(-1,x.shape[-1])
    med=np.median(xt,axis=0)
    scale=np.maximum((np.percentile(xt,75,axis=0)-np.percentile(xt,25,axis=0))/1.349,0.15)
    z=(x-med)/scale
    forest=IsolationForest(n_estimators=CONFIG['isolation_trees'],
                          max_samples=CONFIG['isolation_max_samples'],
                          contamination='auto',random_state=CONFIG['seed'],n_jobs=1)
    forest.fit(xt)
    fs=-forest.score_samples(x.reshape(-1,x.shape[-1])).reshape(x.shape[:-1])
    return dict(persistent_sideband=x[...,3], dynamic_robust=np.max(z[...,[0,1,2,5]],axis=-1),
                combined_robust=np.max(z,axis=-1), isolation_forest=fs,
                high_only=z[...,1],low_only=z[...,0],noise_only=z[...,2])


def weighted_threshold(score,weight,fraction):
    if fraction==0: return float('inf')
    order=np.argsort(score.ravel(),kind='stable')
    w=weight.ravel()[order]; s=score.ravel()[order]
    idx=np.searchsorted(np.cumsum(w),(1-fraction)*w.sum())
    return float(s[min(idx,len(s)-1)])


def metrics(e0,e1,n0,n1):
    f=e1/e0; ratio=(n1/e1)/(n0/e0) if e1>0 and n0>0 else None
    return dict(baseline_exposure=float(e0),retained_exposure=float(e1),exposure_fraction=float(f),
        baseline_roi_events=int(n0),final_roi_events=int(n1),
        baseline_roi_background_rate=float(n0/e0),final_roi_background_rate=float(n1/e1) if e1>0 else None,
        background_ratio=float(ratio) if ratio is not None else None,
        F_equal=float(f/ratio) if ratio else None,
        sensitivity_proxy=float(np.sqrt(f/ratio)) if ratio else None)


def frontier(rows):
    for r in rows:
        r['pareto']=not any((s['exposure_fraction']>=r['exposure_fraction'] and s['background_ratio']<=r['background_ratio'])
            and (s['exposure_fraction']>r['exposure_fraction']+1e-12 or s['background_ratio']<r['background_ratio']-1e-12) for s in rows)


def evaluate(records,scores,exposure,roi,train,size):
    rows=[]; masks={}
    for name,score in scores.items():
        for frac in CONFIG['removed_training_fractions']:
            threshold=weighted_threshold(score[train],exposure[train],frac)
            keep=score<=threshold
            masks[(name,frac)]=keep
            for split in sorted(set(r['meta']['split'] for r in records)):
                ix=np.array([r['meta']['split']==split for r in records])
                row=dict(method=name,tile_size=size,training_removed_target=frac,
                         threshold=threshold if np.isfinite(threshold) else None,split=split,
                         **metrics(exposure[ix].sum(),(exposure*keep)[ix].sum(),roi[ix].sum(),(roi*keep)[ix].sum()))
                rows.append(row)
    return rows,masks


def bootstrap(per_acquisition):
    """Paired circular moving-block bootstrap, retaining both modules/eight HDUs."""
    a=np.array([[r[k] for k in ['baseline_exposure','retained_exposure','baseline_roi_events','final_roi_events']] for r in per_acquisition])
    n=len(a); rng=np.random.default_rng(CONFIG['seed']); out=[]
    for _ in range(CONFIG['bootstrap_replicates']):
        starts=rng.integers(n,size=int(np.ceil(n/CONFIG['bootstrap_block_acquisitions'])))
        idx=np.concatenate([(s+np.arange(CONFIG['bootstrap_block_acquisitions']))%n for s in starts])[:n]
        e0,e1,n0,n1=a[idx].sum(axis=0)
        out.append([e1/e0,(n1/e1)/(n0/e0),(e1/e0)**2*n0/n1])
    return {k:np.percentile(np.array(out)[:,i],[2.5,97.5]).tolist() for i,k in enumerate(['exposure_fraction','background_ratio','F_equal'])}


def main():
    t0=time.monotonic()
    parser=argparse.ArgumentParser(); parser.add_argument('--test',action='store_true'); args=parser.parse_args()
    if args.test:
        frozen=json.loads((ROOT/'results/selection.json').read_text())
        assert frozen['config_sha256']==hashlib.sha256((ROOT/'configs/analysis.json').read_bytes()).hexdigest()
    records=load_records(args.test); rows=[]
    for size in CONFIG['tile_sizes']:
        print('Fitting tile',size,flush=True)
        x,e,n,tr,groups=make_arrays(records,size)
        scores=score_features(x,tr)
        rr,masks=evaluate(records,scores,e,n,tr,size); rows.extend(rr)
        if args.test and size==frozen['tile_size']:
            keep=masks[(frozen['method'],frozen['training_removed_target'])]
            score=scores[frozen['method']]
            # Refit reproducibility check: frozen threshold is not retuned on test.
            actual=next(r for r in rr if r['method']==frozen['method'] and r['training_removed_target']==frozen['training_removed_target'] and r['split']=='validation')
            assert actual['threshold']==frozen['threshold']
            for k in ['exposure_fraction','background_ratio','F_equal']: assert np.isclose(actual[k],frozen[k],rtol=1e-12)
            np.savez_compressed(ROOT/'cache/selected_masks.npz',keep=keep,score=score,exposure=e,roi=n)
            by_hdu=[]
            for i,r in enumerate(records):
                by_hdu.append(dict(module=r['meta']['module'],image_id=r['meta']['image_id'],hdu=r['meta']['hdu'],split=r['meta']['split'],
                    **metrics(e[i].sum(),(e[i]*keep[i]).sum(),n[i].sum(),(n[i]*keep[i]).sum())))
            write_csv(ROOT/'results/per_hdu.csv',by_hdu)
            testrows=[r for r in by_hdu if r['split']=='test']
            by_acq=[]
            for im in sorted(set(r['image_id'] for r in testrows)):
                sel=[r for r in testrows if r['image_id']==im]
                by_acq.append(dict(image_id=im,**metrics(*[sum(r[k] for r in sel) for k in ['baseline_exposure','retained_exposure','baseline_roi_events','final_roi_events']])))
            write_csv(ROOT/'results/test_acquisitions.csv',by_acq)
    for split in set(r['split'] for r in rows): frontier([r for r in rows if r['split']==split])
    write_csv(ROOT/'results'/('operating_points.csv' if args.test else 'development_operating_points.csv'),rows)
    if not args.test:
        candidates=[r for r in rows if r['split']=='validation' and r['method'] in ELIGIBLE and r['exposure_fraction']>=CONFIG['minimum_validation_exposure']]
        selected=max(candidates,key=lambda r:(r['F_equal'],r['exposure_fraction']))
        selected=dict(selected,config_sha256=hashlib.sha256((ROOT/'configs/analysis.json').read_bytes()).hexdigest(),
            selection_rule='max validation F_equal among independent multivariate candidates with f_exp >= 0.8; static/one-channel methods are comparisons only',
            fit_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        (ROOT/'results/selection.json').write_text(json.dumps(selected,indent=2)+'\n')
        print('SELECTED',json.dumps(selected,indent=2))
    else:
        final=next(r for r in rows if r['split']=='test' and all(r[k]==frozen[k] for k in ['method','tile_size','training_removed_target']))
        manifest=json.loads((ROOT/'data/manifest.json').read_text())
        summary=dict(schema_version='1.0',agent='CODEX',model='gpt-6-astra',effort='high',method_name=final['method'],
            data_files=[r['filename'] for r in manifest],
            split_strategy={'group':'paired module image_id, all HDUs','train':[22,86],'validation':[87,108],'test':[109,130],'chronological':True},
            evaluation_split='test',exposure_unit='pixel day',background_rate_unit='events / pixel day',
            roi_definition='8-connected pixels >0.7e; summed continuous charge in [3.5,7.5); maximum-charge-pixel anchor',
            **{k:v for k,v in final.items() if k not in ['method','split','threshold','pareto']},
            selected_hyperparameters={k:frozen[k] for k in ['method','tile_size','training_removed_target','threshold']},
            bootstrap_95_intervals=bootstrap(by_acq),random_seed=CONFIG['seed'],runtime_seconds=time.monotonic()-t0,
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            physical_exposure_kg_days=None,
            notes=['Physical mass not verified for these modules: no kg-day conversion.',
                   'Exposure assumes uniform readout clock and zero integration before readout.',
                   'Masks accept pre-reconstructed events by maximum-charge anchor, no reclustering.',
                   'sqrt(F_equal) is a conditional background-dominated uniform-signal counting proxy, not a limit.',
                   'Bootstrap describes acquisition fluctuations conditional on fitted method, not calibration/systematic uncertainty.',
                   'git_commit is the code commit used to produce results, not the later artifact commit.'])
        (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        print('TEST',json.dumps({k:v for k,v in summary.items() if k!='data_files'},indent=2))

if __name__=='__main__': main()
