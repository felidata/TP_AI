"""Audit final numbers with separate arithmetic; perform predefined stress checks.

This is an independent calculation within the same agent session, not an
independent researcher/model review. No check chooses a new operating point.
"""
import json
import os
import time
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from core import ROOT, CONFIG, calibrate, reconstruct, sideband_maps
from experiment import load_records, make_arrays, weighted_threshold, metrics, write_csv


def main():
    t0=time.monotonic()
    summary=json.loads((ROOT/'results/summary.json').read_text())
    frozen=json.loads((ROOT/'results/selection.json').read_text())
    records=load_records(True); saved=np.load(ROOT/'cache/selected_masks.npz')
    keep=saved['keep']; size=frozen['tile_size']; e=saved['exposure']; n=saved['roi']
    test=np.array([r['meta']['split']=='test' for r in records])
    assert len({r['meta']['image_id'] for r in records if r['meta']['split']=='test'})==22
    assert len(records)==872 and int(test.sum())==176
    assert len({r['meta']['image_id'] for r in records})==109
    assert all(sum(r['meta']['image_id']==i for r in records)==8 for i in range(22,131))
    e0=e1=0.; n0=n1=0; spatial=[]
    # Exposure rederived by arithmetic-series average readout index for each tile.
    for i,r in enumerate(records):
        if not test[i]:continue
        h=r['meta']; dt=h['readout_seconds']/(515*3200)/86400
        for ty in range(512//size):
            for tx in range(3072//size):
                mean_index=3200*(ty*size+(size-1)/2)+8+tx*size+(size-1)/2
                cell=(mean_index*dt+h['nominal_exposure_seconds']/86400)*size**2
                e0+=cell
                if keep[i,ty,tx]:e1+=cell
        selected=(r['charge']>=3.5)&(r['charge']<7.5)
        for y,x in zip(r['y'][selected],r['x'][selected]):
            n0+=1
            if keep[i,int(y)//size,int(x)//size]:n1+=1
    assert np.isclose(e0,summary['baseline_exposure'],rtol=1e-12)
    assert np.isclose(e1,summary['retained_exposure'],rtol=1e-12)
    assert n0==summary['baseline_roi_events'] and n1==summary['final_roi_events']
    assert np.isclose((n1/e1)/(n0/e0),summary['background_ratio'])
    assert np.isclose((e1/e0)**2*n0/n1,summary['F_equal'])

    # Retain the original sample; assess the whole paired acquisition with a gain transient.
    without114=test & np.array([r['meta']['image_id']!=114 for r in records])
    transient=dict(
        observation='S99 image114 has shifted raw 1e peaks, fitted gains170-181 ADU/e, and a10-second longer readout than its paired module. Cause unknown; primary sample is unchanged.',
        gains_adu_per_e={str(r['meta']['hdu']):r['meta']['gain_adu_per_e'] for r in records if r['meta']['module']=='S99' and r['meta']['image_id']==114},
        paired_acquisition_exclusion=metrics(e[without114].sum(),(e*keep)[without114].sum(),n[without114].sum(),(n*keep)[without114].sum()),
        interpretation='Post-test diagnostic only, not a new selection; removes both modules and all eight HDUs.')
    (ROOT/'results/acquisition114_check.json').write_text(json.dumps(transient,indent=2)+'\n')

    # Persistent structure measured only within training: first/second half.
    for group in sorted({(r['meta']['module'],r['meta']['hdu']) for r in records}):
        rr=[r for r in records if (r['meta']['module'],r['meta']['hdu'])==group and r['meta']['split']=='train']
        a=np.stack([r['features'][...,0] for r in rr[:32]]).mean(0)
        b=np.stack([r['features'][...,0] for r in rr[32:]]).mean(0)
        spatial.append(dict(module=group[0],hdu=group[1],training_half_sideband_spearman=float(spearmanr(a.ravel(),b.ravel()).statistic)))
    write_csv(ROOT/'results/persistence.csv',spatial)

    # Matched operating-point ablations of the selected algorithm, not retuned.
    x,_,_,tr,_=make_arrays(records,size)
    ablations=[]
    feature_sets={'full':[0,1,2,3,4,5], 'no_persistence':[0,1,2,5],
                  'no_high_charge':[0,2,3,5], 'no_negative_tail':[0,1,3,4,5], 'no_neighbor':[0,1,2,3,4]}
    for name,cols in feature_sets.items():
        xx=x[...,cols]; model=IsolationForest(n_estimators=CONFIG['isolation_trees'],max_samples=CONFIG['isolation_max_samples'],random_state=CONFIG['seed'],n_jobs=1)
        model.fit(xx[tr].reshape(-1,len(cols)))
        score=-model.score_samples(xx.reshape(-1,len(cols))).reshape(xx.shape[:-1])
        threshold=weighted_threshold(score[tr],e[tr],frozen['training_removed_target'])
        k=score<=threshold
        if name=='full':np.testing.assert_array_equal(keep,k)
        for split in ['validation','test']:
            ix=np.array([r['meta']['split']==split for r in records])
            ablations.append(dict(ablation=name,split=split,**metrics(e[ix].sum(),(e*k)[ix].sum(),n[ix].sum(),(n*k)[ix].sum())))
        print('Ablation',name,flush=True)
    write_csv(ROOT/'results/ablations.csv',ablations)

    # Deliberately destroy spatial alignment, preserving mask area within HDUs.
    rng=np.random.default_rng(CONFIG['seed']); null=[]
    for _ in range(200):
        en=nn=0
        for i in np.flatnonzero(test):
            shifted=np.roll(keep[i],(rng.integers(keep.shape[1]),rng.integers(keep.shape[2])),axis=(0,1))
            en+=np.sum(e[i]*shifted);nn+=np.sum(n[i]*shifted)
        null.append((nn/en)/(n0/e0))

    # Exposure-weighted point-signal acceptance check with Poisson-independent positions.
    prob=e[test].ravel();prob=prob/prob.sum()
    idx=rng.choice(prob.size,size=100000,p=prob)
    acceptance=float(np.mean(keep[test].ravel()[idx]))
    binomial_se=float(np.sqrt(summary['exposure_fraction']*(1-summary['exposure_fraction'])/len(idx)))
    assert abs(acceptance-summary['exposure_fraction'])<5*binomial_se

    # Real-image ROI-erasure invariance and frozen-mask reconstruction systematics.
    data=Path(os.environ.get('SKIPPER_DATA_DIR','../pp_proc'))
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    lookup={(r['meta']['module'],r['meta']['image_id'],r['meta']['hdu']):i for i,r in enumerate(records)}
    variants={'nominal':(1.,.7),'gain_plus_1pct':(1.01,.7),'gain_minus_1pct':(.99,.7),'threshold_0.6e':(1.,.6),'threshold_0.8e':(1.,.8)}
    counts={k:[0,0] for k in variants}; invariance=[]
    for file in manifest:
        if file['split']!='test':continue
        with fits.open(data/file['filename'],memmap=False) as hdul:
            for h,hdu in enumerate(hdul):
                q,_=calibrate(hdu.data);i=lookup[(file['module'],file['image_id'],h)]
                if file['image_id']==109:
                    lab,ev=reconstruct(q);features=sideband_maps(q,lab,ev)
                    roi=(ev['charge']>=3.5)&(ev['charge']<7.5)
                    erase=np.r_[False,roi][lab]; altered=q.copy();altered[erase]=0
                    l2,e2=reconstruct(altered)
                    identical=bool(np.array_equal(features,sideband_maps(altered,l2,e2)))
                    assert identical
                    invariance.append(dict(module=file['module'],hdu=h,erased_roi_events=int(roi.sum()),identical_features=identical))
                for name,(gain,threshold) in variants.items():
                    _,ev=reconstruct(q/gain,threshold)
                    roi=(ev['charge']>=3.5)&(ev['charge']<7.5)
                    counts[name][0]+=int(roi.sum())
                    counts[name][1]+=int(keep[i,ev['y'][roi]//size,ev['x'][roi]//size].sum())
        if file['image_id']%5==0:print('Reconstruction checks',file['module'],file['image_id'],flush=True)
    assert counts['nominal']==[n0,n1]
    stress=[dict(variation=k,**metrics(e0,e1,*v)) for k,v in counts.items()]
    write_csv(ROOT/'results/reconstruction_checks.csv',stress)
    result=dict(independent_exposure_relative_difference=float(e1/summary['retained_exposure']-1),
        independent_roi_counts=[n0,n1],real_roi_erasure=invariance,
        shifted_mask_background_ratio_95=np.percentile(null,[2.5,50,97.5]).tolist(),
        injected_uniform_point_acceptance=acceptance,injection_binomial_se=binomial_se,
        synthetic_injection_note='100000 signal anchors drawn proportional to readout exposure; signal reconstruction efficiency is not inferred.',
        checks_passed=True,runtime_seconds=time.monotonic()-t0,
        limitations='Independent arithmetic/code paths, same model/agent. No independent scientist or alternate-model review.')
    (ROOT/'results/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
