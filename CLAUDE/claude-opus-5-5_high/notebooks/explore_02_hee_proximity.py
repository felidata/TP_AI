"""Exploration (dev set, even acquisitions, S101 HDU0 excluded): ROI clusters vs distance and direction to
high-energy deposits (Q>=100 e-) in the same image. Writes results/explore_hee_proximity*.{csv,json}."""
import sys; sys.path.insert(0,'src')
import numpy as np
from scipy import ndimage
from skbg import config, dataset, cache, exposure
cfg=config.load_config(); cd=config.cache_dir(cfg)
recs=[r for r in dataset.records(cfg) if r['usable'] and dataset.split_of(r['img'],cfg)=='dev']
SH=(512,3072); bins=np.array([0,1,2,3,5,8,12,20,30,50,80,120,200,400,4000])
res={}
for r in [x for x in recs if x['img']%2==0]:
    z=cache.load_image(cd,r); meta=cache.load_meta(cd,r)
    E=exposure.exposure_map_days(meta['timing'],(0,512),(8,3080))
    for h in range(4):
        if r['module']=='S101' and h==0: continue
        Q=z[f'cl{h}_Q']; 
        # HEE footprint: pixels of clusters with Q>=100 -> rebuild from bbox is lossy; use pixels n>=1 inside those clusters via label
        n=np.zeros(SH,np.int32); n[z[f'pix{h}_r'],z[f'pix{h}_c']]=z[f'pix{h}_n']
        lab,_=ndimage.label(n>0,structure=np.ones((3,3)))
        heelab=np.flatnonzero(Q>=100)+1
        foot=np.isin(lab,heelab)
        d,ind=ndimage.distance_transform_edt(~foot,return_indices=True)
        dy=np.arange(512)[:,None]-ind[0]; dx=np.arange(3072)[None,:]-ind[1]
        s=(Q>=4)&(Q<=7); rc=np.rint(z[f'cl{h}_rc'][s]).astype(int); cc=np.rint(z[f'cl{h}_cc'][s]).astype(int)
        key=r['module']
        a=res.setdefault(key,{'ev':np.zeros(len(bins)-1),'ex':np.zeros(len(bins)-1),'evdir':np.zeros(4),'exdir':np.zeros(4)})
        a['ev']+=np.histogram(d[rc,cc],bins)[0]
        a['ex']+=np.histogram(d.ravel(),bins,weights=E.ravel())[0]
        # direction for d in (1,30): classes: right(+x), left, up(+y), down
        def cls(dx,dy):
            c=np.where(np.abs(dx)>=np.abs(dy),np.where(dx>0,0,1),np.where(dy>0,2,3)); return c
        near=(d>0)&(d<=20)
        a['exdir']+=np.bincount(cls(dx[near],dy[near]),weights=E[near],minlength=4)
        ne=(d[rc,cc]>0)&(d[rc,cc]<=20)
        a['evdir']+=np.bincount(cls(dx[rc,cc][ne],dy[rc,cc][ne]),minlength=4)
import pandas as pd, json
rows=[]
for k,a in res.items():
    far_=a['ev'].sum()/a['ex'].sum()
    for i in range(len(bins)-1):
        rows.append({'module':k,'d_lo':int(bins[i]),'d_hi':int(bins[i+1]),'events':int(a['ev'][i]),'exposure_frac':a['ex'][i]/a['ex'].sum(),
                     'rate_ratio':(a['ev'][i]/a['ex'][i])/far_ if a['ex'][i]>0 else None})
pd.DataFrame(rows).to_csv('results/explore_hee_proximity.csv',index=False)
json.dump({k:{'direction_rate_ratio_right_left_up_down':((a['evdir']/a['exdir'])/(a['ev'].sum()/a['ex'].sum())).tolist(),
              'frac_roi_d_le_20':a['ev'][:7].sum()/a['ev'].sum(),'frac_exposure_d_le_20':a['ex'][:7].sum()/a['ex'].sum()} for k,a in res.items()},
          open('results/explore_hee_proximity_direction.json','w'),indent=1)
for k,a in res.items():
    rate=a['ev']/a['ex']; far=a['ev'].sum()/a['ex'].sum()
    print(k,'rate ratio vs quadrant-mean:')
    for i in range(len(bins)-1): print(f'  d in [{bins[i]},{bins[i+1]}): events {a["ev"][i]:.0f}  exp_frac {a["ex"][i]/a["ex"].sum():.4f}  ratio {rate[i]/far:.2f}')
    print('  direction (right,left,up,down) rate ratio', np.round((a['evdir']/a['exdir'])/far,2), a['evdir'])
    print('  fraction of ROI events with d<=20:', a['ev'][:7].sum()/a['ev'].sum(), 'exposure frac', a['ex'][:7].sum()/a['ex'].sum())
