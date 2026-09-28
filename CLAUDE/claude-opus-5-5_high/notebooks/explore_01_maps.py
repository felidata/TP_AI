"""Exploration (dev set only): stacked maps of ROI clusters, 1e- pixels, HEE."""
import sys; sys.path.insert(0,'src')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from skbg import config, dataset, cache
cfg=config.load_config(); cd=config.cache_dir(cfg)
recs=[r for r in dataset.records(cfg) if r['usable'] and dataset.split_of(r['img'],cfg)=='dev']
SH=(512,3072)
maps={}
for r in recs:
    z=cache.load_image(cd,r)
    for h in range(4):
        k=(r['module'],h)
        m=maps.setdefault(k,{'roi':np.zeros(SH),'one':np.zeros(SH),'hee':np.zeros(SH),'mid':np.zeros(SH)})
        Q=z[f'cl{h}_Q']; rc=np.rint(z[f'cl{h}_rc']).astype(int); cc=np.rint(z[f'cl{h}_cc']).astype(int)
        s=(Q>=4)&(Q<=7); np.add.at(m['roi'],(rc[s],cc[s]),1)
        s=(Q>=8)&(Q<100); np.add.at(m['mid'],(rc[s],cc[s]),1)
        n=z[f'pix{h}_n']; pr=z[f'pix{h}_r']; pc=z[f'pix{h}_c']
        s=n==1; np.add.at(m['one'],(pr[s],pc[s]),1)
        s=n>=20; np.add.at(m['hee'],(pr[s],pc[s]),1)
np.savez_compressed(cd/'dev_stack_maps.npz',**{f'{k[0]}_{k[1]}_{q}':v for k,m in maps.items() for q,v in m.items()})
from scipy.ndimage import uniform_filter
fig,axs=plt.subplots(8,3,figsize=(20,26))
for i,(k,m) in enumerate(sorted(maps.items())):
    for j,q in enumerate(['roi','one','hee']):
        b=uniform_filter(m[q],16)*256
        axs[i,j].imshow(np.log10(b+1e-1),origin='lower',aspect='auto',cmap='magma')
        axs[i,j].set_title(f'{k} {q} total={m[q].sum():.0f}')
plt.tight_layout(); plt.savefig('figures/explore_dev_stack_maps.png',dpi=45)
