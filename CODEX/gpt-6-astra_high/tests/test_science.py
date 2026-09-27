"""Small independent checks of definitions, not a second scientific review."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from core import reconstruct, sideband_maps, pixel_time, tile_sum
from experiment import metrics, frontier, weighted_threshold


def flood_fill(q,t=0.7):
    """Independent Python graph walk, no scipy.label or bincount."""
    visited=set(); components=[]
    for y in range(q.shape[0]):
        for x in range(q.shape[1]):
            if q[y,x]<=t or (y,x) in visited: continue
            pending=[(y,x)]; visited.add((y,x)); pixels=[]
            while pending:
                yy,xx=pending.pop(); pixels.append((yy,xx))
                for dy in [-1,0,1]:
                    for dx in [-1,0,1]:
                        p=(yy+dy,xx+dx)
                        if 0<=p[0]<q.shape[0] and 0<=p[1]<q.shape[1] and p not in visited and q[p]>t:
                            pending.append(p); visited.add(p)
            components.append((sum(float(q[p]) for p in pixels),len(pixels)))
    return sorted(components)


class ScientificChecks(unittest.TestCase):
    def test_components_independent(self):
        rng=np.random.default_rng(912)
        for _ in range(12):
            q=rng.choice([0,0.7,1,2,4,6,30],size=(16,16),p=[.65,.1,.1,.05,.04,.03,.03])
            _,ev=reconstruct(q)
            self.assertEqual(sorted(zip(ev['charge'],ev['size'])),flood_fill(q))

    def test_roi_blind_feature_injection(self):
        q=np.zeros((64,64),dtype=float);q[8,8]=1;q[40,40]=21;q[5,32]=-1
        lab,ev=reconstruct(q);before=sideband_maps(q,lab,ev)
        for charge in [3.5,4,5,6,7,7.499]:
            injected=q.copy(); injected[24,24]=charge/2;injected[24,25]=charge/2
            lab,ev=reconstruct(injected)
            np.testing.assert_array_equal(before,sideband_maps(injected,lab,ev))

    def test_exposure_closed_form(self):
        h=dict(DATESTART='2026-01-01T00:00:00',DATEEND='2026-01-01T01:00:00',NROW=515,NCOL=3200,EXPOSURE=0)
        e,d=pixel_time(h)
        # Arithmetic series, independent of production map implementation.
        expected=(3072*3200*511*512/2 + 512*(8+3079)*3072/2)*(3600/(515*3200))/86400
        self.assertAlmostEqual(float(e.sum()),expected,places=7)
        self.assertTrue(e[-1,0]>e[0,-1]); self.assertEqual(d,3600)
        self.assertAlmostEqual(float(tile_sum(e,32).sum()),expected,places=7)

    def test_metrics_not_counts(self):
        m=metrics(100,50,100,50)
        self.assertEqual(m['background_ratio'],1);self.assertEqual(m['F_equal'],.5)
        m=metrics(100,80,100,40)
        self.assertEqual(m['background_ratio'],.5);self.assertEqual(m['F_equal'],1.6)

    def test_threshold_and_frontier(self):
        self.assertEqual(weighted_threshold(np.arange(4),np.ones(4),.5),1)
        a=[dict(exposure_fraction=1,background_ratio=1),dict(exposure_fraction=.9,background_ratio=.5),dict(exposure_fraction=.8,background_ratio=.7)]
        frontier(a);self.assertEqual([r['pareto'] for r in a],[True,True,False])

if __name__=='__main__': unittest.main()
