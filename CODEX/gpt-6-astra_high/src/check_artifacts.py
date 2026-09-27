"""Cross-check published artifacts and all displayed primary numbers."""
import csv
import json
from html.parser import HTMLParser
import numpy as np
from core import ROOT,sha256

def main():
    s=json.loads((ROOT/'results/summary.json').read_text());doc=(ROOT/'report.html').read_text()
    required=['agent','model','effort','method_name','data_files','split_strategy','baseline_exposure','retained_exposure',
              'exposure_fraction','baseline_roi_events','final_roi_events','baseline_roi_background_rate',
              'final_roi_background_rate','background_ratio','F_equal','selected_hyperparameters','random_seed','git_commit','notes']
    assert all(k in s for k in required)
    assert s['physical_exposure_kg_days'] is None
    assert len(s['data_files'])==218
    assert np.isclose(s['final_roi_events']/s['baseline_roi_events'],s['exposure_fraction']*s['background_ratio'])
    spectrum=list(csv.DictReader((ROOT/'results/spectrum.csv').open()))
    for k,key in [('baseline_count','baseline_roi_events'),('cleaned_count','final_roi_events')]:
        assert sum(int(r[k]) for r in spectrum if 3.5<=float(r['charge_bin_center'])<7.5)==s[key]
    groups=list(csv.DictReader((ROOT/'results/per_group.csv').open()))
    assert sum(int(r['final_roi_events']) for r in groups)==s['final_roi_events']
    assert np.isclose(sum(float(r['retained_exposure']) for r in groups),s['retained_exposure'])
    points=list(csv.DictReader((ROOT/'results/operating_points.csv').open()))
    assert len(points)==378
    for r in points:
        assert 0<float(r['exposure_fraction'])<=1 and 0<=int(r['final_roi_events'])<=int(r['baseline_roi_events'])
        assert np.isclose(float(r['F_equal']),float(r['exposure_fraction'])/float(r['background_ratio']))
    for name,digest in json.loads((ROOT/'results/report_assets.json').read_text()).items():assert sha256(ROOT/name)==digest
    class Images(HTMLParser):
        def __init__(self):super().__init__();self.images=[]
        def handle_starttag(self,tag,attrs):
            if tag=='img':self.images.append(dict(attrs))
    p=Images();p.feed(doc);assert len(p.images)==7
    assert all(i['src'].startswith('data:image/png;base64,') and i.get('alt') for i in p.images)
    for number in [f"{s['baseline_roi_background_rate']*1e6:.3f}",f"{s['final_roi_background_rate']*1e6:.3f}",f"{s['F_equal']:.4f}",f"{100*s['exposure_fraction']:.2f}%"]:assert number in doc
    assert 'Negative result:' in doc and 'no independent human' in doc
    print('Artifact checks passed: metric identities, 378 operating points, counts, 7 embedded figures, displayed values.')

if __name__=='__main__':main()
