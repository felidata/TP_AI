"""Inventory and baseline; deliberately supports processing development data first."""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from core import ROOT, extract_file


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--include-test',action='store_true')
    p.add_argument('--limit',type=int)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    data=Path(os.environ.get('SKIPPER_DATA_DIR','../pp_proc'))
    files=sorted(data.glob('*.fits'))
    if not files: raise SystemExit('No FITS data found. Set SKIPPER_DATA_DIR.')
    import re
    if not args.include_test: files=[f for f in files if int(re.search(r'_img(\d+)',f.name).group(1))<=108]
    if args.limit: files=files[:args.limit]
    cache=ROOT/'cache'; cache.mkdir(exist_ok=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(extract_file,[(str(f),str(cache)) for f in files]):
            rows.append(row)
            print(f"{len(rows)}/{len(files)} {row['module']} {row['image_id']}: ROI={sum(h['roi_events'] for h in row['hdus'])}",flush=True)
    name='manifest.json' if args.include_test else 'development_manifest.json'
    (ROOT/'data'/name).write_text(json.dumps(rows,indent=2)+'\n')
    hdus=[h for r in rows for h in r['hdus']]
    baseline=dict(files=len(rows),hdus=len(hdus),exposure_pixel_days=sum(h['exposure_pixel_days'] for h in hdus),roi_events=sum(h['roi_events'] for h in hdus))
    baseline['roi_rate_per_million_pixel_days']=1e6*baseline['roi_events']/baseline['exposure_pixel_days']
    (ROOT/'results'/('baseline_all.json' if args.include_test else 'baseline_development.json')).write_text(json.dumps(baseline,indent=2)+'\n')
    print(baseline)

if __name__=='__main__': main()
