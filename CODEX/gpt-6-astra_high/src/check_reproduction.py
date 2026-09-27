"""Snapshot and compare numerical outputs across an empty-cache reproduction."""
import argparse
import json
import time
import subprocess
from pathlib import Path
from core import ROOT,sha256

def products():
    out={}
    for name in ['summary.json','selection.json','verification.json']:
        v=json.loads((ROOT/'results'/name).read_text())
        for k in ['runtime_seconds','git_commit','fit_commit']:v.pop(k,None)
        out[name]=v
    for name in ['operating_points.csv','development_operating_points.csv','per_hdu.csv','test_acquisitions.csv',
                 'ablations.csv','reconstruction_checks.csv','persistence.csv','per_group.csv','temporal.csv','spectrum.csv','report_assets.json']:
        out[name]=sha256(ROOT/'results'/name)
    out['manifest.json']=sha256(ROOT/'data/manifest.json')
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['snapshot','compare']);p.add_argument('path',type=Path);a=p.parse_args()
    now=products()
    if a.mode=='snapshot':a.path.write_text(json.dumps({'products':now,'time':time.time()},indent=2));print('Snapshot saved.');return
    previous=json.loads(a.path.read_text()); mismatch=[k for k in now if now[k]!=previous['products'][k]]
    result=dict(passed=not mismatch,mismatches=mismatch,compared_products=list(now),
        numerical_json_exact_equal=True if not mismatch else None,
        csv_manifest_and_figure_hashes_byte_equal=True if not mismatch else None,
        elapsed_seconds_since_snapshot=time.time()-previous['time'],
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        workflow='SKIPPER_DATA_DIR=../pp_proc bash CODEX/gpt-6-astra_high/reproduce.sh',
        scope='Empty analysis cache, same existing Python environment; input SHA-256 checked. Runtime/git metadata excluded. Not a new-machine/environment install test.')
    (ROOT/'results/reproduction_check.json').write_text(json.dumps(result,indent=2)+'\n')
    if mismatch:raise SystemExit(f'Reproduction mismatches: {mismatch}')
    print('Exact reproduction passed:',len(now),'products.')

if __name__=='__main__':main()
