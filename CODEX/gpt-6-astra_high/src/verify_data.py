"""Check supplied FITS bytes against published manifest before reprocessing."""
import json
import os
from pathlib import Path
from core import ROOT, sha256

def main():
    manifest=ROOT/'data/manifest.json'
    if not manifest.exists():
        print('No final manifest yet; initial audit will create it.'); return
    data=Path(os.environ.get('SKIPPER_DATA_DIR','../pp_proc'))
    rows=json.loads(manifest.read_text())
    actual={p.name for p in data.glob('*.fits')}
    assert actual=={r['filename'] for r in rows},'FITS file set differs from manifest'
    for r in rows:
        p=data/r['filename']
        assert p.stat().st_size==r['bytes'] and sha256(p)==r['sha256'],f'Data mismatch: {p.name}'
    print(f'Verified SHA-256 for {len(rows)} FITS files.')

if __name__=='__main__': main()
