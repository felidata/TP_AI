#!/usr/bin/env python3
"""Obtain a user-supplied public/shared Google Drive file or folder.

Optional dependency: pip install gdown==5.2.0. No URL is known or embedded.
Archives are downloaded, never silently extracted; follow README for placement.
"""
import argparse
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',help='Actual Google Drive URL supplied by the dataset owner')
p.add_argument('--file-id',help='Actual Google Drive file ID supplied by the owner')
p.add_argument('--output',type=Path,default=Path(__file__).parent/'downloads')
a=p.parse_args()
if bool(a.url)==bool(a.file_id): p.error('Supply exactly one of --url or --file-id. DATASET_URL=NOT_YET_PROVIDED.')
try: import gdown
except ImportError: raise SystemExit('Install optional downloader: python -m pip install gdown==5.2.0')
a.output.mkdir(parents=True,exist_ok=True)
if a.url and '/folders/' in a.url:
    result=gdown.download_folder(url=a.url,output=str(a.output),quiet=False)
else:
    result=gdown.download(url=a.url,id=a.file_id,output=str(a.output)+'/',quiet=False,fuzzy=True)
if not result: raise SystemExit('Download failed. Check sharing access with the data owner.')
print('Downloaded to',a.output,'; set SKIPPER_DATA_DIR to the directory containing pp_proc*.fits.')
