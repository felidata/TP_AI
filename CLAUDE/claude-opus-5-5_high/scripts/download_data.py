"""Download helper for a future public copy of the pp_proc dataset.

No public URL exists yet. When one is published, set it either in
configs/data_source.yaml (key `gdrive_folder_url` or `gdrive_file_ids`) or via
the SKIPPER_DATA_URL environment variable, then run

    python scripts/download_data.py --dest /path/to/pp_proc
    SKIPPER_DATA_DIR=/path/to/pp_proc python scripts/00_manifest.py --verify

Downloading uses the third-party `gdown` package (pip install gdown), which is
only needed for this optional step.
"""
import argparse, os, sys
from pathlib import Path
import yaml

RUN_DIR = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    src = yaml.safe_load(open(RUN_DIR / "configs" / "data_source.yaml"))
    url = os.environ.get("SKIPPER_DATA_URL") or src.get("gdrive_folder_url")
    if not url or "PLACEHOLDER" in str(url):
        sys.exit("No dataset URL configured yet (configs/data_source.yaml holds a placeholder). "
                 "Ask the data owner for the Google Drive link, or point SKIPPER_DATA_DIR at a local copy.")
    try:
        import gdown
    except ImportError:
        sys.exit("pip install gdown  (optional dependency for downloading)")
    dest = Path(a.dest); dest.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(url=url, output=str(dest), quiet=False)
    print(f"Downloaded to {dest}. Now verify: SKIPPER_DATA_DIR={dest} python scripts/00_manifest.py --verify")


if __name__ == "__main__":
    main()
