"""Stage 0: write data/manifest.csv (file names, sizes, SHA-256, key header metadata).

Usage: SKIPPER_DATA_DIR=/path/to/pp_proc python scripts/00_manifest.py [--no-hash]
       python scripts/00_manifest.py --verify   # check local files against the manifest
"""
import argparse, hashlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
from astropy.io import fits  # noqa: E402
from skbg import config, io  # noqa: E402

KEYS = ["DATESTART", "DATEEND", "RUNID", "NSAMP", "NROW", "NCOL", "CCDNROW", "CCDNCOL", "EXPOSURE", "CLEARNCOL"]


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    cfg = config.load_config()
    dd = config.data_dir(cfg)
    out = config.RUN_DIR / "data" / "manifest.csv"
    recs = io.discover(dd, cfg["data"]["file_glob"])
    if a.verify:
        man = pd.read_csv(out)
        local = {r["file"]: r["path"] for r in recs}
        bad = [f for f in man.file if f not in local or sha256(local[f]) != man.set_index("file").loc[f, "sha256"]]
        print(f"{len(man)} files in manifest, {len(local)} local, {len(bad)} missing/mismatched")
        for f in bad[:20]:
            print("  ", f)
        sys.exit(1 if bad else 0)
    rows = []
    for r in recs:
        h = fits.getheader(r["path"], 0)
        row = {"file": r["file"], "module": r["module"], "img": r["img"], "bytes": r["path"].stat().st_size,
               "n_hdu": len(fits.open(r["path"], memmap=True)),
               "chid_hdu": ",".join(str(fits.getheader(r["path"], i).get("CHID")) for i in range(4))}
        row.update({k: h.get(k) for k in KEYS})
        row["sha256"] = None if a.no_hash else sha256(r["path"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} files)")


if __name__ == "__main__":
    main()
