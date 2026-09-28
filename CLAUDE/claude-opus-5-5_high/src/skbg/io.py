"""Dataset discovery and FITS reading."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

NAME_RE = re.compile(r"mod(S\d+)_.*_img(\d+)\.fits$")


def discover(data_dir: Path, file_glob: str) -> list[dict]:
    """Return one record per FITS file: path, module, image number."""
    out = []
    for p in sorted(data_dir.glob(file_glob)):
        m = NAME_RE.search(p.name)
        if not m:
            continue
        out.append({"path": p, "file": p.name, "module": m.group(1), "img": int(m.group(2))})
    out.sort(key=lambda r: (r["img"], r["module"]))
    return out


def read_timing(header) -> dict:
    t0 = datetime.fromisoformat(str(header["DATESTART"]).strip())
    t1 = datetime.fromisoformat(str(header["DATEEND"]).strip())
    return {
        "datestart": t0.isoformat(),
        "dateend": t1.isoformat(),
        "readout_s": (t1 - t0).total_seconds(),
        "exposure_s": float(header["EXPOSURE"]),
        "nrow": int(header["NROW"]),
        "ncol": int(header["NCOL"]),
        "nsamp": int(header["NSAMP"]),
    }


def read_hdus(path: Path) -> tuple[list[np.ndarray], list]:
    with fits.open(path, memmap=False) as hdul:
        data = [np.asarray(h.data, dtype=np.float64) for h in hdul]
        headers = [h.header.copy() for h in hdul]
    return data, headers
