"""Cluster (event) reconstruction (written for this project; uses scipy.ndimage).

An event is an 8-connected component of pixels with integer charge n >= 1.
Its charge Q is the sum of integer pixel charges; its position is the
charge-weighted centroid, and it is assigned to the pixel containing the
rounded centroid (used to decide whether the event is inside the mask).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

STRUCT8 = np.ones((3, 3), dtype=bool)
STRUCT4 = ndimage.generate_binary_structure(2, 1)


def reconstruct(n: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, dict]:
    struct = STRUCT8 if connectivity == 8 else STRUCT4
    lab, nlab = ndimage.label(n > 0, structure=struct)
    if nlab == 0:
        empty = np.zeros(0)
        return lab, {k: empty for k in ["Q", "npix", "rc", "cc", "qmax", "r0", "r1", "c0", "c1"]}
    idx = lab.ravel()
    sel = idx > 0
    li = idx[sel] - 1
    w = n.ravel()[sel].astype(np.float64)
    rr, cc = np.divmod(np.flatnonzero(sel), n.shape[1])
    Q = np.bincount(li, weights=w, minlength=nlab)
    npix = np.bincount(li, minlength=nlab)
    rc = np.bincount(li, weights=w * rr, minlength=nlab) / Q
    ccen = np.bincount(li, weights=w * cc, minlength=nlab) / Q
    qmax = np.zeros(nlab)
    np.maximum.at(qmax, li, w)
    sl = ndimage.find_objects(lab)
    r0 = np.array([s[0].start for s in sl]); r1 = np.array([s[0].stop for s in sl])
    c0 = np.array([s[1].start for s in sl]); c1 = np.array([s[1].stop for s in sl])
    return lab, {"Q": Q.astype(np.int64), "npix": npix, "rc": rc, "cc": ccen, "qmax": qmax.astype(np.int64),
                 "r0": r0, "r1": r1, "c0": c0, "c1": c1}
