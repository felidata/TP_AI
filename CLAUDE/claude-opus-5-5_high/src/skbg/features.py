"""Per-pixel covariates for the background-intensity model (written for this project).

Two families, never using the target (4-7 e-) events of the image being scored:

Static (spatially persistent) covariates
    Built from *other* acquisitions only. For a count map N (events per pixel,
    stacked over a training set of images) and the matching stacked exposure map
    E, a window w gives a shrunk relative-rate estimate

        s_w(p) = log[(N_w(p) + k) / (rho_q * E_w(p) + k)],

    with N_w, E_w box sums over w, rho_q the quadrant-mean rate and k a
    pseudo-count (Gamma-Poisson shrinkage toward the quadrant mean). Windows are
    points, squares at several scales and vertical/horizontal segments, so the
    model can represent single hot pixels, hot columns and extended hot zones.

Dynamic (same-image) covariates
    * high-energy-deposit (HEE) proximity: exponential kernels applied with
      first-order recursive filters (numba loops, written here) to the map of
      log(1 + n) for pixels of clusters with Q >= hee_min_e, separately along
      +x (serial readout direction), -x, +y (vertical transfer direction), -y,
      and a separable isotropic kernel;
    * single-electron activity: counts of Q == 1 clusters in the same row and
      the same column of the same image, relative to the image mean (serial
      register events and transient hot columns); these pixels are disjoint
      from any 4-7 e- cluster by construction.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from numba import njit

SHAPE = (512, 3072)

# ---------------------------------------------------------------------------
# static covariates
# ---------------------------------------------------------------------------
STATIC_WINDOWS = {
    "p1": (1, 1),
    "sq5": (5, 5),
    "sq17": (17, 17),
    "sq65": (65, 65),
    "sq257": (257, 257),
    "col65": (65, 1),
    "col513": (1025, 1),     # whole column (window taller than the image)
    "row65": (1, 65),
}


def _boxsum(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if size == (1, 1):
        return a.astype(np.float64)
    return ndimage.uniform_filter(a.astype(np.float64), size=size, mode="constant") * (size[0] * size[1])


def static_covariates(N: np.ndarray, E: np.ndarray, k: float = 1.0,
                      windows: dict | None = None) -> dict[str, np.ndarray]:
    windows = windows or STATIC_WINDOWS
    rho = N.sum() / E.sum()
    out = {}
    for name, w in windows.items():
        Nw, Ew = _boxsum(N, w), _boxsum(E, w)
        out[name] = np.log((Nw + k) / (rho * Ew + k)).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# dynamic covariates
# ---------------------------------------------------------------------------
@njit(cache=True)
def _trail_x(a, lam, reverse):
    ny, nx = a.shape
    y = np.zeros_like(a)
    for i in range(ny):
        acc = 0.0
        if not reverse:
            for j in range(nx):
                y[i, j] = acc
                acc = lam * (acc + a[i, j])
        else:
            for j in range(nx - 1, -1, -1):
                y[i, j] = acc
                acc = lam * (acc + a[i, j])
    return y


@njit(cache=True)
def _trail_y(a, lam, reverse):
    ny, nx = a.shape
    y = np.zeros_like(a)
    acc = np.zeros(nx)
    rng = range(ny - 1, -1, -1) if reverse else range(ny)
    for i in rng:
        for j in range(nx):
            y[i, j] = acc[j]
            acc[j] = lam * (acc[j] + a[i, j])
    return y


def _exp_trail(a: np.ndarray, L: float, axis: int, reverse: bool) -> np.ndarray:
    """y[i] = sum_{j<i} a[j] * exp(-(i-j)/L): strictly one-sided trail along axis
    (reverse=True: sum over j>i)."""
    lam = float(np.exp(-1.0 / L))
    a = np.ascontiguousarray(a, dtype=np.float64)
    return _trail_x(a, lam, reverse) if axis == 1 else _trail_y(a, lam, reverse)


def _exp_sym(a: np.ndarray, L: float, axis: int) -> np.ndarray:
    return a + _exp_trail(a, L, axis, False) + _exp_trail(a, L, axis, True)


def hee_source(n: np.ndarray, lab: np.ndarray, Q: np.ndarray, hee_min_e: int) -> np.ndarray:
    hee_labels = np.flatnonzero(Q >= hee_min_e) + 1
    src = np.zeros(n.shape, dtype=np.float64)
    if hee_labels.size:
        foot = np.isin(lab, hee_labels)
        src[foot] = np.log1p(n[foot])
    return src


def dynamic_covariates(src: np.ndarray, one_r: np.ndarray, one_c: np.ndarray,
                       scales: tuple[float, ...] = (3.0, 15.0), iso_scales: tuple[float, ...] = (3.0, 15.0),
                       band: int = 1) -> dict[str, np.ndarray]:
    out = {}
    # narrow band (|dy|<=band or |dx|<=band) for the directional trails
    src_bx = ndimage.uniform_filter1d(src, 2 * band + 1, axis=0, mode="constant") * (2 * band + 1)
    src_by = ndimage.uniform_filter1d(src, 2 * band + 1, axis=1, mode="constant") * (2 * band + 1)
    for L in scales:
        out[f"hee_right_L{L:g}"] = _exp_trail(src_bx, L, axis=1, reverse=False)
        out[f"hee_left_L{L:g}"] = _exp_trail(src_bx, L, axis=1, reverse=True)
        out[f"hee_up_L{L:g}"] = _exp_trail(src_by, L, axis=0, reverse=False)
        out[f"hee_down_L{L:g}"] = _exp_trail(src_by, L, axis=0, reverse=True)
    for L in iso_scales:
        out[f"hee_iso_L{L:g}"] = _exp_sym(_exp_sym(src, L, 0), L, 1)
    for k in list(out):
        out[k] = np.log1p(out[k]).astype(np.float32)
    # same-image single-electron row / column activity
    rows = np.bincount(one_r, minlength=SHAPE[0]).astype(np.float64)
    cols = np.bincount(one_c, minlength=SHAPE[1]).astype(np.float64)
    # expected per-row counts grow ~linearly with exposure; use a robust linear fit in row index
    ridx = np.arange(SHAPE[0])
    A = np.vstack([np.ones_like(ridx), ridx]).T
    coef, *_ = np.linalg.lstsq(A, rows, rcond=None)
    exp_rows = np.clip(A @ coef, 1e-3, None)
    med_col = max(np.median(cols), 1e-3)
    out["one_row"] = np.broadcast_to(np.log((rows + 1) / (exp_rows + 1))[:, None], SHAPE).astype(np.float32)
    out["one_col"] = np.broadcast_to(np.log((cols + 1) / (med_col + 1))[None, :], SHAPE).astype(np.float32)
    return out


def image_arrays(z: dict, h: int) -> dict:
    """Rebuild the integer-charge image and labels of one HDU from the cache."""
    n = np.zeros(SHAPE, dtype=np.int32)
    n[z[f"pix{h}_r"], z[f"pix{h}_c"]] = z[f"pix{h}_n"]
    lab, nlab = ndimage.label(n > 0, structure=np.ones((3, 3), bool))
    Q = z[f"cl{h}_Q"]
    assert nlab == len(Q), "cluster catalogue / label mismatch"
    return {"n": n, "lab": lab, "Q": Q,
            "rc": np.rint(z[f"cl{h}_rc"]).astype(int), "cc": np.rint(z[f"cl{h}_cc"]).astype(int)}
