"""Charge calibration of one quadrant image (written for this project).

Steps
1. Row pedestal: subtract the median of the reverse-overscan columns of each
   row (the same *region choice* as MOSKITA's CalibrationConfig, 3100:3200).
2. Fit the active-region pixel histogram with n_peaks Gaussians centred at
   mu0 + n*g with a common width sigma and free amplitudes A_n
   (scipy.optimize.curve_fit, Poisson-weighted least squares).
3. q = (x - mu0)/g in e-; integer charge n = floor(q + rounding_offset), n>=0.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def row_pedestal_subtract(img: np.ndarray, baseline_cols: tuple[int, int]) -> np.ndarray:
    base = np.median(img[:, baseline_cols[0]:baseline_cols[1]], axis=1)
    return img - base[:, None]


def _model(x, mu0, g, sigma, *amps):
    out = np.zeros_like(x)
    for n, a in enumerate(amps):
        out += a * np.exp(-0.5 * ((x - mu0 - n * g) / sigma) ** 2)
    return out


def fit_peaks(values: np.ndarray, cfg: dict) -> dict:
    c = cfg["calibration"]
    edges = np.arange(c["hist_min_adu"], c["hist_max_adu"] + c["hist_bin_adu"], c["hist_bin_adu"])
    h, e = np.histogram(values, bins=edges)
    x = 0.5 * (e[1:] + e[:-1])
    npk = int(c["n_peaks"])
    g0, s0 = c["gain_guess_adu"], c["sigma_guess_adu"]
    a0 = float(h.max())
    p0 = [0.0, g0, s0] + [a0] + [max(a0 * 10.0 ** (-2 - n), 1.0) for n in range(1, npk)]
    lo = [-60.0, 120.0, 10.0] + [0.0] * npk
    hi = [60.0, 320.0, 90.0] + [np.inf] * npk
    w = np.sqrt(np.maximum(h, 1.0))
    popt, pcov = curve_fit(lambda xx, *p: _model(xx, *p), x, h, p0=p0, bounds=(lo, hi),
                           sigma=w, maxfev=20000)
    resid = (h - _model(x, *popt)) / w
    sel = (x > popt[0] - 3 * popt[2]) & (x < popt[0] + (npk - 0.5) * popt[1])
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    return {
        "mu0_adu": float(popt[0]), "gain_adu": float(popt[1]), "sigma_adu": float(popt[2]),
        "gain_err_adu": float(perr[1]),
        "noise_e": float(popt[2] / popt[1]),
        "amps": [float(a) for a in popt[3:]],
        "see_a1a0": float(popt[4] / popt[3]) if popt[3] > 0 else np.nan,
        "chi2_ndf": float(np.sum(resid[sel] ** 2) / max(sel.sum() - len(popt), 1)),
    }


def calibrate_quadrant(img: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (q_active [e-, float32], n_active [int32], fit dict) for the active region."""
    g = cfg["geometry"]
    x = row_pedestal_subtract(img, tuple(g["baseline_cols"]))
    r0, r1 = g["active_rows"]
    c0, c1 = g["active_cols"]
    act = x[r0:r1, c0:c1]
    fit = fit_peaks(act.ravel(), cfg)
    q = ((act - fit["mu0_adu"]) / fit["gain_adu"]).astype(np.float32)
    n = np.floor(q + cfg["calibration"]["rounding_offset"]).astype(np.int64)
    n[n < 0] = 0
    return q, n.astype(np.int32), fit
