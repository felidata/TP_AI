"""Operating points, uncertainties and Pareto frontier (written for this project).

For a score s(p) (log predicted rate density) and threshold t, a pixel is kept
if s(p) < t. With E the exposure (pixel-day) and N the number of 4-7 e-
clusters whose centroid pixel is kept:

    f_exp = E_after / E_before
    r_B   = (N_after / E_after) / (N_before / E_before)
    F_equal = f_exp / r_B

Uncertainties: bootstrap over complete acquisitions (image numbers) of the
evaluated set, so image-to-image variability and correlations between the
quadrants of one acquisition are propagated.
"""
from __future__ import annotations

import numpy as np


def stack_items(parts: dict, key_E="E", key_N="N"):
    """Concatenate per-quadrant outputs -> (img ids, quadrant ids, E[n, nb], N[n, nb])."""
    imgs, qs, Es, Ns = [], [], [], []
    for qi, o in sorted(parts.items()):
        if len(o["img"]) == 0:
            continue
        imgs.append(np.array(o["img"])); qs.append(np.full(len(o["img"]), qi))
        Es.append(o[key_E]); Ns.append(o[key_N])
    return np.concatenate(imgs), np.concatenate(qs), np.vstack(Es), np.vstack(Ns)


def curve(E: np.ndarray, N: np.ndarray) -> dict:
    """E, N: 1-D histograms over ascending score bins (pooled). Row k keeps bins <= k."""
    cE, cN = np.cumsum(E), np.cumsum(N)
    Et, Nt = cE[-1], cN[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f = cE / Et
        rB = (cN / cE) / (Nt / Et)
        F = f / rB
    return {"f_exp": f, "r_B": rB, "F_equal": F, "E_after": cE, "N_after": cN, "E_before": Et, "N_before": Nt}


def index_at_fexp(c: dict, target_f: float) -> int:
    return int(np.searchsorted(c["f_exp"], target_f))


def bootstrap(imgs, E, N, thresholds_idx, n_boot=1000, seed=0):
    """Acquisition-level bootstrap of (f_exp, r_B, F) at fixed threshold indices."""
    rng = np.random.default_rng(seed)
    u = np.unique(imgs)
    pos = {i: np.flatnonzero(imgs == i) for i in u}
    cE = np.cumsum(E, axis=1)[:, thresholds_idx]
    cN = np.cumsum(N, axis=1)[:, thresholds_idx]
    tE, tN = E.sum(1), N.sum(1)
    # per-acquisition sums
    A_cE = np.array([cE[pos[i]].sum(0) for i in u]); A_cN = np.array([cN[pos[i]].sum(0) for i in u])
    A_tE = np.array([tE[pos[i]].sum() for i in u]); A_tN = np.array([tN[pos[i]].sum() for i in u])
    out = np.zeros((n_boot, 3, len(thresholds_idx)))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)).astype(float)
        e, n, te, tn = w @ A_cE, w @ A_cN, w @ A_tE, w @ A_tN
        f = e / te
        rB = (n / e) / (tn / te)
        out[b] = [f, rB, f / rB]
    return out


def pareto_mask(f: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Non-dominated points: maximize f, minimize r."""
    order = np.argsort(-f, kind="stable")
    keep = np.zeros(len(f), bool)
    best = np.inf
    for i in order:
        if np.isfinite(r[i]) and r[i] < best:
            keep[i] = True
            best = r[i]
    return keep


def poisson_sig_gain(f_exp, r_B):
    """Background-dominated counting significance gain Z_after/Z_before = sqrt(f_exp/r_B)."""
    return np.sqrt(f_exp / r_B)


def asimov_Z(s, b):
    """Median discovery significance for known background (Cowan et al. 2011, Eq. 97)."""
    s = np.asarray(s, float); b = np.asarray(b, float)
    return np.sqrt(2 * ((s + b) * np.log1p(s / b) - s))
