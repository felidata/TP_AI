"""Log-linear inhomogeneous Poisson point-process model of the event rate density.

rho(p) = exp(b_q + beta . x(p))   [events / pixel-day]

where q is the quadrant (module, HDU), x(p) are the standardized covariates of
pixel p (features.py). The log-likelihood of events {x_i} in exposure maps E is

    log L = sum_i log rho(x_i) - integral rho(p) dE(p)

and the integral is estimated by Monte-Carlo quadrature with dummy points u_j
drawn with probability proportional to exposure, each with weight
w_j = E_image / n_dummy (Berman & Turner 1992; Baddeley & Turner 2000).
The penalized objective is minimized with scipy's L-BFGS-B using the analytic
gradient; the likelihood and gradient code is written for this project.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class PoissonIntensityModel:
    def __init__(self, feature_names: list[str], n_groups: int, l2: float = 1.0):
        self.names = list(feature_names)
        self.G = n_groups
        self.l2 = float(l2)
        self.mu = None
        self.sd = None
        self.b = None
        self.beta = None

    # -- fitting ---------------------------------------------------------
    def fit(self, Xe, ge, Xd, gd, wd):
        """Xe/ge: covariates and group index of events; Xd/gd/wd: dummy points and weights."""
        X_all = np.vstack([Xe, Xd])
        self.mu = X_all.mean(0)
        self.sd = X_all.std(0) + 1e-6
        Ze = (Xe - self.mu) / self.sd
        Zd = (Xd - self.mu) / self.sd
        P = Ze.shape[1]
        ne_g = np.bincount(ge, minlength=self.G).astype(float)

        def f(theta):
            b, beta = theta[: self.G], theta[self.G:]
            ee = b[ge] + Ze @ beta
            ed = np.exp(np.clip(b[gd] + Zd @ beta, -50, 50)) * wd
            ll = ee.sum() - ed.sum()
            obj = -ll + 0.5 * self.l2 * beta @ beta
            gb = -(ne_g - np.bincount(gd, weights=ed, minlength=self.G))
            gbeta = -(Ze.sum(0) - Zd.T @ ed) + self.l2 * beta
            return obj, np.concatenate([gb, gbeta])

        # initialise intercepts at the group mean rates
        wsum = np.bincount(gd, weights=wd, minlength=self.G)
        b0 = np.log((ne_g + 0.5) / np.maximum(wsum, 1e-12))
        theta0 = np.concatenate([b0, np.zeros(P)])
        res = minimize(f, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        self.b, self.beta = res.x[: self.G], res.x[self.G:]
        self.opt = {"success": bool(res.success), "nit": int(res.nit), "fun": float(res.fun)}
        return self

    # -- prediction ------------------------------------------------------
    def log_rate(self, X: np.ndarray, g: int) -> np.ndarray:
        return self.b[g] + ((X - self.mu) / self.sd) @ self.beta

    def log_rate_maps(self, maps: dict, g: int, shape) -> np.ndarray:
        """Same as log_rate, accumulated map by map (no large design matrix)."""
        eta = np.full(shape, self.b[g], dtype=np.float64)
        for k, name in enumerate(self.names):
            c = self.beta[k] / self.sd[k]
            eta += c * maps[name]
            eta -= c * self.mu[k]
        return eta

    def coefficients(self) -> dict:
        return {"intercepts": self.b.tolist(),
                "beta_standardized": dict(zip(self.names, self.beta.tolist())),
                "feature_mean": dict(zip(self.names, self.mu.tolist())),
                "feature_sd": dict(zip(self.names, self.sd.tolist())),
                "l2": self.l2, "optimizer": self.opt}


class HGBIntensityModel:
    """Non-linear alternative: gradient-boosted classifier of events vs dummy points.

    For events with intensity rho(x) per unit exposure and dummy points with
    intensity 1/w, P(event | x) = rho / (rho + 1/w), so
    log rho(x) = logit P(event | x) - log w  (Baddeley et al. 2014, logistic
    point-process likelihood). sklearn's HistGradientBoostingClassifier is used
    as the classifier (library functionality); the quadrant is a categorical
    input.
    """

    def __init__(self, feature_names, n_groups, max_iter=300, learning_rate=0.05, max_leaf_nodes=31, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.names = list(feature_names)
        self.G = n_groups
        self.clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=learning_rate,
                                                  max_leaf_nodes=max_leaf_nodes, categorical_features=[0],
                                                  early_stopping=True, validation_fraction=0.15,
                                                  random_state=seed)

    def fit(self, Xe, ge, Xd, gd, wd):
        X = np.vstack([np.column_stack([ge, Xe]), np.column_stack([gd, Xd])])
        y = np.concatenate([np.ones(len(ge)), np.zeros(len(gd))])
        self.logw = float(np.log(np.mean(wd)))
        self.clf.fit(X, y)
        self.opt = {"n_iter": int(self.clf.n_iter_)}
        return self

    def log_rate_maps(self, maps, g, shape):
        X = np.column_stack([np.full(shape[0] * shape[1], g)] + [np.asarray(maps[k]).ravel() for k in self.names])
        p = np.clip(self.clf.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        return (np.log(p) - np.log1p(-p) - self.logw).reshape(shape)

    def coefficients(self):
        return {"type": "hgb", "features": self.names, "optimizer": self.opt,
                "params": {k: v for k, v in self.clf.get_params().items() if isinstance(v, (int, float, str))}}


class IsolationForestScore:
    """Unsupervised alternative: Isolation Forest (Liu, Ting & Zhou 2008; sklearn)
    fitted on covariates of exposure-sampled dummy pixels only (no events used).
    Pixels are scored on 4x4-block averages of the covariates for speed; the
    anomaly score a = -score_samples in [0, 1] is mapped to the histogram axis
    as eta = -20 + 26 a (monotone; only the ordering matters)."""

    def __init__(self, feature_names, n_groups, n_estimators=200, block=4, seed=0):
        from sklearn.ensemble import IsolationForest
        self.names = list(feature_names)
        self.G = n_groups
        self.block = block
        self.b = np.zeros(n_groups)
        self.iso = IsolationForest(n_estimators=n_estimators, max_samples=4096, random_state=seed)

    def fit(self, Xe, ge, Xd, gd, wd):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(Xd), size=min(len(Xd), 400000), replace=False)
        self.iso.fit(np.column_stack([gd[idx], Xd[idx]]))
        self.opt = {}
        return self

    def log_rate_maps(self, maps, g, shape):
        B = self.block
        ny, nx = shape[0] // B, shape[1] // B
        cols = [np.full(ny * nx, g)]
        for k in self.names:
            m = np.asarray(maps[k], dtype=np.float32)
            cols.append(m.reshape(ny, B, nx, B).mean((1, 3)).ravel())
        a = -self.iso.score_samples(np.column_stack(cols))
        eta = -20.0 + 26.0 * a.reshape(ny, nx)
        return np.repeat(np.repeat(eta, B, axis=0), B, axis=1)

    def coefficients(self):
        return {"type": "iforest", "features": self.names, "block": self.block}
