"""Training-set construction, model fitting and streaming evaluation.

Leakage rules implemented here
* Splits are by complete acquisition (image number); both modules and all HDUs
  of an acquisition share a group.
* Static covariates of an image never include that image: for training images
  they are cross-fitted (built from the other training groups); for evaluation
  images they are built from all training groups.
* Dynamic covariates use only non-target pixels of the same image
  (clusters with Q >= hee_min_e, and Q == 1 clusters).
* Target events (4 <= Q <= 7) of evaluation images are used only to count
  events after the mask has been fixed.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from . import cache, config, dataset, exposure, features, stacks
from .model import HGBIntensityModel, IsolationForestScore, PoissonIntensityModel

ETA_STEP = 0.01
ETA_EDGES = -20.0 + ETA_STEP * np.arange(2601)   # -20 .. +6 in log(rate / pixel-day)
NQ = 64  # spectrum bins: Q = 0..62, 63 = overflow


def default_feature_spec(cfg: dict) -> dict:
    m = cfg.get("model", {})
    return {
        "static_source": m.get("static_source", "roi"),
        "static_windows": m.get("static_windows", list(features.STATIC_WINDOWS)),
        "pseudo_count": m.get("pseudo_count", 1.0),
        "use_static": m.get("use_static", True),
        "use_dynamic": m.get("use_dynamic", True),
        "use_onee": m.get("use_onee", True),
        "hee_min_e": m.get("hee_min_e", cfg["events"]["hee_min_e"]),
        "target": m.get("target", "roi"),
        "n_dummy": m.get("n_dummy", 3000),
        "l2": m.get("l2", 1.0),
        "model_type": m.get("model_type", "glm"),
    }


def _kmap(cfg):
    return exposure.exposure_map_days({"exposure_s": 0.0, "readout_s": 86400.0, "nrow": 515, "ncol": 3200},
                                      tuple(cfg["geometry"]["active_rows"]), tuple(cfg["geometry"]["active_cols"]))


def _static(cfg, spec, groups, m, h):
    if not spec["use_static"] or not groups:
        return {}
    st = stacks.load(cfg, groups, m, h)
    win = {k: features.STATIC_WINDOWS[k] for k in spec["static_windows"]}
    out = {}
    for src in spec["static_source"].split("+"):
        cov = features.static_covariates(st[src], st["E"], spec["pseudo_count"], win)
        out.update({f"st_{src}_{k}": v for k, v in cov.items()})
    return out


def _dynamic(z, h, spec):
    arr = features.image_arrays(z, h)
    out = {}
    if spec["use_dynamic"] or spec["use_onee"]:
        one = arr["Q"] == 1
        dyn = features.dynamic_covariates(
            features.hee_source(arr["n"], arr["lab"], arr["Q"], spec["hee_min_e"]),
            arr["rc"][one], arr["cc"][one])
        for k, v in dyn.items():
            if k.startswith("hee") and spec["use_dynamic"]:
                out[k] = v
            if k.startswith("one") and spec["use_onee"]:
                out[k] = v
    return arr, out


def feature_names(spec: dict) -> list[str]:
    names = []
    if spec["use_static"]:
        for src in spec["static_source"].split("+"):
            names += [f"st_{src}_{k}" for k in spec["static_windows"]]
    if spec["use_dynamic"]:
        names += [f"hee_{d}_L{L:g}" for L in (3.0, 15.0) for d in ("right", "left", "up", "down")]
        names += [f"hee_iso_L{L:g}" for L in (3.0, 15.0)]
    if spec["use_onee"]:
        names += ["one_row", "one_col"]
    return names


def _target_mask(Q, target):
    lo, hi = stacks.SOURCES[target]
    return (Q >= lo) & (Q <= hi)


# ---------------------------------------------------------------------------
# training data (one worker per quadrant)
# ---------------------------------------------------------------------------
def _collect_quadrant(args):
    cfg, spec, train_groups, qi, recs, folds, seed = args
    m, h = stacks.QUADRANTS[qi]
    names = feature_names(spec)
    cd = config.cache_dir(cfg)
    kmap = _kmap(cfg)
    static_cf = {g: _static(cfg, spec, [x for x in train_groups if x != g], m, h) for g in train_groups}
    Xe, Xd, wd = [], [], []
    for r in recs:
        g = stacks.group_of(r, cfg, folds)
        if r["module"] != m or g not in train_groups:
            continue
        z = cache.load_image(cd, r)
        t = cache.load_meta(cd, r)["timing"]
        E = kmap * (t["readout_s"] / 86400.0) + t["exposure_s"] / 86400.0
        arr, dyn = _dynamic(z, h, spec)
        maps = {**static_cf[g], **dyn}
        sel = _target_mask(arr["Q"], spec["target"])
        er, ec = arr["rc"][sel], arr["cc"][sel]
        rng = np.random.default_rng([seed, r["img"], qi])
        cdf = np.cumsum(E.ravel())
        idx = np.searchsorted(cdf, rng.random(spec["n_dummy"]) * cdf[-1], side="right")
        dr, dc = np.divmod(idx, features.SHAPE[1])
        Xe.append(np.column_stack([maps[k][er, ec] for k in names]) if names else np.zeros((len(er), 0)))
        Xd.append(np.column_stack([maps[k][dr, dc] for k in names]) if names else np.zeros((len(dr), 0)))
        wd.append(np.full(len(dr), E.sum() / spec["n_dummy"]))
    return qi, np.vstack(Xe), np.vstack(Xd), np.concatenate(wd)


def fit_model(cfg, spec, train_groups, workers=4, seed=None):
    seed = cfg["seed"] if seed is None else seed
    recs = [r for r in dataset.records(cfg) if r["usable"]]
    folds = dataset.dev_folds(cfg, [r["img"] for r in recs])
    jobs = [(cfg, spec, list(train_groups), qi, recs, folds, seed) for qi in range(len(stacks.QUADRANTS))]
    with mp.get_context("spawn").Pool(workers) as pool:  # fork after OpenMP init can deadlock
        parts = pool.map(_collect_quadrant, jobs)
    Xe = np.vstack([p[1] for p in parts]); ge = np.concatenate([np.full(len(p[1]), p[0]) for p in parts])
    Xd = np.vstack([p[2] for p in parts]); gd = np.concatenate([np.full(len(p[2]), p[0]) for p in parts])
    wd = np.concatenate([p[3] for p in parts])
    mt = spec.get("model_type", "glm")
    if mt == "hgb":
        mdl = HGBIntensityModel(feature_names(spec), len(stacks.QUADRANTS), seed=seed)
    elif mt == "iforest":
        mdl = IsolationForestScore(feature_names(spec), len(stacks.QUADRANTS), seed=seed)
    else:
        mdl = PoissonIntensityModel(feature_names(spec), len(stacks.QUADRANTS), spec["l2"])
    if Xe.shape[1] == 0:  # intercept-only model
        Xe = np.zeros((len(ge), 1)); Xd = np.zeros((len(gd), 1)); mdl.names = ["const"]
    mdl.fit(Xe, ge, Xd, gd, wd)
    mdl.n_events = int(len(ge))
    return mdl


# ---------------------------------------------------------------------------
# evaluation (one worker per quadrant)
# ---------------------------------------------------------------------------
def _eval_quadrant(args):
    cfg, spec, mdl, train_groups, eval_groups, qi, recs, folds, seed, keep_maps = args
    m, h = stacks.QUADRANTS[qi]
    cd = config.cache_dir(cfg)
    kmap = _kmap(cfg)
    st = _static(cfg, spec, list(train_groups), m, h)
    names = mdl.names
    nb = len(ETA_EDGES) - 1
    out = {"img": [], "E": [], "N": [], "Erand": [], "Nrand": [], "spec": np.zeros((NQ, nb)), "maps": {}}
    for r in recs:
        g = stacks.group_of(r, cfg, folds)
        if r["module"] != m or g not in eval_groups:
            continue
        z = cache.load_image(cd, r)
        t = cache.load_meta(cd, r)["timing"]
        E = kmap * (t["readout_s"] / 86400.0) + t["exposure_s"] / 86400.0
        arr, dyn = _dynamic(z, h, spec)
        maps = {**st, **dyn}
        if names == ["const"]:
            eta = np.full(features.SHAPE, float(mdl.b[qi]))
        else:
            eta = mdl.log_rate_maps(maps, qi, features.SHAPE)
        bi = np.clip(np.floor((eta - ETA_EDGES[0]) / ETA_STEP).astype(np.int64), 0, nb - 1)
        out["E"].append(np.bincount(bi.ravel(), weights=E.ravel(), minlength=nb))
        sel = _target_mask(arr["Q"], "roi")
        out["N"].append(np.bincount(bi[arr["rc"][sel], arr["cc"][sel]], minlength=nb))
        qb = np.clip(arr["Q"], 0, NQ - 1)
        np.add.at(out["spec"], (qb, bi[arr["rc"], arr["cc"]]), 1)
        # random-score control (pure area removal)
        rng = np.random.default_rng([seed, r["img"], qi, 7])
        u = rng.random(features.SHAPE)
        ri = np.minimum((u * 1000).astype(np.int64), 999)
        out["Erand"].append(np.bincount(ri.ravel(), weights=E.ravel(), minlength=1000))
        out["Nrand"].append(np.bincount(ri[arr["rc"][sel], arr["cc"][sel]], minlength=1000))
        thr = spec.get("control_threshold")
        if thr is not None:
            # shifted-mask control: same mask, rolled along columns by a random offset
            keep = eta < thr
            rr, cc_ = arr["rc"][sel], arr["cc"][sel]
            dx = int(rng.integers(300, features.SHAPE[1] - 300))
            keep_s = np.roll(keep, dx, axis=1)
            if "dens" not in out:
                out["dens"] = {k: np.zeros(features.SHAPE, np.float32) for k in ("E_all", "E_kept", "N_all", "N_kept")}
            out["dens"]["E_all"] += E; out["dens"]["E_kept"] += E * keep
            np.add.at(out["dens"]["N_all"], (rr, cc_), 1)
            np.add.at(out["dens"]["N_kept"], (rr[keep[rr, cc_]], cc_[keep[rr, cc_]]), 1)
            out.setdefault("shift", []).append([E[keep].sum(), keep[rr, cc_].sum(), E[keep_s].sum(),
                                                keep_s[rr, cc_].sum(), E.sum(), sel.sum()])
        out["img"].append(r["img"])
        if keep_maps and r["img"] in keep_maps:
            out["maps"][r["img"]] = {"eta": eta.astype(np.float32), "n": arr["n"].astype(np.int32),
                                     "roi_rc": arr["rc"][sel], "roi_cc": arr["cc"][sel]}
    for k in ("E", "N", "Erand", "Nrand", "shift"):
        if k in out:
            out[k] = np.array(out[k])
    return qi, out


def evaluate(cfg, spec, mdl, train_groups, eval_groups, workers=4, seed=None, keep_maps=None):
    """Returns dict with per-(image, quadrant) histograms over the score (log predicted rate)."""
    seed = cfg["seed"] if seed is None else seed
    recs = [r for r in dataset.records(cfg) if r["usable"]]
    folds = dataset.dev_folds(cfg, [r["img"] for r in recs])
    jobs = [(cfg, spec, mdl, list(train_groups), list(eval_groups), qi, recs, folds, seed, keep_maps)
            for qi in range(len(stacks.QUADRANTS))]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = dict(pool.map(_eval_quadrant, jobs))
    return parts
