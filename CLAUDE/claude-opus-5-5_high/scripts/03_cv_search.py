"""Stage 3: blocked 4-fold cross-validation on the development acquisitions.

Usage: python scripts/03_cv_search.py [--only name1,name2] [--workers 4]
Writes cache/cv/<name>.npz (per image-quadrant score histograms, out-of-fold)
and results/cv/<name>_coefficients.json. The test block is never read.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, yaml  # noqa: E402
from skbg import config, pipeline, metrics  # noqa: E402


def run_cv(cfg, name, overrides, workers):
    spec = {**pipeline.default_feature_spec(cfg), **overrides}
    K = cfg["split"]["n_cv_folds"]
    groups = [f"f{k}" for k in range(K)]
    acc = {"img": [], "q": [], "E": [], "N": [], "Er": [], "Nr": [], "fold": []}
    spec_hist = 0
    coefs = {}
    t0 = time.time()
    for k in range(K):
        train = [g for g in groups if g != f"f{k}"]
        mdl = pipeline.fit_model(cfg, spec, train, workers=workers)
        parts = pipeline.evaluate(cfg, spec, mdl, train, [f"f{k}"], workers=workers)
        imgs, qs, E, N = metrics.stack_items(parts)
        _, _, Er, Nr = metrics.stack_items(parts, "Erand", "Nrand")
        for key, v in zip(("img", "q", "E", "N", "Er", "Nr"), (imgs, qs, E, N, Er, Nr)):
            acc[key].append(v)
        acc["fold"].append(np.full(len(imgs), k))
        spec_hist = spec_hist + sum(o["spec"] for o in parts.values())
        coefs[f"fold{k}"] = {**mdl.coefficients(), "n_train_events": mdl.n_events}
        print(f"  [{name}] fold {k} done ({time.time()-t0:.0f}s)", flush=True)
    out = {k: np.concatenate(v) for k, v in acc.items()}
    cd = config.cache_dir(cfg) / "cv"; cd.mkdir(exist_ok=True)
    np.savez_compressed(cd / f"{name}.npz", spec=spec_hist, **out)
    rd = config.results_dir() / "cv"; rd.mkdir(exist_ok=True)
    (rd / f"{name}_coefficients.json").write_text(json.dumps({"spec": spec, **coefs}, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--search", default=str(config.RUN_DIR / "configs" / "cv_search.yaml"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = config.load_config(a.config)
    search = yaml.safe_load(open(a.search))["configs"]
    names = a.only.split(",") if a.only else list(search)
    for n in names:
        out = run_cv(cfg, n, search[n] or {}, a.workers)
        c = metrics.curve(out["E"].sum(0), out["N"].sum(0))
        grid = np.round(np.arange(0.80, 0.991, 0.01), 2)
        S = np.mean([np.log(c["r_B"][metrics.index_at_fexp(c, f)]) for f in grid])
        print(f"{n}: S = {S:.4f}", flush=True)


if __name__ == "__main__":
    main()
