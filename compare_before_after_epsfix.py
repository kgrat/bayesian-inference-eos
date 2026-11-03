"""
Old vs new (energy-density fix) comparison of the emcee posterior.

Old physics (results_before_epsfix/old_engines/run_mcmc.py) on the old chain,
new physics (tov_model.py) on the new chain.
Usage:
  python compare_before_after_epsfix.py [new_h5] [--ndraw N]
"""
import argparse
import importlib.util
import multiprocessing as mp
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
OLD_DIR = os.path.join(HERE, "results_before_epsfix")

_OLD = None
def old_obs(params):
    global _OLD
    if _OLD is None:
        path = os.path.join(OLD_DIR, "old_engines", "run_mcmc.py")
        spec = importlib.util.spec_from_file_location("old_run_mcmc", path)
        _OLD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_OLD)
    return _OLD.generate_observables(params)

def new_obs(params):
    from tov_model import generate_observables
    return generate_observables(params)

def draws(h5_path, n, seed=0):
    with h5py.File(h5_path, "r") as f:
        flat = f["flat_samples"][:]
    idx = np.random.default_rng(seed).choice(len(flat), size=min(n, len(flat)), replace=False)
    return flat[idx]

def summarize(x):
    x = np.asarray(x)
    x = x[np.isfinite(x) & (x > 0)]
    lo, med, hi = np.percentile(x, [5, 50, 95])
    return f"{med:.2f} [{lo:.2f}, {hi:.2f}]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new_h5", nargs="?", default=os.path.join(HERE, "mcmc_results.h5"))
    ap.add_argument("--ndraw", type=int, default=500)
    args = ap.parse_args()

    runs = [("old", os.path.join(OLD_DIR, "mcmc_results.h5"), old_obs)]
    if os.path.exists(args.new_h5) and os.path.realpath(args.new_h5) != os.path.realpath(runs[0][1]):
        runs.append(("new", args.new_h5, new_obs))

    print(f"\n{'run':5s}{'M_max [Msun] med [90%]':>26s}{'R_1.4 [km] med [90%]':>26s}{'R_2.08 [km] med [90%]':>26s}")
    for tag, path, fn in runs:
        p = draws(path, args.ndraw)
        with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
            obs = np.array(pool.map(fn, [list(x) for x in p]))
        print(f"{tag:5s}{summarize(obs[:, 0]):>26s}{summarize(obs[:, 1]):>26s}{summarize(obs[:, 2]):>26s}")

if __name__ == "__main__":
    main()
