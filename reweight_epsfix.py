"""
Importance reweighting of the pre-fix emcee chain to the energy-density-corrected likelihood.

The prior is unchanged by the fix, so a draw from the old posterior gets weight
  w = L_new / L_old = exp(lnL_new - lnL_old).
The result is only trustworthy if the effective sample size ESS = (sum w)^2 / sum w^2
is a reasonable fraction of the draws; otherwise re-run run_mcmc.py.

Usage:
  python reweight_epsfix.py [--ndraw N] [--out reweighted_epsfix.h5]
"""
import argparse
import importlib.util
import multiprocessing as mp
import os
import h5py
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
OLD_H5 = os.path.join(HERE, "results_before_epsfix", "mcmc_results.h5")

_OLD = _NEW = None

def _load():
    global _OLD, _NEW
    if _OLD is None:
        spec = importlib.util.spec_from_file_location(
            "old_run_mcmc", os.path.join(HERE, "results_before_epsfix", "old_engines", "run_mcmc.py"))
        _OLD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_OLD)
        import run_mcmc as new
        _NEW = new

def evaluate(params):
    _load()
    params = list(params)
    lnl_old = _OLD.log_likelihood(params)
    lnl_new = _NEW.log_likelihood(params)
    m_max, r14, r208 = _NEW.generate_observables(params)
    return lnl_old, lnl_new, m_max, r14, r208

def weighted_quantiles(x, w, qs):
    ok = np.isfinite(x) & (x > 0) & (w > 0)
    x, w = x[ok], w[ok]
    i = np.argsort(x)
    cw = np.cumsum(w[i]) / np.sum(w)
    return np.interp(qs, cw, x[i])

def fmt(x, w):
    lo, med, hi = weighted_quantiles(x, w, [0.05, 0.5, 0.95])
    return f"{med:.2f} [{lo:.2f}, {hi:.2f}]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndraw", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(HERE, "reweighted_epsfix.h5"))
    args = ap.parse_args()

    with h5py.File(OLD_H5, "r") as f:
        flat = f["flat_samples"][:]
    draws = flat[np.random.default_rng(0).choice(len(flat), size=args.ndraw, replace=False)]

    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        res = np.array(pool.map(evaluate, draws), dtype=float)
    lnl_old, lnl_new, m_max, r14, r208 = res.T

    dl = np.where(np.isfinite(lnl_new), lnl_new - lnl_old, -np.inf)
    w = np.exp(dl - np.max(dl[np.isfinite(dl)]))
    ess = np.sum(w)**2 / np.sum(w**2)
    top = np.sort(w)[::-1] / np.sum(w)

    print(f"\ndraws: {len(w)}; rejected by new likelihood: {np.sum(~np.isfinite(lnl_new))}")
    print(f"ESS = {ess:.0f} ({100*ess/len(w):.1f}% of draws); largest single weight = {100*top[0]:.1f}%, "
          f"top 10 weights = {100*top[:10].sum():.1f}%")
    uniform = np.isfinite(lnl_old).astype(float)
    print(f"\n{'':32s}{'M_max [Msun]':>22s}{'R_1.4 [km]':>22s}{'R_2.08 [km]':>22s}")
    print(f"{'old chain, corrected physics':32s}{fmt(m_max, uniform):>22s}{fmt(r14, uniform):>22s}{fmt(r208, uniform):>22s}")
    print(f"{'reweighted (corrected posterior)':32s}{fmt(m_max, w):>22s}{fmt(r14, w):>22s}{fmt(r208, w):>22s}")
    names = ["logP1", "logP2", "logP3"]
    for k in range(3):
        print(f"  {names[k]}: old {fmt(draws[:, k], uniform)}  ->  reweighted {fmt(draws[:, k], w)}")

    with h5py.File(args.out, "w") as f:
        f.create_dataset("samples", data=draws)
        f.create_dataset("weights", data=w / np.sum(w))
        f.create_dataset("lnL_old", data=lnl_old)
        f.create_dataset("lnL_new", data=lnl_new)
        f.create_dataset("observables_new", data=np.stack([m_max, r14, r208], axis=1))
        f.attrs["ess"] = ess
    print(f"\nSaved {args.out}")

if __name__ == "__main__":
    main()
