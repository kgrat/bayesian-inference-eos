#!/usr/bin/env python3
"""
mcmc_runner.py

The log_likelihood function is now at the top level and reads data from global variables.
"""

import argparse
import multiprocessing as mp
import os
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import interp1d

try:
    import emcee
    import h5py # using h5py to save results
except Exception:
    raise RuntimeError("emcee and h5py are required. Install with: pip install emcee h5py")

# -------------------------------
# Physics / EoS / TOV functions
# (single source of truth: tov_model.py)
# -------------------------------
from tov_model import generate_observables

# -------------------------------
# Data loading (as global variables)
# -------------------------------
print("Loading observational data...")
try:
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    gw_path = os.path.join(script_dir, 'EoS-insensitive_posterior_samples.dat')
    j0030_path = os.path.join(script_dir, 'J0030_data.dat')
    j0740_path = os.path.join(script_dir, 'J0740_data.dat')

    gw_data = np.loadtxt(gw_path, skiprows=1)
    m1_gw = gw_data[:, 0]; m2_gw = gw_data[:, 1]; r1_gw = gw_data[:, 4]; r2_gw = gw_data[:, 5]
    mask1_gw = (m1_gw > 1.35) & (m1_gw < 1.45); mask2_gw = (m2_gw > 1.35) & (m2_gw < 1.45)
    R_1_4_SAMPLES_LIGO = np.concatenate((r1_gw[mask1_gw], r2_gw[mask2_gw]))
    R_LIGO_MEAN = np.mean(R_1_4_SAMPLES_LIGO); R_LIGO_STD = np.std(R_1_4_SAMPLES_LIGO)

    j0030_data = np.loadtxt(j0030_path, skiprows=2)
    j0030_radii = j0030_data[:, 0]; j0030_masses = j0030_data[:, 1]
    mask_j0030 = (j0030_masses > 1.35) & (j0030_masses < 1.45)
    R_1_4_SAMPLES_J0030 = j0030_radii[mask_j0030]
    R_J0030_MEAN = np.mean(R_1_4_SAMPLES_J0030); R_J0030_STD = np.std(R_1_4_SAMPLES_J0030)

    j0740_data = np.loadtxt(j0740_path, skiprows=2)
    j0740_radii = j0740_data[:, 0]; j0740_masses = j0740_data[:, 1]
    mask_j0740 = (j0740_masses > 2.0) & (j0740_masses < 2.15)
    R_2_08_SAMPLES_J0740 = j0740_radii[mask_j0740]
    R_J0740_MEAN = np.mean(R_2_08_SAMPLES_J0740); R_J0740_STD = np.std(R_2_08_SAMPLES_J0740)
    
    print("--- Successfully loaded all data ---")
    print(f"   R_1.4 (LIGO): {R_LIGO_MEAN:.2f} +/- {R_LIGO_STD:.2f} km")
    print(f"   R_1.4 (J0030): {R_J0030_MEAN:.2f} +/- {R_J0030_STD:.2f} km")
    print(f"   R_2.08 (J0740): {R_J0740_MEAN:.2f} +/- {R_J0740_STD:.2f} km")

except Exception as e:
    print(f"CRITICAL ERROR: Could not load data files. Make sure they are in the same folder as this script.")
    print(f"Error details: {e}")
    exit()

# -------------------------------
# Log_Likelihood function 
# -------------------------------
def log_likelihood(params):
    """
    Calculates the log-likelihood using the globally loaded data.
    """
    try:
        M_max, R_1_4, R_2_08 = generate_observables(params)
    except Exception:
        return -np.inf
        
    if M_max < 2.01:
        return -np.inf
    if R_1_4 == 0:
        return -np.inf
        
    logP1, logP2, logP3 = params
    if not (33.0 < logP1 < 35.0): return -np.inf
    if not (34.0 < logP2 < 35.5): return -np.inf
    if not (35.0 < logP3 < 36.5): return -np.inf

    chi2_nicer_j0030 = ((R_1_4 - R_J0030_MEAN)**2) / (R_J0030_STD**2)
    chi2_ligo_radius = ((R_1_4 - R_LIGO_MEAN)**2) / (R_LIGO_STD**2)
    final_score = -0.5 * (chi2_nicer_j0030 + chi2_ligo_radius)
    
    if R_2_08 != 0:
        chi2_j0740 = ((R_2_08 - R_J0740_MEAN)**2) / (R_J0740_STD**2)
        final_score += -0.5 * chi2_j0740
        
    return final_score

# -------------------------------
# MCMC functions
# -------------------------------
def run_emcee(n_walkers, n_burn, n_steps, n_proc, initial_params, output):
    
    n_dim = len(initial_params)
    initial_pos = initial_params + 1e-3 * np.random.randn(n_walkers, n_dim)

    print(f"Running emcee with {n_walkers} walkers, {n_burn} burn, {n_steps} steps, {n_proc} processes")
    start = time.time()

    # Using multiprocessing Pool to parallelize
    with mp.Pool(processes=n_proc) as pool:
        # We pass the log_likelihood function, which is now pickle-able
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_likelihood, pool=pool)
        sampler.run_mcmc(initial_pos, n_burn + n_steps, progress=True)

    end = time.time()
    print(f"MCMC completed in {(end-start)/60.0:.2f} minutes")

    # Save results
    samples = sampler.get_chain()
    flat = sampler.get_chain(discard=n_burn, flat=True)
    
    # Save as HDF5
    output_file = output.replace('.npz', '.h5') # we repalce .npz with .h5 as .h5 runs better on our environment and is less buggy
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('chain', data=samples)
        f.create_dataset('flat_samples', data=flat)
        f.create_dataset('labels', data=np.array(['logP1', 'logP2', 'logP3'], dtype='S'))
        
    print(f"Saved sampler output to {output_file}")


def parse_args():
    p = argparse.ArgumentParser(description='Parallel MCMC runner for EoS project')
    p.add_argument('--n-walkers', type=int, default=50)
    p.add_argument('--n-burn', type=int, default=100)
    p.add_argument('--n-steps', type=int, default=200)
    p.add_argument('--n-proc', type=int, default=max(1, mp.cpu_count()-1), help='Number of processes to use')
    p.add_argument('--output', type=str, default='mcmc_results.h5')
    p.add_argument('--init', type=float, nargs=3, default=[33.79, 34.43, 35.56], help='Initial logP1 logP2 logP3')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_emcee(args.n_walkers, args.n_burn, args.n_steps, args.n_proc, np.array(args.init), args.output)
