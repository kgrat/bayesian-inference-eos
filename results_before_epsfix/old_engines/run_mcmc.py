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
# (All functions from tov_model.py used here)
# -------------------------------
G = 6.67430e-11
c = 2.99792458e8
M_sun = 1.989e30
RHO_NUC = 2.7e17
CRUST_GAMMA = 1.58425
CRUST_K = 1.0557e5
RHO_STITCH = 0.5 * RHO_NUC
P_STITCH = CRUST_K * RHO_STITCH**CRUST_GAMMA
RHO_1 = 1.85 * RHO_NUC
RHO_2 = 3.7 * RHO_NUC
RHO_3 = 7.4 * RHO_NUC
LOG_RHO_STITCH = np.log(RHO_STITCH)
LOG_RHO_1 = np.log(RHO_1)
LOG_RHO_2 = np.log(RHO_2)
LOG_RHO_3 = np.log(RHO_3)
RHO_MAX = 20 * RHO_NUC
LOG_RHO_MAX = np.log(RHO_MAX)

def crust_eos(rho):
    return CRUST_K * rho**CRUST_GAMMA

def inverse_crust_eos(P):
    if P <= 0.0:
        return 0.0
    return (P / CRUST_K)**(1.0 / CRUST_GAMMA)

def build_eos_from_params(params):
    logP1, logP2, logP3 = params
    P1 = 10**logP1
    P2 = 10**logP2
    P3 = 10**logP3
    log_rhos = np.array([LOG_RHO_STITCH, LOG_RHO_1, LOG_RHO_2, LOG_RHO_3])
    log_Ps = np.array([np.log(P_STITCH), np.log(P1), np.log(P2), np.log(P3)])
    if not (np.all(np.diff(log_Ps) > 0)):
        raise ValueError("Pressure must increase with density.")
    gammas = (log_Ps[1:] - log_Ps[:-1]) / (log_rhos[1:] - log_rhos[:-1])
    gammas = np.append(gammas, [4.0])
    log_P_max = log_Ps[-1] + gammas[-1] * (LOG_RHO_MAX - log_rhos[-1])
    log_Ps = np.append(log_Ps, [log_P_max])
    log_rhos = np.append(log_rhos, [LOG_RHO_MAX])
    log_Ks = log_Ps[:-1] - gammas * log_rhos[:-1]

    def final_eos(rho):
        if rho < RHO_STITCH:
            return crust_eos(rho)
        if rho > RHO_MAX:
            rho = RHO_MAX
        log_rho = np.log(rho)
        if log_rho < LOG_RHO_1:
            log_P = log_Ks[0] + gammas[0] * log_rho
        elif log_rho < LOG_RHO_2:
            log_P = log_Ks[1] + gammas[1] * log_rho
        elif log_rho < LOG_RHO_3:
            log_P = log_Ks[2] + gammas[2] * log_rho
        else:
            log_P = log_Ks[3] + gammas[3] * log_rho
        return np.exp(log_P)

    def final_inverse_eos(P):
        if P < P_STITCH:
            return inverse_crust_eos(P)
        if P > np.exp(log_Ps[-1]):
            return RHO_MAX
        log_P = np.log(P)
        if log_P < np.log(P1):
            log_rho = (log_P - log_Ks[0]) / gammas[0]
        elif log_P < np.log(P2):
            log_rho = (log_P - log_Ks[1]) / gammas[1]
        elif log_P < np.log(P3):
            log_rho = (log_P - log_Ks[2]) / gammas[2]
        else:
            log_rho = (log_P - log_Ks[3]) / gammas[3]
        return np.exp(log_rho)
    return final_eos, final_inverse_eos, None

def tov_solver(r, state, eos_func, inverse_eos_func):
    M, P = state
    if r < 1e-6: r = 1e-6
    denominator_term = (1 - 2 * G * M / (r * c**2))
    if denominator_term <= 0:
        return [0, 0]
    rho = inverse_eos_func(P)
    if rho <= 0 or P <= 0:
        return [0, 0]
    dMdr = 4 * np.pi * r**2 * rho
    term1_P = (rho + P / c**2) * (M + 4 * np.pi * r**3 * P / c**2)
    term2_P = r**2 * denominator_term
    dPdr = -G * term1_P / term2_P
    return [dMdr, dPdr]

def solve_for_central_density(rho_c, eos_func, inverse_eos_func, _cs2_func=None):
    if rho_c <= 0: return 0, 0
    P_c = eos_func(rho_c)
    if P_c <= 0: return 0, 0
    epsilon_c = rho_c * c**2
    r_initial = 1e-6
    M_initial = (4.0/3.0) * np.pi * rho_c * r_initial**3
    P_initial = P_c - (2.0*np.pi*G/c**2) * (epsilon_c + P_c) * (epsilon_c/3.0 + P_c) * r_initial**2
    def surface(r, state, eos_func, inverse_eos_func):
        return state[1] - 1.0
    surface.terminal = True
    sol = solve_ivp(
        tov_solver, [r_initial, 30e3], [M_initial, P_initial],
        args=(eos_func, inverse_eos_func), events=surface,
        method='RK45', rtol=1e-5,
    )
    if not sol.success:
        return 0, 0
    final_R = sol.t[-1]
    final_M = sol.y[0][-1]
    return final_M / M_sun, final_R / 1000.0

def generate_observables(params):
    try:
        eos_func, inverse_eos_func, cs2_func = build_eos_from_params(params)
    except Exception:
        return 0, 0, 0
    central_densities = np.logspace(np.log10(RHO_NUC), np.log10(15 * RHO_NUC), 50)
    masses = []
    radii = []
    for rho_c in central_densities:
        M, R = solve_for_central_density(rho_c, eos_func, inverse_eos_func, cs2_func)
        if M > 0.1:
            masses.append(M)
            radii.append(R)
    if not masses:
        return 0, 0, 0
    M_max = np.max(masses)
    if M_max < 2.01:
        return M_max, 0, 0
    sort_idx = np.argsort(masses)
    masses_sorted = np.array(masses)[sort_idx]
    radii_sorted = np.array(radii)[sort_idx]
    unique_masses, unique_idx = np.unique(masses_sorted, return_index=True)
    if len(unique_masses) < 2:
        return M_max, 0, 0
    R_1_4 = np.interp(1.4, unique_masses, radii_sorted[unique_idx])
    if 2.08 > unique_masses[-1]:
        R_2_08 = 0
    else:
        R_2_08 = np.interp(2.08, unique_masses, radii_sorted[unique_idx])
    if np.isnan(R_1_4) or np.isnan(R_2_08):
        return M_max, 0, 0
    return M_max, R_1_4, R_2_08

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
