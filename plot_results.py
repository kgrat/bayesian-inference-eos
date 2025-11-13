import numpy as np
import matplotlib.pyplot as plt
import corner
import time
import h5py 
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import os
import multiprocessing as mp 

# --- 1. PHYSICAL CONSTANTS ---
G = 6.67430e-11
c = 2.99792458e8
M_sun = 1.989e30
RHO_NUC = 2.7e17

# --- 2. CRUST EOS FUNCTIONS ---
CRUST_GAMMA = 1.58425
CRUST_K = 1.0557e5
RHO_STITCH = 0.5 * RHO_NUC
P_STITCH = CRUST_K * RHO_STITCH**CRUST_GAMMA

def crust_eos(rho):
    return CRUST_K * rho**CRUST_GAMMA

def inverse_crust_eos(P):
    if P <= 0.0:
        return 0.0
    return (P / CRUST_K)**(1.0 / CRUST_GAMMA)

# --- 3. CORE EOS BUILDER ---
RHO_1 = 1.85 * RHO_NUC; RHO_2 = 3.7 * RHO_NUC; RHO_3 = 7.4 * RHO_NUC
LOG_RHO_STITCH = np.log(RHO_STITCH); LOG_RHO_1 = np.log(RHO_1)
LOG_RHO_2 = np.log(RHO_2); LOG_RHO_3 = np.log(RHO_3)
RHO_MAX = 20 * RHO_NUC; LOG_RHO_MAX = np.log(RHO_MAX)
def build_eos_from_params(params):
    logP1, logP2, logP3 = params
    P1 = 10**logP1; P2 = 10**logP2; P3 = 10**logP3
    log_rhos = np.array([LOG_RHO_STITCH, LOG_RHO_1, LOG_RHO_2, LOG_RHO_3])
    log_Ps = np.array([np.log(P_STITCH), np.log(P1), np.log(P2), np.log(P3)])
    if not (np.all(np.diff(log_Ps) > 0)): raise ValueError("Pressure must increase with density.")
    gammas = (log_Ps[1:] - log_Ps[:-1]) / (log_rhos[1:] - log_rhos[:-1])
    gammas = np.append(gammas, [4.0])
    log_P_max = log_Ps[-1] + gammas[-1] * (LOG_RHO_MAX - log_rhos[-1])
    log_Ps = np.append(log_Ps, [log_P_max]); log_rhos = np.append(log_rhos, [LOG_RHO_MAX])
    log_Ks = log_Ps[:-1] - gammas * log_rhos[:-1]
    def final_eos(rho):
        if rho < RHO_STITCH: return crust_eos(rho)
        if rho > RHO_MAX: rho = RHO_MAX
        log_rho = np.log(rho)
        if log_rho < LOG_RHO_1: log_P = log_Ks[0] + gammas[0] * log_rho
        elif log_rho < LOG_RHO_2: log_P = log_Ks[1] + gammas[1] * log_rho
        elif log_rho < LOG_RHO_3: log_P = log_Ks[2] + gammas[2] * log_rho
        else: log_P = log_Ks[3] + gammas[3] * log_rho
        return np.exp(log_P)
    def final_inverse_eos(P):
        if P < P_STITCH: return inverse_crust_eos(P)
        if P > np.exp(log_Ps[-1]): return RHO_MAX
        log_P = np.log(P)
        if log_P < np.log(P1): log_rho = (log_P - log_Ks[0]) / gammas[0]
        elif log_P < np.log(P2): log_rho = (log_P - log_Ks[1]) / gammas[1]
        elif log_P < np.log(P3): log_rho = (log_P - log_Ks[2]) / gammas[2]
        else: log_rho = (log_P - log_Ks[3]) / gammas[3]
        return np.exp(log_rho)
    return final_eos, final_inverse_eos, None

# --- 4. TOV SOLVER ---
def tov_solver(r, state, eos_func, inverse_eos_func):
    M, P = state
    if r < 1e-6: r = 1e-6
    denominator_term = (1 - 2 * G * M / (r * c**2))
    if denominator_term <= 0: return [0, 0]
    rho = inverse_eos_func(P)
    if rho <= 0 or P <= 0: return [0, 0]
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
    def surface(r, state, eos_func, inverse_eos_func): return state[1] - 1.0
    surface.terminal = True
    sol = solve_ivp(
        tov_solver, [r_initial, 30e3], [M_initial, P_initial],
        args=(eos_func, inverse_eos_func), events=surface,
        method='RK45', rtol=1e-5
    )
    if not sol.success: return 0, 0
    final_R = sol.t[-1]; final_M = sol.y[0][-1]
    return final_M / M_sun, final_R / 1000.0
print("All physics functions defined.")

# --- 5. NEW: A "WORKER" FUNCTION FOR PARALLEL PLOTTING ---
def generate_one_mr_curve(params):
    """
    This is the "worker" function that one CPU core will run.
    It generates a single M-R curve for one set of parameters.
    """
    try:
        eos_func, inverse_eos_func, _ = build_eos_from_params(params)
        central_densities_plot = np.logspace(np.log10(RHO_NUC), np.log10(15 * RHO_NUC), 100)
        masses_plot = []
        radii_plot = []
        for rho_c in central_densities_plot:
            M, R = solve_for_central_density(rho_c, eos_func, inverse_eos_func, None)
            if M > 0.1:
                masses_plot.append(M)
                radii_plot.append(R)
        
        if masses_plot:
             return radii_plot, masses_plot
        else:
             return None # Return None if it failed
    except (ValueError, OverflowError):
        return None # Return None if it failed

# --- 6. Main execution block ---
# This "if" statement is CRITICAL for multiprocessing
if __name__ == "__main__":

    # --- 7. LOAD OBSERVATIONAL DATA (For plotting) ---
    script_dir = os.path.dirname(os.path.realpath(__file__))
    try:
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
        print("--- Successfully loaded all observational data for plotting ---")
    except Exception as e:
        print(f"--- ERROR loading data files ---")
        print(e)
        exit() # Exit if data can't be loaded

    # --- 8. LOAD MCMC RESULTS ---
    try:
        results_file = 'mcmc_results.h5'
        with h5py.File(results_file, 'r') as f:
            flat_samples = f['flat_samples'][:]
            chain = f['chain'][:] # Load the full chain for the trace plot
        print(f"\nSuccessfully loaded {len(flat_samples)} samples from '{results_file}'")
    except Exception as e:
        print(f"--- ERROR loading '{results_file}' ---")
        print(e)
        exit() # Exit if results can't be loaded

    # --- 9. MCMC TRACE PLOT ---
    print("Generating trace plot...")
    labels = [r"$\log_{10} P_1$", r"$\log_{10} P_2$", r"$\log_{10} P_3$"]
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    print("Shape of loaded chain:", chain.shape) 
    for i in range(3): 
        ax = axes[i]
        ax.plot(chain[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(chain))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("Step number");
    plt.suptitle("MCMC Walker Traces")
    plt.savefig("mcmc_traces.png")
    print("Saved 'mcmc_traces.png'")

    # --- 10. PLOT CORNER PLOT ---
    print("Generating corner plot...")
    print("Shape of flat_samples:", flat_samples.shape) 
    fig = corner.corner(
        flat_samples, labels=labels, show_titles=True, title_fmt=".3f"
    );
    plt.suptitle("Posterior Distribution for EoS Parameters")
    plt.savefig("mcmc_corner_plot.png")
    print("Saved 'mcmc_corner_plot.png'")

    # --- 11. PLOT M-R CREDIBLE BAND (Parallelized) ---
    print("Generating M-R credible band plot (in parallel)...")
    start_plot_time = time.time()
    
    if len(flat_samples) > 1000:
        inds = np.random.randint(len(flat_samples), size=1000)
        sample_params_list = flat_samples[inds]
    else:
        sample_params_list = flat_samples

    # --- This is the new parallel part ---
    # Determine number of processes to use
    n_proc = max(1, mp.cpu_count() - 1)
    print(f"Using {n_proc} cores to generate curves...")
    
    with mp.Pool(processes=n_proc) as pool:
        # pool.map runs the 'generate_one_mr_curve' function
        # for every item in 'sample_params_list' IN PARALLEL
        results = pool.map(generate_one_mr_curve, sample_params_list)
    # --- End of parallel part ---
    
    plt.figure(figsize=(10, 7))

    # Now we just plot the results
    for curve in results:
        if curve is not None:
            radii_plot, masses_plot = curve
            plt.plot(radii_plot, masses_plot, color='cornflowerblue', alpha=0.05)

    # --- Add Observational Constraints ---
    plt.errorbar(R_J0030_MEAN, 1.44, xerr=R_J0030_STD, fmt='o', color='red', capsize=5, label=f'NICER J0030 (R={R_J0030_MEAN:.2f} $\pm$ {R_J0030_STD:.2f} km)')
    plt.errorbar(R_LIGO_MEAN, 1.4, xerr=R_LIGO_STD, fmt='s', color='green', capsize=5, label=f'LIGO/Virgo (R={R_LIGO_MEAN:.2f} $\pm$ {R_LIGO_STD:.2f} km)')
    plt.errorbar(R_J0740_MEAN, 2.08, xerr=R_J0740_STD, fmt='^', color='orange', capsize=5, label=f'NICER J0740 (R={R_J0740_MEAN:.2f} $\pm$ {R_J0740_STD:.2f} km)')
    plt.axhspan(2.01, 2.15, color='grey', alpha=0.3, label='Max Mass Constraint')

    # --- Plot Formatting ---
    plt.title('Constrained Mass-Radius Relation', fontsize=16)
    plt.xlabel('Radius (km)', fontsize=14)
    plt.ylabel('Mass (Solar Masses)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xlim(8, 16)
    plt.ylim(0.5, 2.5)
    plt.savefig("mcmc_credible_band_rigorous.png")

    end_plot_time = time.time()
    print(f"Credible band plot complete in {(end_plot_time - start_plot_time)/60.0:.1f} minutes.")
    
    # --- 12. Show All Plots at the End ---
    print("\nAll plots saved. Displaying plots now...")
    plt.show() # Plotting ended

