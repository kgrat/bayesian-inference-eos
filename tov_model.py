import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import scipy.stats as stats

# Physical Constants (SI Units) 
G = 6.67430e-11        # Gravitational constant
c = 2.99792458e8         # Speed of light
M_sun = 1.989e30         # Mass of the sun in kg
RHO_NUC = 2.7e17         # Nuclear saturation density (kg/m^3)
# The Low-Density "Crust" EoS 
# Physically-correct constants
CRUST_GAMMA = 1.58425
CRUST_K = 1.0557e5
RHO_STITCH = 0.5 * RHO_NUC

def crust_eos(rho):
    """Calculates pressure from density for the low-density crust."""
    return CRUST_K * rho**CRUST_GAMMA

def inverse_crust_eos(P):
    """Calculates density from pressure for the low-density crust."""
    if P <= 0.0:
        return 0.0
    return (P / CRUST_K)**(1.0 / CRUST_GAMMA)

P_STITCH = crust_eos(RHO_STITCH)
print(f"Crust-Core Stitch Pressure (P_STITCH): {P_STITCH:.2e} Pa")
# The Parameterized "Core" EoS (Piecewise Polytrope)


RHO_1 = 1.85 * RHO_NUC
RHO_2 = 3.7 * RHO_NUC
RHO_3 = 7.4 * RHO_NUC

LOG_RHO_STITCH = np.log(RHO_STITCH)
LOG_RHO_1 = np.log(RHO_1)
LOG_RHO_2 = np.log(RHO_2)
LOG_RHO_3 = np.log(RHO_3)

RHO_MAX = 20 * RHO_NUC
LOG_RHO_MAX = np.log(RHO_MAX)

def build_eos_from_params(params):
    """
    Builds EoS functions from [logP1, logP2, logP3] parameters.
    Returns eos_func, inverse_eos_func, and None for cs2_func.
    """
    logP1, logP2, logP3 = params
    P1 = 10**logP1
    P2 = 10**logP2
    P3 = 10**logP3

    log_rhos = np.array([LOG_RHO_STITCH, LOG_RHO_1, LOG_RHO_2, LOG_RHO_3])
    log_Ps = np.array([np.log(P_STITCH), np.log(P1), np.log(P2), np.log(P3)])

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
# TOV Solver 

def tov_solver(r, state, eos_func, inverse_eos_func):
    """
    Defines the simplified differential equations for Mass and Pressure.
    state = [M, P]
    """
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

def solve_for_central_density(rho_c, eos_func, inverse_eos_func, _cs2_func):
    """
    Solves the TOV equations for a single star using correct initial conditions.
    Returns the final Mass and Radius.
    (_cs2_func is ignored)
    """
    if rho_c <= 0: return 0, 0
    P_c = eos_func(rho_c)
    if P_c <= 0: return 0, 0
    epsilon_c = rho_c * c**2 # Needed for correct P_initial

    r_initial = 1e-6
    M_initial = (4.0/3.0) * np.pi * rho_c * r_initial**3
    P_initial = P_c - (2.0*np.pi*G/c**2) * (epsilon_c + P_c) * (epsilon_c/3.0 + P_c) * r_initial**2

    def surface(r, state, eos_func, inverse_eos_func):
        return state[1] - 1.0
    surface.terminal = True

    sol = solve_ivp(
        tov_solver,
        [r_initial, 30e3],
        [M_initial, P_initial], # Only solve for [M, P]
        args=(eos_func, inverse_eos_func),
        events=surface,
        method='RK45',
        rtol=1e-5
    )

    if not sol.success:
        # print(f"Solver failed at rho_c={rho_c:.2e}: {sol.message}")
        return 0, 0

    final_R = sol.t[-1]
    final_M = sol.y[0][-1]

    return final_M / M_sun, final_R / 1000.0
# --- 5. The engine function ---
def generate_observables(params):
    """
    This is the master function that the MCMC will call.
    It takes the EoS parameters and returns M_max, R_1.4, and R_2.08
    """
    logP1, logP2, logP3 = params

    try:
        eos_func, inverse_eos_func, cs2_func = build_eos_from_params(params)
    except (ValueError, OverflowError):
        return 0, 0, 0 # M_max, R_1.4, R_2.08

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

    # This is the part that calculates R_2_08 
    if 2.08 > unique_masses[-1]:
         R_2_08 = 0 # Cannot calculate R_2.08 if M_max is too low
    else:
         R_2_08 = np.interp(2.08, unique_masses, radii_sorted[unique_idx])
    

    if np.isnan(R_1_4) or np.isnan(R_2_08):
        return M_max, 0, 0

    return M_max, R_1_4, R_2_08
# 6. Testing 
test_params = [33.79, 34.43, 35.56]

print(f"Testing with parameters: {test_params}")
M_max, R_1_4, R_2_08 = generate_observables(test_params) # Expects 3 values

print("--- Results ---")
print(f"Maximum Mass (M_sun): {M_max:.2f}")
print(f"Radius at 1.4 M_sun (km): {R_1_4:.2f}")
print(f"Radius at 2.08 M_sun (km): {R_2_08:.2f}") # Prints 3 values
# --- 7. Load ALL Real Observational Data ---


    # 1. Load LIGO/Virgo GW170817 Data
    # Columns: 0=m1, 1=m2, 4=Radius1, 5=Radius2
gw_data = np.loadtxt('EoS-insensitive_posterior_samples.dat', skiprows=1) #skip 1 header line
m1_gw = gw_data[:, 0]
m2_gw = gw_data[:, 1]
r1_gw = gw_data[:, 4]
r2_gw = gw_data[:, 5]

mask1_gw = (m1_gw > 1.35) & (m1_gw < 1.45)
mask2_gw = (m2_gw > 1.35) & (m2_gw < 1.45)
R_1_4_SAMPLES_LIGO = np.concatenate((r1_gw[mask1_gw], r2_gw[mask2_gw]))

R_LIGO_MEAN = np.mean(R_1_4_SAMPLES_LIGO)
R_LIGO_STD = np.std(R_1_4_SAMPLES_LIGO)

print("--- Successfully loaded GW170817 data ---")
print(f"R_1.4 (LIGO): {R_LIGO_MEAN:.2f} +/- {R_LIGO_STD:.2f} km")




# 2. Load NICER J0030+0451 Data
# Load J0030_data.dat
# Columns: 0=Radius, 1=Mass
j0030_data = np.loadtxt('J0030_data.dat', skiprows=2) #skip 2 header lines
j0030_radii = j0030_data[:, 0]  # Column 0 is Radius
j0030_masses = j0030_data[:, 1] # Column 1 is Mass

# We need R_1.4
mask_j0030 = (j0030_masses > 1.35) & (j0030_masses < 1.45)
R_1_4_SAMPLES_J0030 = j0030_radii[mask_j0030]

R_J0030_MEAN = np.mean(R_1_4_SAMPLES_J0030)
R_J0030_STD = np.std(R_1_4_SAMPLES_J0030)

print("--- Successfully loaded J0030+0451 data ---")
print(f"R_1.4 (J0030): {R_J0030_MEAN:.2f} +/- {R_J0030_STD:.2f} km")




# 3. Load NICER J0740+0620 Data
# Load J0740_data.dat
# Columns: 0=Radius, 1=Mass
j0740_data = np.loadtxt('J0740_data.dat', skiprows=2) # Skip 2 header lines


j0740_radii = j0740_data[:, 0]  # Column 0 is Radius
j0740_masses = j0740_data[:, 1] # Column 1 is Mass


# We need R_2.08
mask_j0740 = (j0740_masses > 2.0) & (j0740_masses < 2.15)
R_2_08_SAMPLES_J0740 = j0740_radii[mask_j0740]

R_J0740_MEAN = np.mean(R_2_08_SAMPLES_J0740)
R_J0740_STD = np.std(R_2_08_SAMPLES_J0740)

print("--- Successfully loaded J0740+0620 data ---")
print(f"R_2.08 (J0740): {R_J0740_MEAN:.2f} +/- {R_J0740_STD:.2f} km")


# 8. Likelihood Function

def log_likelihood(params):
    """
    Calculates the log-likelihood using data from all three multi-messenger sources.
    """
    # --- 1.  Model Predictions ---
    try:
        M_max, R_1_4, R_2_08 = generate_observables(params) # Expects 3 values
    except Exception as e:
        return -np.inf

    # 2. The Judge's First Cut for errors
    if M_max < 2.01:
        return -np.inf
    if R_1_4 == 0: # The 1.4 M_sun star MUST exist
        return -np.inf
    # Note: We can't fail if R_2_08 is 0, because the EoS might be too soft

    logP1, logP2, logP3 = params
    if not (33.0 < logP1 < 35.0): return -np.inf
    if not (34.0 < logP2 < 35.5): return -np.inf
    if not (35.0 < logP3 < 36.5): return -np.inf

    # 3. The Scoring (The "Likelihood")
    
    # Evidence A: NICER Radius (PSR J0030+0451)
    chi2_nicer_j0030 = ((R_1_4 - R_J0030_MEAN)**2) / (R_J0030_STD**2)
    
    # Evidence B: LIGO/Virgo Radius (GW170817)
    chi2_ligo_radius = ((R_1_4 - R_LIGO_MEAN)**2) / (R_LIGO_STD**2)
    
    # Start the score with the two tests that always apply
    final_score = -0.5 * (chi2_nicer_j0030 + chi2_ligo_radius)
    
    # Evidence C: NICER J0740 (Conditional Test)
    # we add the J0740 score *if* the EoS was stiff enough
    # to actually produce a 2.08 M_sun star.
    if R_2_08 != 0:
        chi2_nicer_j0740 = ((R_2_08 - R_J0740_MEAN)**2) / (R_J0740_STD**2)
        final_score += -0.5 * chi2_nicer_j0740

    return final_score

# Test the Likelihood Function
print("Testing the fully rigorous log_likelihood function...")
good_params = [33.79, 34.43, 35.56] 

print(f"Score for good parameters: {log_likelihood(good_params):.2f}")
