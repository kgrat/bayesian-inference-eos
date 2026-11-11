# Bayesian Inference of the Neutron Star Equation of State

This project constrains the neutron star (NS) Equation of State (EoS) by performing a Bayesian MCMC analysis on a 3-parameter piecewise polytropic EoS model.

The likelihood uses three multi-messenger astronomical observations, each reduced to a Gaussian on the radius at a fixed mass:
1.  **LIGO/Virgo:** The gravitational wave event GW170817 (using the EoS-insensitive radius posteriors).
2.  **NICER (PSR J0030+0451):** The mass-radius posteriors for the 1.44 M solar mass pulsar.
3.  **NICER (PSR J0740+6620):** The mass-radius posteriors for the high-mass ~2.08 M solar mass pulsar. 

For the full theoretical background and derivations, see [PHYSICS.md](./PHYSICS.md).


---

## Results

MCMC with `emcee`: 50 walkers, 300 burn-in + 700 steps, with the corrected physics and physical cuts (see [Corrections](#corrections)). Medians and 90% intervals:

| Quantity | Posterior |
| :--- | :--- |
| Maximum mass M_max | 2.22 [2.08, 2.38] M☉ |
| Radius R_1.4 | 12.55 [11.68, 13.40] km |
| Radius R_2.08 | 11.92 [11.15, 12.66] km |
| log₁₀ P₁ (at 1.85 ρ_nuc, Pa) | 33.38 [33.20, 33.53] |
| log₁₀ P₂ (at 3.7 ρ_nuc, Pa) | 34.48 [34.37, 34.57] |
| log₁₀ P₃ (at 7.4 ρ_nuc, Pa) | 35.16 [35.02, 35.32] |

These agree with the nested-sampling polytrope in [EoS-model-selection](https://github.com/kgrat/EoS-model-selection) (M_max ≈ 2.24 M☉, R_1.4 ≈ 12.7 km).

**Convergence.** The mean acceptance fraction is 0.56, and no walkers are stuck. The integrated autocorrelation time is about 31 steps, so the 700 post-burn-in steps are about 22 autocorrelation times (roughly 1,100 independent samples). That is enough for medians and 90% intervals; a longer chain would refine the tails. The medians of the first and second halves of the chain agree to within 0.013.

**Prior edge.** About 12% of the log P₃ samples lie within 0.05 of the lower prior bound (35.0). The causality cut favours a softer high-density EoS, so the lower end of the log P₃ interval is partly set by the prior.

### 1. Constrained Mass-Radius Relation

The credible band shows the EoS models allowed by the data. The EoS must be soft enough to match the GW170817 radius constraint (green) and stiff enough to support massive pulsars such as J0740 (orange) and the 2.01 M☉ limit (grey).

![Constrained M-R Plot](mr_constrained.png)

### 2. Parameter Posterior Distributions

Posterior distributions of the three EoS parameters (log-pressure at 1.85, 3.7 and 7.4 times nuclear saturation density) and their correlations.

![Corner Plot](mcmc_corner_plot.png)

### 3. MCMC Convergence

Walker traces, showing burn-in and stationarity.

![Trace Plot](mcmc_traces.png)

---

## How to Run This Project

This project is structured into three main Python files that work together. The correct sequence to reproduce the results is:

1.  `tov_model.py`: **This is the core physics engine.** It contains the functions for building the EoS, its energy density, and solving the TOV equations. It is imported as a module by the other scripts; running it directly performs a test and evaluates the likelihood on example parameters.

2.  `run_mcmc.py`: **This is the main computation script.** It imports the engine from `tov_model.py` and runs the MCMC simulation using `emcee` and `multiprocessing`. This is the script that takes several hours.

3.  `plot_results.py`: **This is the analysis script.** After the MCMC is done, you run this script. It imports the *same* physics functions from `tov_model.py` and loads the results from the MCMC run (`mcmc_results.h5`) to generate all the final plots.

### Step-by-Step Instructions

#### 1. Setup
Make sure you have all the necessary files in the same directory:
* `tov_model.py`
* `run_mcmc.py`
* `plot_results.py`
* `EoS-insensitive_posterior_samples.dat`
* `J0030_data.dat`
* `J0740_data.dat`

#### 2. Dependencies
And install the required libraries:
```bash
pip install numpy scipy matplotlib emcee corner h5py
```

#### 3. Run the TOV engine
Run the [tov_model.py](tov_model.py) file. 
```bash
python tov_model.py
```

#### 4. Run the MCMC Simulation
Execute the [run_mcmc.py](run_mcmc.py) script from your terminal. This will use all available CPU cores to run the simulation.

```bash
# Full run: 50 walkers, 500 burn-in + 2000 steps
python run_mcmc.py --n-burn 500 --n-steps 2000
```
This will run for several hours and create a new file: mcmc_results.h5.

#### 5. Generate the Plots
Once mcmc_results.h5 exists, run the plotting script:

```bash
python plot_results.py
```

This script will load mcmc_results.h5, generate all three plots (Trace, Corner, and M-R Band), and save them as .png files in your folder.

---

## Corrections

An earlier version of this code had physics errors. They are fixed, and the results above come from the corrected code.

* **Energy density in the TOV equations.** The solver used the rest-mass density ρ where the TOV equations need the total energy density ε. It now uses ε(ρ) = (1 + aᵢ)ρc² + Kᵢρ^Γᵢ/(Γᵢ − 1) for each polytrope segment, with aᵢ set by continuity (Read et al. 2009; see PHYSICS.md). For a fixed EoS this lowers M_max by about 0.4 M☉. The central series expansion also had a c² that should have been c⁴.
* **Physical cuts.** The likelihood now rejects acausal EoSs (sound speed above c inside a stable star) and reads radii off the stable branch only. The J0740 term is averaged over the pulsar's mass distribution instead of being skipped when M_max < 2.08 M☉. Most of the pre-correction posterior was acausal, and its median M_max was 3.2 M☉.
* **Code structure.** `run_mcmc.py` and `plot_results.py` now import the physics from `tov_model.py` instead of carrying their own copies.

The pre-correction results are kept under the git tag `epsfix-comparison-archive`.

---

## Acknowledgments

**AI disclosure:** In a later debugging stage I used an AI coding assistant to help find and fix bugs and update the documentation. The project design and original implementation are my own.

---

## Reproducibility

All data files and posterior samples used in this project are publicly available:

1. LIGO/Virgo GW170817 posteriors (Abbott et al., 2018)

2. NICER PSR J0030+0451 (Miller et al., 2019)

3. NICER PSR J0740+6620 (Riley et al., 2021)

The entire pipeline can be reproduced on any system with Python ≥ 3.8.

---

## References

- **LIGO/Virgo:** Abbott et al. (2018), *Phys. Rev. Lett.*, 121, 161101.  
- **NICER (J0030):** Miller et al. (2019), Riley et al. (2019), *ApJ Letters*, 887.  
- **NICER (J0740):** Miller et al. (2021), Riley et al. (2021), *ApJ Letters*, 918.

---
