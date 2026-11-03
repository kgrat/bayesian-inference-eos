# The Physics Behind the Neutron Star EoS Inference

This document explains the scientific motivation, theoretical framework, and Bayesian methodology used to constrain the **Neutron Star Equation of State (EoS)** in this project.

---

## 1. Motivation

### Why Constraining the EoS Matters

The **Equation of State (EoS)** of dense matter describes the relationship between pressure and energy density, *P(ρ)*, at the extreme conditions inside neutron stars (NSs). These densities — exceeding that of an atomic nucleus (> 2.7 × 10¹⁷ kg/m³) — cannot be reproduced in terrestrial laboratories.

Constraining the EoS is a major goal of modern nuclear and astrophysical research.

---

### The Multi-Messenger “Tension”

Modern astrophysical observations create a **tension** between “soft” and “stiff” EoS models:

| Observation | Type of EoS Favored |
|--------------|---------------------|
| **High-mass pulsars** (e.g. PSR J0740+0620, ~2.08 M⊙) | **Stiff EoS** (larger radii) |
| **LIGO/Virgo GW170817** (low tidal deformability) | **Soft EoS** (smaller radii) |

The challenge is to find the “sweet spot” — EoS models stiff enough to support ~2 M⊙ pulsars yet soft enough to satisfy the compactness inferred from LIGO/Virgo and NICER observations.

---

## 2. The Piecewise Polytropic Model

To describe an unknown EoS flexibly, we use a **three-segment piecewise polytropic model**, following Read et al. (2009).

The star is divided into:

1. **Crust:**  
   For low densities (ρ < 0.5 ρₙᵤ𝚌), the EoS is well understood.  
   A fixed polytropic relation is used:  
   *P = K ρ^Γ*, with *K = 1.0557 × 10⁵* and *Γ = 1.58425*.

2. **Core:**  
   For higher densities, we define three “knot” points at fixed multiples of nuclear saturation density (ρₙᵤ𝚌):

   - ρ₁ = 1.85 ρₙᵤ𝚌  
   - ρ₂ = 3.7 ρₙᵤ𝚌  
   - ρ₃ = 7.4 ρₙᵤ𝚌  

   The corresponding log-pressures (**log P₁, log P₂, log P₃**) are the three free parameters of the model.

These determine the stiffness of the EoS in different regions — low densities (radius-sensitive) to high densities (mass-sensitive).

---

### Continuity and Segment Derivation

Between two knots (ρᵢ, Pᵢ) and (ρᵢ₊₁, Pᵢ₊₁), the EoS is defined as:

(ρ) = Kᵢ * ρ^(Γᵢ)

Continuity requires solving for Γᵢ as:

Γᵢ = [ ln(Pᵢ₊₁ / Pᵢ) ] / [ ln(ρᵢ₊₁ / ρᵢ) ]


and then determining:

Kᵢ = Pᵢ / ρᵢ^(Γᵢ)


This ensures a smooth EoS fully defined by (P₁, P₂, P₃).

---

## 3. The TOV Equations

The EoS is converted into observable stellar properties (mass and radius) by integrating the **Tolman–Oppenheimer–Volkoff (TOV)** equations — the relativistic equations of stellar structure.

### TOV System

1. **Mass Continuity**  
   `dM/dr = 4πr² ε(r)/c²`

2. **Hydrostatic Equilibrium (TOV Equation)**  
   `dP/dr = - [ G (ε/c² + P/c²) (M + 4πr³P/c²) ] / [ r² (1 - 2GM/(rc²)) ]`

Here:
- *M(r)* is the mass-energy enclosed within radius *r*,
- *ε(r)* is the **total energy density** (rest mass plus internal energy) and *P(r)* the pressure,
- *G* and *c* are the gravitational constant and speed of light.

Gravity couples to all of the energy, so it is ε, not the rest-mass density ρ, that appears in both equations. Earlier versions of this code used ρ in place of ε/c²; see the Changelog in the README.

### Energy Density of a Piecewise Polytrope

For cold matter the first law of thermodynamics reads `d(ε/ρ) = -P d(1/ρ)`. For a segment `P = Kᵢ ρ^Γᵢ` on `ρᵢ₋₁ ≤ ρ < ρᵢ` this integrates to (Read et al. 2009, Eqs. 6–7)

`ε(ρ) = (1 + aᵢ) ρc² + Kᵢ ρ^Γᵢ / (Γᵢ − 1)`

`aᵢ = ε(ρᵢ₋₁)/(ρᵢ₋₁c²) − 1 − Kᵢ ρᵢ₋₁^(Γᵢ−1) / ((Γᵢ − 1)c²)`

The constants aᵢ make ε continuous at each dividing density, and a₀ = 0 for the crust, so ε → ρc² as ρ → 0. The second term, P/(Γᵢ − 1), is the internal (compression) energy. The segments are the crust, the three core segments between the knots, and a Γ = 4 extension above ρ₃. Inside the ODE, ε(P) is obtained as P → ρ (inverse EoS) → ε(ρ).

### Numerical Integration

To generate an **M–R curve** for a given EoS:

- Start at r₀ = 1 µm with the series solution
  `M(r₀) = (4/3)π (ε_c/c²) r₀³` and
  `P(r₀) = P_c − (2πG/3c⁴)(ε_c + P_c)(ε_c + 3P_c) r₀²`,
  where ε_c = ε(ρ_c).
- Integrate outward (`scipy` RK45) until **P(r) ≈ 0**; that radius is the star's **surface radius (R)**.
- The corresponding M(R) is the total **gravitational mass**.

Repeating this for 50 central densities between ρ_nuc and 15 ρ_nuc yields the mass–radius relation.

---

## 4. Bayesian Inference Framework

We infer the posterior distribution of EoS parameters **θ = (log P₁, log P₂, log P₃)** given data *D* using Bayes’ theorem:

P(θ | D) ∝ L(D | θ) × π(θ)


Where:
- **P(θ | D)**: Posterior (what we seek)
- **L(D | θ)**: Likelihood (how well a model fits the data)
- **π(θ)**: Prior (physical and empirical constraints)

---

### Priors

1. The EoS must support **Mₘₐₓ > 2.01 M⊙**.  
2. Pressures must lie in a physically reasonable range (e.g., 33.0 < log P₁ < 36.5).  
3. Pressure must increase monotonically across the knots (enforced). Causality (c_s ≤ c) is **not** enforced.

---

### Likelihood Function

The total log-likelihood is:

ln L_total = ln L_J0030 + ln L_LIGO + ln L_J0740


Each term is modeled as a Gaussian χ² likelihood comparing predicted and observed mass–radius values:

- **L_J0030:** Radius of a 1.44 M⊙ star (NICER)  
- **L_LIGO:** Radius constraint from GW170817  
- **L_J0740:** Radius at 2.08 M⊙, conditional on supporting that mass  

The combined likelihood encodes the relation between soft and stiff constraints.

---

### MCMC Sampling

We use **`emcee`**, a Markov Chain Monte Carlo (MCMC) sampler, to efficiently explore the 3D parameter space and produce the posterior distribution visualized in the **corner plot**.

---

## 5. Interpreting the Results

The **mass–radius credible band** represents the posterior EoS ensemble.  
Each line corresponds to one sampled EoS, and the band width reflects uncertainty.

Findings:
- EoS must be **stiff enough** to reach >2.01 M⊙.  
- EoS must be **soft enough** to match NICER and LIGO radius constraints.  

### Limitations & Future Work

Limitations:
1. Smooth, piecewise-polytropic behavior (no phase transitions).  
2. Gaussian-approximated observational posteriors.
3. Exclusion of tidal deformibility incorporation due to bugs.

Future Work:
- **Hybrid EoS models:** Include quark matter transitions/phase transitions.  
- **2D likelihoods:** Use full M–R posterior contours from NICER and LIGO (Kernel Density Estimation).
- **Direct Λ (tidal deformability):** Incorporate LIGO’s Λ posteriors directly.
- **Nested sampling** Use it to better calculate the bayesian evidence.

---

## 6. References

- **LIGO/Virgo:** Abbott et al. (2018), *Phys. Rev. Lett.*, 121, 161101.  
- **NICER (J0030):** Miller et al. (2019), Riley et al. (2019), *ApJ Letters*, 887.  
- **NICER (J0740):** Miller et al. (2021), Riley et al. (2021), *ApJ Letters*, 918.  
- **Piecewise Polytrope:** Read et al. (2009), *Phys. Rev. D*, 79, 124032.  
- **TOV Equations:** Tolman (1939); Oppenheimer & Volkoff (1939), *Phys. Rev.*, 55.

---





