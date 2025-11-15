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
   `dM/dr = 4πr²ρ(r)`

2. **Hydrostatic Equilibrium (TOV Equation)**  
   `dP/dr = - [ G (ρ + P/c²) (M + 4πr³P/c²) ] / [ r² (1 - 2GM/(rc²)) ]`

Here:
- *M(r)* is the mass enclosed within radius *r*,
- *ρ(r)* and *P(r)* are local density and pressure,
- *G* and *c* are the gravitational constant and speed of light.

---

### Numerical Integration

To generate an **M–R curve** for a given EoS:

- Start at the center with a chosen **central density (ρ_c)** and integrate outward until **P(r) ≈ 0**.
- The radius at which pressure vanishes defines the star’s **surface radius (R)**.
- The corresponding mass M(R) gives the total **gravitational mass**.

Repeating this for multiple ρ_c values yields a full mass–radius relation.

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
3. Causality and monotonicity constraints are enforced automatically via the TOV solver.

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





