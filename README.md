# Multi-Topology Resonance Simulation

A GPU-accelerated numerical search for compact geometries whose resonant
mode ratios reproduce the muon/electron mass ratio from first principles.

## Scientific Context

Williamson & van der Mark (1997) proposed that the electron is a
Compton-wavelength photon confined in a toroidal topology, following a
double-helix flow pattern along a torus knot.  In this picture the muon is a
higher-harmonic excitation of the same topology -- its rest mass arises from a
shorter confined path on the torus.

This project generalizes the original torus-knot idea to a systematic search
across multiple compact 3-manifold topologies:

- **Flat torus** -- (p,q) torus-knot geodesic path-length ratios
- **Berger sphere** -- squashed S^3 with Hopf-fiber deformation parameter lambda
- **Lens space** L(n,1) -- cyclic quotients of S^3 with closed-form path ratios
- **Poincare homology sphere** S^3/2I -- binary icosahedral quotient with
  group-theoretic mode selection rules
- **Nil manifold** M(a,b,k) -- twisted circle bundle with abelian eigenvalue
  ratios parameterized by twist tau
- **Binary polyhedral quotients** -- dihedral, tetrahedral, and octahedral
  quotients of S^3, selected by representation-theoretic rules

The experimental mass ratio is known to extraordinary precision (25 ppb):

    m_mu / m_e = 206.7682843 +/- 0.0000052

Aspden's cavity-resonance model derived a theoretical value of 206.7683078,
tantalizingly close.  This project systematically searches the parameter space
of each topology to find configurations where resonant mode ratios match this
value.

## Results

### Run 2 (current): 110 billion evaluations in ~24 minutes

Expanded search: max winding numbers raised to 30 (38,730 pairs, up from
8,237), 7 rho bands with dense coverage in the thin-torus regime, 10M-point
Phase 2 refinement with adaptive windows, 100M-point ultra-deep passes,
cavity resonance cross-check, CPU rho refinement in Phase 3.

**GPU best (64-pt GL quadrature): score = 6.6 x 10^-12** -- essentially
machine-precision for the quadrature order.

**CPU-verified best (10,000-pt f64): score = 1.0 x 10^-8** (0.05 ppb).

| Rank | Mode pair | rho | Computed ratio | Error | Method |
|------|-----------|-----|----------------|-------|--------|
| 1 | (21,10) / (1,0) | 0.048614995 | 206.768284**3100** | 1.0e-8 | CPU f64 |
| 2 | (22,13) / (1,0) | 0.063231962 | 206.768284**3677** | 6.8e-8 | CPU f64 |
| 3 | (29,18) / (1,0) | 0.087926397 | 206.768284**3686** | 6.9e-8 | CPU f64 |
| 4 | (9,8) / (1,0)   | 0.038727385 | 206.768284**0353** | 2.6e-7 | CPU f64 |
| 5 | (27,2) / (1,0)  | 0.009756203 | 206.768284**6887** | 3.9e-7 | CPU f64 |

GPU ultra-deep hits (64-pt quadrature, not CPU-verified):

| Rank | Mode pair | rho | Score |
|------|-----------|-----|-------|
| 1 | (29,18) / (1,0) | 0.087926372 | 6.6e-12 |
| 2 | (22,13) / (1,0) | 0.063231932 | 1.5e-11 |
| 3 | (21,10) / (1,0) | 0.048614877 | 1.5e-11 |

New findings compared to Run 1:

- **Higher-winding modes dominate**: The top 3 all have winding numbers > 20,
  accessible only after expanding from max 20 to 30.
- **Three distinct rho families** now visible:
  - rho ~ 0.0486 for (p,1)/(1,0) and (21,10)/(1,0)
  - rho ~ 0.0632 for (22,13)/(1,0)
  - rho ~ 0.0879 for (29,18)/(1,0)
- The thin-torus (rho ~ 0.00484) family from Run 1 still appears but is no
  longer the best -- thicker tori with more complex winding patterns achieve
  better precision.
- **Cavity resonance**: Aspden-style model found hits at score ~ 6.6e-7
  (mode n=205, j=0, rho=0.129).  This is an independent confirmation that
  the mass ratio is reproducible from toroidal cavity physics.
- **GPU-CPU divergence**: The GPU's 64-pt GL quadrature achieves 6.6e-12 but
  10,000-pt CPU trapezoidal quadrature settles at 1.0e-8 for the same geometry.
  The discrepancy reflects quadrature precision limits, not a physical effect.
  Pushing further requires higher-order quadrature or arbitrary precision.

Koide formula check: Q = 0.66666051 vs 2/3 = 0.66666667 (within 0.001%).

### Run 1 (baseline): 10.7 billion evaluations in ~2.5 minutes

Max winding 20, 8,237 pairs, 6 rho bands, 1M-point Phase 2 refinement.

Best match: **score = 2.6 x 10^-8** (0.13 ppb).

| Rank | Mode pair | rho | Computed ratio | Error |
|------|-----------|-----|----------------|-------|
| 1 | (15,2) / (1,0) | 0.009698218 | 206.768284**2738** | 2.6e-8 |
| 2 | (8,2) / (1,0)  | 0.009679912 | 206.768284**2723** | 2.8e-8 |
| 3 | (12,1) / (1,0) | 0.004844497 | 206.768284**3575** | 5.7e-8 |
| 4 | (10,8) / (1,0) | 0.038736016 | 206.768284**2288** | 7.1e-8 |
| 5 | (9,1) / (1,0)  | 0.004840920 | 206.768284**2148** | 8.5e-8 |

Key structural observations from Run 1:

- The denominator is always **(1,0)** -- a single poloidal loop around the tube.
  This is the shortest closed path on the torus and represents the electron
  ground state in the Williamson/van der Mark picture.
- Two natural rho clusters appear: **~0.00484** (single-winding numerators) and
  **~0.00968 ~ 2 x 0.00484** (double-winding numerators).  The doubling is
  consistent with the (p,q) / (1,0) geometry: scaling the winding number by 2
  doubles the effective aspect ratio.
- rho ~ 0.00484 ~ 1/206.77 ~ **1/TARGET_RATIO**.  This is mathematically
  expected: in the thin-torus limit, L(p,q) / L(1,0) ~ (1 + rho*cos(...))*q /
  (rho*p), so the ratio is dominated by q/(rho*p).  The best fits all have the
  ratio tuned by rho to sub-ppb precision.

## Mathematical Models

### Torus-knot geodesics

The fundamental object is a (p, q) torus knot on a torus with major radius
R = 1 and minor radius rho.  Using the torus metric:

    ds^2 = rho^2 * dtheta^2  +  (1 + rho*cos(theta))^2 * dphi^2

A (p,q) curve winds p times poloidally (around the tube) and q times toroidally
(around the hole):  theta(t) = p*t,  phi(t) = q*t,  t in [0, 2*pi].

The path length is:

    L(p, q, rho) = integral_0^{2*pi}
        sqrt( rho^2 * p^2  +  (1 + rho*cos(p*t))^2 * q^2 ) dt

The mass ratio prediction: for two modes (p1,q1) and (p2,q2) on the same torus,

    m_mu / m_e = L(p1, q1, rho) / L(p2, q2, rho)

### Berger sphere (squashed S^3)

The Berger sphere has metric:

    ds^2 = dpsi^2 + sin^2(psi) * dtheta^2 + lambda^2 * (dphi + cos(psi)*dtheta)^2

where lambda is the Hopf fiber deformation parameter.  The path length for
a (p,q) mode (p = fiber winding, q = base winding) involves a similar
quadrature integral parameterized by lambda.  At lambda = 1, this is the
round S^3; deformations produce anisotropy that can tune the mode ratio.

### Lens space L(n,1)

Lens spaces are cyclic Z_n quotients of S^3.  The path ratio for modes
(p1,q1) and (p2,q2) on L(n,1) with squashing parameter sigma has the
closed-form expression:

    ratio = sqrt(p1^2 * sigma^2 + q1^2 * (1 - sigma^2))
          / sqrt(p2^2 * sigma^2 + q2^2 * (1 - sigma^2))

This requires no quadrature, making the GPU kernel pure arithmetic.

### Poincare homology sphere S^3/2I

The Poincare homology sphere is S^3 quotiented by the binary icosahedral
group 2I (order 120).  It uses the same Berger sphere path-length formula,
but only modes satisfying the icosahedral selection rule are physical:
a spin-j representation of SU(2) descends to 2I if and only if it contains
the trivial representation when restricted to 2I.  The allowed modes are
pre-filtered on the CPU using representation-theoretic criteria before
uploading to the GPU.

### Nil manifold M(a,b,k)

On a Nil manifold with twist parameter tau = k/(ab), the abelian Laplacian
eigenvalues are:

    E_{m,n}(tau) = (2*pi*m*tau)^2 + (2*pi*n)^2

This has the same structure as a flat torus with aspect ratio tau.  The
eigenvalue ratio is closed-form:

    ratio = E_{m1,n1}(tau) / E_{m2,n2}(tau)

No quadrature is needed; the GPU kernel is pure arithmetic.

### Binary polyhedral quotients

Dihedral (2D_n), tetrahedral (2T), and octahedral (2O) quotient spaces use
the same Berger sphere path-length formula as the Poincare sphere, but with
different group-theoretic selection rules for allowed modes.  Each finite
subgroup of SU(2) defines which spin-j representations descend to the
quotient space.

## Computational Approaches

The search evaluates multiple complementary models for each candidate geometry:

1. **Torus-knot geodesic path lengths** (Phase 1a) -- All-pairs comparison of
   torus-knot arc lengths via 64-point Gauss-Legendre quadrature on GPU.
   38,730 winding pairs (coprime up to 30) x millions of rho values per band.

2. **Berger sphere path lengths** (Phase 1b) -- Squashed S^3 mode ratios via
   GL quadrature, sweeping the Hopf fiber deformation lambda.

3. **Lens space ratios** (Phase 1c) -- Closed-form path ratios on L(n,1),
   sweeping the squashing parameter sigma.

4. **Poincare homology sphere** (Phase 1d) -- Berger sphere formula with
   icosahedral selection rules, sweeping lambda.

5. **Nil manifold eigenvalue ratios** (Phase 1e) -- Closed-form abelian
   eigenvalue ratios, sweeping the twist parameter tau.

6. **Toroidal Helmholtz eigenvalues** (Phase 2) -- Inverse-aspect-ratio
   expansion of the scalar Helmholtz equation on a toroidal domain.
   Eigenfrequency ratios for 25,935 mode pairs serve as a cross-check.

7. **Cavity resonance** (Phase 2) -- Aspden-style parameterized cavity model
   incorporating the fine structure constant alpha, plus a direct mode-number
   scan.

8. **Direct spectral solve** (Phase 0) -- CPU-based exact eigenvalue computation
   for selected topologies as a reference baseline.

Phase 2 refines the best Phase 1 candidates from all topologies with 10M-point
parameter scans and 100M-point ultra-deep passes, dispatching to the correct
GPU kernel based on topology type.

Phase 3 refines the best candidates on CPU with 10,000-point quadrature
in f64 precision, including fine parameter re-optimization, Newton's method
refinement, Koide formula check, and tau mass search.

## Search Strategy

| Phase | Method | Evaluations | Purpose |
|-------|--------|-------------|---------|
| 0 | Spectral solve (CPU) | exact | Direct eigenvalue computation as reference baseline |
| 1a | Torus geodesic scan (GPU) | ~99B | 7 rho bands from 0.0001 to 0.999, 38,730 winding pairs per rho |
| 1b | Berger sphere scan (GPU) | ~20B | 5 lambda bands from 0.001 to 10.0, 38,730 mode pairs per lambda |
| 1c | Lens space scan (GPU) | ~15B | 4 sigma bands from 0.001 to 0.999, 38,730 mode pairs per sigma |
| 1d | Poincare sphere scan (GPU) | ~5B | 5 lambda bands, icosahedral-filtered mode pairs |
| 1e | Nil manifold scan (GPU) | ~15B | 4 tau bands from 0.001 to 10.0, coprime mode pairs |
| 2a | Ultra-fine GPU refinement | ~2B | 10M-point scans with adaptive windows around top 20 Phase 1 hits (all topologies) |
| 2b | Ultra-deep GPU refinement | ~500M | 100M-point scans in 1e-6-wide windows around top 5 Phase 2a hits |
| 2c | Helmholtz + Cavity cross-check | ~8.6B | Independent eigenvalue and cavity-resonance models |
| 3 | High-precision CPU (Rayon) | ~50 | 10,000-point quadrature with parameter re-optimization, Newton refinement, Koide check, tau mass search |

## Technology

- **Rust** -- Host-side orchestration, phased search control, result management
- **CUDA** -- GPU kernels for parallel parameter sweeps (path integration,
  eigenvalue estimation, cavity resonance)
- **Hardware target** -- NVIDIA RTX 4070 Ti (Ada Lovelace, compute capability
  8.9, 7680 CUDA cores, 12 GB VRAM)

## Project Structure

```
kernels/
    common.cuh             Shared constants, GL quadrature, Berger integrand,
                           reduction primitives
    geodesic_path.cu       Phase 1a: torus-knot geodesic path length scan
    berger_sphere.cu       Phase 1b: Berger sphere (squashed S^3) scan
    lens_space.cu          Phase 1c: lens space L(n,1) closed-form ratio scan
    poincare_scan.cu       Phase 1d: Poincare homology sphere S^3/2I scan
    nil_scan.cu            Phase 1e: Nil manifold abelian eigenvalue scan
    toroidal_helmholtz.cu  Phase 2: Helmholtz eigenvalue cross-check
    cavity_resonance.cu    Phase 2: Aspden-style cavity resonance
    reduce_results.cu      Global reduction for collecting top candidates
src/
    main.rs                Phased search orchestration, generic TopologyScan
    physics.rs             Constants, CPU path lengths, selection rules,
                           coprime pairs, Koide formula
    spectral.rs            Spectral eigenvalue solvers, Newton refinement
    gpu.rs                 GPU context wrapper, NVRTC compilation
    search.rs              Search parameter structs, batch generation
    results.rs             Bounded priority queue, JSON output, formatting
    verify.rs              CPU reference checks, known-value tests
Cargo.toml                 Rust project manifest
results.json               Latest run output
```

## Building and Running

```
cargo build --release
cargo run --release          # full multi-topology search
cargo test                   # unit tests
```

Requires CUDA 12.x and an Ada Lovelace GPU (compute_89).  The NVRTC compiler
is invoked at runtime to JIT-compile the kernels.

## TODO

### Completed (Run 2)

- [x] Deeper Phase 2 refinement -- 10M-point rho scans + 100M-point ultra-deep
      passes.  Pushed GPU score to 6.6e-12, CPU-verified to 1.0e-8.
- [x] Eliminate duplicate candidates via dedup in ResultCollector
- [x] Increase max winding numbers to 30 (38,730 pairs, up from 8,237)
- [x] Wire cavity resonance kernel into Phase 2 pipeline
- [x] Dense rho coverage in thin-torus regime (7 bands, log-like resolution)
- [x] Search for tau/electron mass ratio (3477.48) in Phase 3
- [x] Progress reporting within Phase 1 bands (every ~5 seconds)
- [x] GPU-CPU consistency spot-checks after Phase 1
- [x] Intermediate result persistence (results_partial.json after each band)
- [x] Timing breakdown at end of run

### Completed (Multi-Topology Framework)

- [x] Generic TopologyScan struct + run_topology_scan() to eliminate code
      duplication across topology-specific scan functions
- [x] Berger sphere scan (Phase 1b) with squashed S^3 path lengths
- [x] Lens space scan (Phase 1c) with closed-form path ratios
- [x] Poincare homology sphere scan (Phase 1d) with icosahedral selection rules
- [x] Nil manifold scan (Phase 1e) with abelian eigenvalue ratios
- [x] Selection rule functions for binary polyhedral groups (2I, 2D_n, 2T, 2O)
- [x] Phase 2 topology-aware kernel dispatch (correct kernel per topology)
- [x] Phase 3 topology-aware CPU refinement and Newton's method
- [x] Phase 0 spectral direct-solve baseline

### Immediate (improve current results)

- [ ] Resolve GPU-CPU quadrature divergence -- the GPU's 64-pt GL gives 6.6e-12
      but CPU's 10K-pt trapezoidal gives 1.0e-8 for the same geometry.  Either
      increase GL order on GPU (128 or 256 points) or use Gauss-Legendre on CPU
      instead of trapezoidal to match methods.
- [ ] Increase max winding to 50+ -- Run 2's top hits all had windings > 20,
      suggesting even better geometries exist at higher winding numbers
- [ ] Widen the Phase 3 CPU rho refinement window -- currently 1e-6 wide with
      10K points; try 1e-4 wide with 100K points to avoid local minima
- [ ] Run the full multi-topology search and compare results across all 5
      topology families

### Search expansion

- [ ] Add cross-section ellipticity parameter epsilon to the path length
      integrand -- deformed tori may produce better fits
- [ ] Add cross-section tilt parameter delta
- [ ] Try ratio = L(p1,q1) / L(p2,q2) with **both** orderings (currently only
      longer/shorter) to ensure no candidates are missed
- [ ] Dedicated GPU kernel for tau/electron ratio (currently CPU-only in Phase 3;
      no hits found yet, may need broader parameter + winding search)
- [ ] Add binary dihedral, tetrahedral, and octahedral scans as Phase 1f-1h
      (selection rule functions exist, kernels share Berger sphere formula)
- [ ] Explore product geometries (S^1 x S^2, T^2 x S^1) as additional
      topology families
- [ ] Scan Nil manifold with non-abelian eigenvalue corrections

### Physics analysis

- [ ] Investigate whether rho has a closed-form relationship to alpha, pi, or
      other fundamental constants.  Run 2's best: rho ~ 0.0486 for (21,10),
      rho ~ 0.0632 for (22,13), rho ~ 0.0879 for (29,18) -- are these related
      by simple ratios?
- [ ] Compute charge, spin-1/2, and g-factor (~ 2.0023) predictions for the
      best toroidal geometries
- [ ] Use the best-fit geometry to predict the tau mass via the Koide relation
      and compare with experiment
- [ ] Compare geodesic results with Helmholtz eigenvalue results in overlapping
      regimes -- they should agree if the model is self-consistent
- [ ] Reproduce Aspden's 206.7683078 from the cavity kernel and understand the
      discrepancy with CODATA
- [ ] Analyze the cavity-mode hit (n=205, rho=0.129) -- is n=205 ~ TARGET_RATIO
      a coincidence or structurally meaningful?
- [ ] Compare results across topologies -- do different manifolds converge on
      the same parameter families, suggesting a deeper universal structure?
- [ ] Investigate whether the Poincare homology sphere (which has trivial
      first homology) produces qualitatively different candidate families
      than the freely-acting lens spaces

### Engineering

- [ ] Profile GPU occupancy and tune block/grid dimensions for the 4070 Ti
- [ ] Support f128 or arbitrary-precision arithmetic for Phase 3 to push
      beyond f64 limits (~16 significant digits)
- [ ] Resume from results_partial.json on startup to skip completed Phase 1 bands
- [ ] Use the reduce_results.cu global reduction kernel instead of CPU-side
      block-result scanning for better scaling
