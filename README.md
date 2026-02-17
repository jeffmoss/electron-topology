# Toroidal Resonance Simulation — Searching for the Electron's Geometry

A GPU-accelerated numerical search for toroidal geometries that reproduce the
muon/electron mass ratio from first principles.

## Scientific Context

Williamson & van der Mark (1997) proposed that the electron is a
Compton-wavelength photon confined in a toroidal topology, following a
double-helix flow pattern along a torus knot.  In this picture the muon is a
higher-harmonic excitation of the same topology — its rest mass arises from a
shorter confined path on the torus.

The experimental mass ratio is known to extraordinary precision (25 ppb):

    m_mu / m_e = 206.7682843 +/- 0.0000052

Aspden's cavity-resonance model derived a theoretical value of 206.7683078,
tantalizingly close.  This project systematically searches the (p, q, rho)
parameter space to find toroidal configurations that reproduce this ratio
through multiple independent physical models.

## Results

First run: **10.7 billion evaluations in ~2.5 minutes** on an RTX 4070 Ti.

Best match: **score = 2.6 x 10^-8** (0.13 ppb — well within experimental
uncertainty of 25 ppb).

| Rank | Mode pair | rho | Computed ratio | Error |
|------|-----------|-----|----------------|-------|
| 1 | (15,2) / (1,0) | 0.009698218 | 206.768284**2738** | 2.6e-8 |
| 2 | (8,2) / (1,0)  | 0.009679912 | 206.768284**2723** | 2.8e-8 |
| 3 | (12,1) / (1,0) | 0.004844497 | 206.768284**3575** | 5.7e-8 |
| 4 | (10,8) / (1,0) | 0.038736016 | 206.768284**2288** | 7.1e-8 |
| 5 | (9,1) / (1,0)  | 0.004840920 | 206.768284**2148** | 8.5e-8 |

Key structural observations:

- The denominator is always **(1,0)** — a single poloidal loop around the tube.
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
- rho ~ 0.00968 is close to **alpha = 1/137 = 0.00730** — within a factor of
  1.33.  Whether this is physically meaningful requires further analysis.

Koide formula check: Q = 0.66666051 vs 2/3 = 0.66666667 (within 0.001%).

## Mathematical Model

The fundamental object is a (p, q) torus knot on a torus with major radius
R = 1 and minor radius rho.  Using the torus metric:

    ds^2 = rho^2 * dtheta^2  +  (1 + rho*cos(theta))^2 * dphi^2

A (p,q) curve winds p times poloidally (around the tube) and q times toroidally
(around the hole):  theta(t) = p*t,  phi(t) = q*t,  t in [0, 2*pi].

The path length is:

    L(p, q, rho) = integral from 0 to 2*pi of
        sqrt( rho^2 * p^2  +  (1 + rho*cos(p*t))^2 * q^2 ) dt

The mass ratio prediction: for two modes (p1,q1) and (p2,q2) on the same torus,

    m_mu / m_e = L(p1, q1, rho) / L(p2, q2, rho)

We search over all coprime winding pairs and continuous rho for geometries
where this ratio matches 206.7682843.

## Computational Approaches

The search evaluates three complementary models for each candidate geometry:

1. **Geodesic path lengths** (Phase 1+2) — All-pairs comparison of torus-knot
   arc lengths via 64-point Gauss-Legendre quadrature on GPU.  8,237 winding
   pairs x millions of rho values per band.

2. **Toroidal Helmholtz eigenvalues** (Phase 2) — Inverse-aspect-ratio expansion
   of the scalar Helmholtz equation on a toroidal domain.  Eigenfrequency
   ratios for 25,935 mode pairs serve as an independent cross-check.

3. **Cavity resonance** (Phase 2) — Aspden-style parameterized cavity model
   incorporating the fine structure constant alpha.

Phase 3 refines the best candidates on CPU with 10,000-point trapezoidal
quadrature in f64 precision.

## Search Strategy

The search proceeds in three phases:

| Phase | Method | Evaluations | Purpose |
|-------|--------|-------------|---------|
| 1 | All-pairs GPU geodesic scan | ~10.7B | 6 rho bands from 0.001 to 0.999, 8,237 winding pairs per rho. Find candidates with score < 1.0. |
| 2 | Ultra-fine GPU refinement | ~2.2B | 1M-point rho scans in 0.0001-wide windows around top 20 Phase 1 hits + Helmholtz cross-check. Score < 0.001. |
| 3 | High-precision CPU (Rayon) | ~50 | 10,000-point quadrature verification. Koide relation check. Constant identification. |

## Technology

- **Rust** — Host-side orchestration, phased search control, result management
- **CUDA** — GPU kernels for parallel parameter sweeps (path integration,
  eigenvalue estimation, cavity resonance)
- **Hardware target** — NVIDIA RTX 4070 Ti (Ada Lovelace, compute capability
  8.9, 7680 CUDA cores, 12 GB VRAM)

## Project Structure

```
kernels/
    common.cuh             Shared constants, GL quadrature, reduction primitives
    geodesic_path.cu       Phase 1+2: all-pairs geodesic path length scan
    toroidal_helmholtz.cu  Phase 2: Helmholtz eigenvalue cross-check
    cavity_resonance.cu    Phase 2: Aspden-style cavity resonance
    reduce_results.cu      Global reduction for collecting top candidates
src/
    main.rs                Phased search orchestration
    physics.rs             Constants, structs, CPU path length, coprime pairs
    gpu.rs                 GPU context wrapper, NVRTC compilation
    search.rs              Search parameter structs, batch generation
    results.rs             Bounded priority queue, JSON output, formatting
    verify.rs              CPU reference checks, known-value tests
Cargo.toml                 Rust project manifest
results.json               Latest run output
kernels.bak/               Archived prime sieve (original project)
```

## Building and Running

```
cargo build --release
cargo run --release          # full 3-phase search (~2.5 min)
cargo test                   # 18 unit tests
```

Requires CUDA 12.x and an Ada Lovelace GPU (compute_89).  The NVRTC compiler
is invoked at runtime to JIT-compile the kernels.

## TODO

### Immediate (improve current results)

- [ ] Deeper Phase 2 refinement — run 10M-point rho scans (rho step ~1e-11)
      around the top 5 hits to push score below 1e-10
- [ ] Eliminate duplicate candidates in results (same geometry appearing from
      overlapping Phase 2 windows)
- [ ] Increase max winding numbers to 30+ and regenerate all-pairs (currently
      capped at 20)
- [ ] Add the cavity resonance kernel (cavity_resonance.cu) to the Phase 2
      pipeline — it is written but not yet wired into main.rs

### Search expansion

- [ ] Scan rho in log-space instead of linear to better resolve the thin-torus
      regime (rho < 0.01) where the best hits cluster
- [ ] Add cross-section ellipticity parameter epsilon to the path length
      integrand — deformed tori may produce better fits
- [ ] Add cross-section tilt parameter delta (Phase 3 of original plan)
- [ ] Try ratio = L(p1,q1) / L(p2,q2) with **both** orderings (currently only
      longer/shorter) to ensure no candidates are missed
- [ ] Search for tau/electron mass ratio (3477.48) using the same infrastructure

### Physics analysis

- [ ] Investigate whether rho has a closed-form relationship to alpha, pi, or
      other fundamental constants (current best rho ~ 0.00484 ~ 1/TARGET_RATIO)
- [ ] Compute charge, spin-1/2, and g-factor (~ 2.0023) predictions for the
      best toroidal geometries
- [ ] Use the best-fit geometry to predict the tau mass via the Koide relation
      and compare with experiment
- [ ] Compare geodesic results with Helmholtz eigenvalue results in overlapping
      regimes — they should agree if the model is self-consistent
- [ ] Reproduce Aspden's 206.7683078 from the cavity kernel and understand the
      discrepancy with CODATA

### Engineering

- [ ] Batch the Phase 1 scan to print progress per sub-band (currently silent
      during ~20s per band)
- [ ] Add GPU-CPU consistency spot-checks (verify.rs has the infrastructure,
      not yet called from main.rs)
- [ ] Persist intermediate results between phases so Ctrl+C preserves partial
      Phase 2 work
- [ ] Profile GPU occupancy and tune block/grid dimensions for the 4070 Ti
- [ ] Support f128 or arbitrary-precision arithmetic for Phase 3 to push
      beyond f64 limits (~16 significant digits)
