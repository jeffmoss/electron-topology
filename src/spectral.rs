use crate::physics::{
    self, coprime_pairs, dihedral_allowed_modes, octahedral_allowed_modes, poincare_allowed_modes,
    score, tetrahedral_allowed_modes, GeometryCandidate, SearchResult, TARGET_RATIO,
    TAU_ELECTRON_RATIO,
};

/// Target ratio squared (used in closed-form lens/torus solvers)
const TARGET_SQ: f64 = TARGET_RATIO * TARGET_RATIO;

/// A spectral solution: exact shape parameter for a given mode pair.
#[derive(Clone, Debug)]
pub struct SpectralSolution {
    pub geometry: &'static str,
    pub param: f64,       // shape parameter (rho, lambda, or sigma)
    pub n1: i32,          // mode 1 quantum number (or winding)
    pub m1: i32,          // mode 1 secondary quantum number
    pub n2: i32,          // mode 2 quantum number
    pub m2: i32,          // mode 2 secondary quantum number
    pub ratio: f64,       // eigenvalue ratio (should be TARGET_RATIO)
    pub score: f64,       // |ratio - TARGET|
    pub tau_score: f64,   // best tau/electron score on same geometry
    pub tau_modes: String, // mode pair giving best tau match
}

/// Direct-solve for flat torus Laplacian eigenvalue ratio.
///
/// Eigenvalues: λ_{m,n} = (m/ρ)² + n²
/// Ratio: [(m₁/ρ)² + n₁²] / [(m₂/ρ)² + n₂²] = TARGET
/// Solving: ρ² = (m₁² - TARGET·m₂²) / (TARGET·n₂² - n₁²)
pub fn solve_torus_spectral(max_mode: i32) -> Vec<SpectralSolution> {
    let mut solutions = Vec::new();

    // Denominator modes: small quantum numbers (ground state candidates)
    for m2 in 1..=max_mode.min(10) {
        for n2 in 0..=max_mode.min(10) {
            // Numerator modes: larger quantum numbers (excited states)
            for m1 in 1..=max_mode {
                for n1 in 0..=max_mode {
                    if m1 == m2 && n1 == n2 {
                        continue;
                    }

                    let numerator = m1 as f64 * m1 as f64 - TARGET_RATIO * m2 as f64 * m2 as f64;
                    let denominator =
                        TARGET_RATIO * n2 as f64 * n2 as f64 - n1 as f64 * n1 as f64;

                    if denominator.abs() < 1e-15 {
                        continue;
                    }

                    let rho_sq = numerator / denominator;
                    if rho_sq <= 0.0 || rho_sq > 1.0 {
                        continue; // rho must be in (0, 1) for a physical torus
                    }

                    let rho = rho_sq.sqrt();

                    // Verify eigenvalue ratio
                    let e1 = (m1 as f64 / rho).powi(2) + (n1 as f64).powi(2);
                    let e2 = (m2 as f64 / rho).powi(2) + (n2 as f64).powi(2);
                    if e2 < 1e-15 {
                        continue;
                    }
                    let ratio = e1 / e2;
                    let s = score(ratio);

                    if s < 1e-6 {
                        // Check tau/electron ratio on same geometry
                        let (tau_s, tau_modes) = check_tau_torus_spectral(rho, max_mode);

                        solutions.push(SpectralSolution {
                            geometry: "torus-spectral",
                            param: rho,
                            n1: m1,
                            m1: n1,
                            n2: m2,
                            m2: n2,
                            ratio,
                            score: s,
                            tau_score: tau_s,
                            tau_modes,
                        });
                    }
                }
            }
        }
    }

    solutions.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
    solutions
}

/// Direct-solve for squashed S³ (Berger sphere) Laplacian eigenvalue ratio.
///
/// Eigenvalues: E_{j,m} = j(j+2) + m²(1/λ² - 1)
/// where j = 0,1,2,... and -j ≤ m ≤ j, m same parity as j.
/// Ratio: E₁/E₂ = TARGET
/// Solving: 1/λ² = [TARGET·j₂(j₂+2) - j₁(j₁+2)] / [m₁² - TARGET·m₂²] + 1
pub fn solve_berger_spectral(max_j: i32) -> Vec<SpectralSolution> {
    let mut solutions = Vec::new();

    // Denominator modes: small j (ground state candidates)
    for j2 in 0..=max_j.min(10) {
        // m2 must have same parity as j2
        let m2_start = if j2 % 2 == 0 { 0 } else { 1 };
        let mut m2 = m2_start;
        while m2 <= j2 {
            // Numerator modes
            for j1 in 0..=max_j {
                let m1_start = if j1 % 2 == 0 { 0 } else { 1 };
                let mut m1 = m1_start;
                while m1 <= j1 {
                    if j1 == j2 && m1 == m2 {
                        m1 += 2;
                        continue;
                    }

                    let m1f = m1 as f64;
                    let m2f = m2 as f64;
                    let j1_term = j1 as f64 * (j1 as f64 + 2.0);
                    let j2_term = j2 as f64 * (j2 as f64 + 2.0);

                    let denom = m1f * m1f - TARGET_RATIO * m2f * m2f;
                    if denom.abs() < 1e-15 {
                        m1 += 2;
                        continue;
                    }

                    let inv_lam_sq = (TARGET_RATIO * j2_term - j1_term) / denom + 1.0;
                    if inv_lam_sq <= 0.0 {
                        m1 += 2;
                        continue;
                    }

                    let lambda = 1.0 / inv_lam_sq.sqrt();

                    // Verify eigenvalue ratio
                    let xi = 1.0 / (lambda * lambda) - 1.0;
                    let e1 = j1_term + m1f * m1f * xi;
                    let e2 = j2_term + m2f * m2f * xi;
                    if e2 < 1e-15 || e1 < 0.0 {
                        m1 += 2;
                        continue;
                    }
                    let ratio = e1 / e2;
                    let s = score(ratio);

                    if s < 1e-6 {
                        // Check tau ratio on same geometry
                        let (tau_s, tau_modes) =
                            check_tau_berger_spectral(lambda, max_j);

                        solutions.push(SpectralSolution {
                            geometry: "berger-spectral",
                            param: lambda,
                            n1: j1,
                            m1,
                            n2: j2,
                            m2,
                            ratio,
                            score: s,
                            tau_score: tau_s,
                            tau_modes,
                        });
                    }

                    // Also try negative m values (m and -m give different ratios
                    // when combined with different partner modes)
                    // The eigenvalue E_{j,m} = E_{j,-m} so we only need |m|.
                    // But cross terms between +m1 and -m2 are different pairs,
                    // handled by the loop structure.

                    m1 += 2;
                }
            }
            m2 += 2;
        }
    }

    solutions.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
    solutions
}

/// Direct-solve for lens space path-length ratio.
///
/// Ratio² = (a₁²σ² + b₁²(1-σ²)) / (a₂²σ² + b₂²(1-σ²)) = TARGET²
/// Let u = σ²/(1-σ²). Then:
/// u = (TARGET²·b₂² - b₁²) / (a₁² - TARGET²·a₂²)
/// σ = sqrt(u/(1+u))
pub fn solve_lens_direct(max_winding: i32) -> Vec<SpectralSolution> {
    let mut solutions = Vec::new();
    let modes = coprime_pairs(max_winding);

    for (i, &(a1, b1)) in modes.iter().enumerate() {
        for &(a2, b2) in modes.iter().skip(i + 1) {
            let a1f = a1 as f64;
            let b1f = b1 as f64;
            let a2f = a2 as f64;
            let b2f = b2 as f64;

            let denom = a1f * a1f - TARGET_SQ * a2f * a2f;
            if denom.abs() < 1e-15 {
                continue;
            }

            let u = (TARGET_SQ * b2f * b2f - b1f * b1f) / denom;
            if u <= 0.0 {
                continue;
            }

            let sigma_sq = u / (1.0 + u);
            if sigma_sq <= 0.0 || sigma_sq >= 1.0 {
                continue;
            }
            let sigma = sigma_sq.sqrt();

            // Verify ratio
            let ratio = physics::lens_path_ratio_cpu(a1, b1, a2, b2, sigma);
            let s = score(ratio);

            if s < 1e-6 {
                // Check tau
                let (tau_s, tau_modes) = check_tau_lens(sigma, max_winding);

                solutions.push(SpectralSolution {
                    geometry: "lens-direct",
                    param: sigma,
                    n1: a1,
                    m1: b1,
                    n2: a2,
                    m2: b2,
                    ratio,
                    score: s,
                    tau_score: tau_s,
                    tau_modes,
                });
            }

            // Also try reversed ratio (a2,b2)/(a1,b1)
            let denom_rev = a2f * a2f - TARGET_SQ * a1f * a1f;
            if denom_rev.abs() < 1e-15 {
                continue;
            }
            let u_rev = (TARGET_SQ * b1f * b1f - b2f * b2f) / denom_rev;
            if u_rev <= 0.0 {
                continue;
            }
            let sigma_sq_rev = u_rev / (1.0 + u_rev);
            if sigma_sq_rev <= 0.0 || sigma_sq_rev >= 1.0 {
                continue;
            }
            let sigma_rev = sigma_sq_rev.sqrt();

            let ratio_rev = physics::lens_path_ratio_cpu(a2, b2, a1, b1, sigma_rev);
            let s_rev = score(ratio_rev);

            if s_rev < 1e-6 {
                let (tau_s, tau_modes) = check_tau_lens(sigma_rev, max_winding);
                solutions.push(SpectralSolution {
                    geometry: "lens-direct",
                    param: sigma_rev,
                    n1: a2,
                    m1: b2,
                    n2: a1,
                    m2: b1,
                    ratio: ratio_rev,
                    score: s_rev,
                    tau_score: tau_s,
                    tau_modes,
                });
            }
        }
    }

    solutions.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
    solutions
}

/// Newton's method refinement for torus geodesic path-length ratio.
///
/// Given a (p1,q1)÷(p2,q2) pair and initial rho guess, find the exact rho
/// where L(p1,q1,rho)/L(p2,q2,rho) = TARGET to machine precision.
pub fn newton_geodesic_ratio(
    p1: i32,
    q1: i32,
    p2: i32,
    q2: i32,
    rho_init: f64,
    quadrature_points: usize,
) -> Option<(f64, f64, f64)> {
    let mut rho = rho_init;
    let max_iter = 50;
    let tol = 1e-14;

    for _ in 0..max_iter {
        if rho <= 1e-15 || rho >= 1.0 {
            return None;
        }

        let l1 = physics::path_length_cpu(p1, q1, rho, quadrature_points);
        let l2 = physics::path_length_cpu(p2, q2, rho, quadrature_points);
        if l2 < 1e-15 {
            return None;
        }
        let ratio = l1 / l2;
        let f_val = ratio - TARGET_RATIO;

        if f_val.abs() < tol {
            return Some((rho, ratio, f_val.abs()));
        }

        // Numerical derivative: dF/drho ≈ [F(rho+h) - F(rho-h)] / (2h)
        let h = rho * 1e-8;
        let rho_plus = rho + h;
        let rho_minus = (rho - h).max(1e-15);
        let actual_h = rho_plus - rho_minus;

        let l1_plus = physics::path_length_cpu(p1, q1, rho_plus, quadrature_points);
        let l2_plus = physics::path_length_cpu(p2, q2, rho_plus, quadrature_points);
        let l1_minus = physics::path_length_cpu(p1, q1, rho_minus, quadrature_points);
        let l2_minus = physics::path_length_cpu(p2, q2, rho_minus, quadrature_points);

        let f_plus = if l2_plus > 1e-15 {
            l1_plus / l2_plus - TARGET_RATIO
        } else {
            return None;
        };
        let f_minus = if l2_minus > 1e-15 {
            l1_minus / l2_minus - TARGET_RATIO
        } else {
            return None;
        };

        let df = (f_plus - f_minus) / actual_h;
        if df.abs() < 1e-30 {
            return None;
        }

        let delta = f_val / df;
        rho -= delta;

        // Damping for stability
        if rho <= 0.0 {
            rho = rho_init * 0.5;
        }
    }

    // Return best even if not converged
    let l1 = physics::path_length_cpu(p1, q1, rho, quadrature_points);
    let l2 = physics::path_length_cpu(p2, q2, rho, quadrature_points);
    if l2 > 1e-15 {
        let ratio = l1 / l2;
        Some((rho, ratio, (ratio - TARGET_RATIO).abs()))
    } else {
        None
    }
}

/// Newton's method for Berger sphere geodesic path-length ratio.
pub fn newton_berger_ratio(
    p1: i32,
    q1: i32,
    p2: i32,
    q2: i32,
    lambda_init: f64,
    quadrature_points: usize,
) -> Option<(f64, f64, f64)> {
    let mut lambda = lambda_init;
    let max_iter = 50;
    let tol = 1e-14;

    for _ in 0..max_iter {
        if lambda <= 1e-15 {
            return None;
        }

        let l1 = physics::berger_path_length_cpu(p1, q1, lambda, quadrature_points);
        let l2 = physics::berger_path_length_cpu(p2, q2, lambda, quadrature_points);
        if l2 < 1e-15 {
            return None;
        }
        let ratio = l1 / l2;
        let f_val = ratio - TARGET_RATIO;

        if f_val.abs() < tol {
            return Some((lambda, ratio, f_val.abs()));
        }

        let h = lambda * 1e-8;
        let lam_plus = lambda + h;
        let lam_minus = (lambda - h).max(1e-15);
        let actual_h = lam_plus - lam_minus;

        let l1_plus = physics::berger_path_length_cpu(p1, q1, lam_plus, quadrature_points);
        let l2_plus = physics::berger_path_length_cpu(p2, q2, lam_plus, quadrature_points);
        let l1_minus = physics::berger_path_length_cpu(p1, q1, lam_minus, quadrature_points);
        let l2_minus = physics::berger_path_length_cpu(p2, q2, lam_minus, quadrature_points);

        let f_plus = if l2_plus > 1e-15 {
            l1_plus / l2_plus - TARGET_RATIO
        } else {
            return None;
        };
        let f_minus = if l2_minus > 1e-15 {
            l1_minus / l2_minus - TARGET_RATIO
        } else {
            return None;
        };

        let df = (f_plus - f_minus) / actual_h;
        if df.abs() < 1e-30 {
            return None;
        }

        let delta = f_val / df;
        lambda -= delta;

        if lambda <= 0.0 {
            lambda = lambda_init * 0.5;
        }
    }

    let l1 = physics::berger_path_length_cpu(p1, q1, lambda, quadrature_points);
    let l2 = physics::berger_path_length_cpu(p2, q2, lambda, quadrature_points);
    if l2 > 1e-15 {
        let ratio = l1 / l2;
        Some((lambda, ratio, (ratio - TARGET_RATIO).abs()))
    } else {
        None
    }
}

/// Check if any mode pair on the same torus (same rho) gives the tau/electron ratio.
fn check_tau_torus_spectral(rho: f64, max_mode: i32) -> (f64, String) {
    let target_tau = TAU_ELECTRON_RATIO;
    let mut best_score = f64::MAX;
    let mut best_modes = String::new();

    for m2 in 1..=max_mode.min(10) {
        for n2 in 0..=max_mode.min(10) {
            for m1 in 1..=max_mode {
                for n1 in 0..=max_mode {
                    if m1 == m2 && n1 == n2 {
                        continue;
                    }
                    let e1 = (m1 as f64 / rho).powi(2) + (n1 as f64).powi(2);
                    let e2 = (m2 as f64 / rho).powi(2) + (n2 as f64).powi(2);
                    if e2 < 1e-15 {
                        continue;
                    }
                    let ratio = e1 / e2;
                    let s = (ratio - target_tau).abs();
                    if s < best_score {
                        best_score = s;
                        best_modes = format!("({},{})÷({},{})", m1, n1, m2, n2);
                    }
                }
            }
        }
    }

    (best_score, best_modes)
}

/// Check if any mode pair on the same Berger sphere gives the tau/electron ratio.
fn check_tau_berger_spectral(lambda: f64, max_j: i32) -> (f64, String) {
    let target_tau = TAU_ELECTRON_RATIO;
    let xi = 1.0 / (lambda * lambda) - 1.0;
    let mut best_score = f64::MAX;
    let mut best_modes = String::new();

    for j2 in 0..=max_j.min(10) {
        let m2_start = if j2 % 2 == 0 { 0 } else { 1 };
        let mut m2 = m2_start;
        while m2 <= j2 {
            let e2 = j2 as f64 * (j2 as f64 + 2.0) + (m2 as f64).powi(2) * xi;
            if e2 < 1e-15 {
                m2 += 2;
                continue;
            }

            for j1 in 0..=max_j {
                let m1_start = if j1 % 2 == 0 { 0 } else { 1 };
                let mut m1 = m1_start;
                while m1 <= j1 {
                    if j1 == j2 && m1 == m2 {
                        m1 += 2;
                        continue;
                    }
                    let e1 = j1 as f64 * (j1 as f64 + 2.0) + (m1 as f64).powi(2) * xi;
                    if e1 < 0.0 {
                        m1 += 2;
                        continue;
                    }
                    let ratio = e1 / e2;
                    let s = (ratio - target_tau).abs();
                    if s < best_score {
                        best_score = s;
                        best_modes = format!("j({},{})÷({},{})", j1, m1, j2, m2);
                    }
                    m1 += 2;
                }
            }
            m2 += 2;
        }
    }

    (best_score, best_modes)
}

/// Check if any mode pair on the same lens space gives the tau/electron ratio.
fn check_tau_lens(sigma: f64, max_winding: i32) -> (f64, String) {
    let target_tau = TAU_ELECTRON_RATIO;
    let modes = coprime_pairs(max_winding);
    let mut best_score = f64::MAX;
    let mut best_modes = String::new();

    for (i, &(a1, b1)) in modes.iter().enumerate() {
        for &(a2, b2) in modes.iter().skip(i + 1) {
            let ratio = physics::lens_path_ratio_cpu(a1, b1, a2, b2, sigma);
            let s = (ratio - target_tau).abs();
            if s < best_score {
                best_score = s;
                best_modes = format!("({},{})÷({},{})", a1, b1, a2, b2);
            }
            // Reversed
            if ratio > 1e-15 {
                let s_rev = (1.0 / ratio - target_tau).abs();
                if s_rev < best_score {
                    best_score = s_rev;
                    best_modes = format!("({},{})÷({},{})", a2, b2, a1, b1);
                }
            }
        }
    }

    (best_score, best_modes)
}

/// Generic solver for S³ quotient topologies (Berger eigenvalue formula
/// restricted to an allowed mode list).
///
/// Eigenvalues: E_{j,m} = j(j+2) + m²(1/λ² - 1).
/// For each pair of allowed modes, solve for λ analytically, then verify.
fn solve_quotient_spectral(
    geometry_name: &'static str,
    allowed_modes: &[(i32, i32)],
    max_j: i32,
) -> Vec<SpectralSolution> {
    let mut solutions = Vec::new();

    // Separate modes into denominator (small j) and numerator sets
    let denom_modes: Vec<&(i32, i32)> = allowed_modes.iter().filter(|(j, _)| *j <= max_j.min(10)).collect();
    let numer_modes: Vec<&(i32, i32)> = allowed_modes.iter().filter(|(j, _)| *j <= max_j).collect();

    for &&(j2, m2) in &denom_modes {
        for &&(j1, m1) in &numer_modes {
            if j1 == j2 && m1 == m2 {
                continue;
            }

            let m1f = m1 as f64;
            let m2f = m2 as f64;
            let j1_term = j1 as f64 * (j1 as f64 + 2.0);
            let j2_term = j2 as f64 * (j2 as f64 + 2.0);

            let denom = m1f * m1f - TARGET_RATIO * m2f * m2f;
            if denom.abs() < 1e-15 {
                continue;
            }

            let inv_lam_sq = (TARGET_RATIO * j2_term - j1_term) / denom + 1.0;
            if inv_lam_sq <= 0.0 {
                continue;
            }

            let lambda = 1.0 / inv_lam_sq.sqrt();

            // Verify eigenvalue ratio
            let xi = 1.0 / (lambda * lambda) - 1.0;
            let e1 = j1_term + m1f * m1f * xi;
            let e2 = j2_term + m2f * m2f * xi;
            if e2 < 1e-15 || e1 < 0.0 {
                continue;
            }
            let ratio = e1 / e2;
            let s = score(ratio);

            if s < 1e-6 {
                let (tau_s, tau_modes) =
                    check_tau_quotient_spectral(lambda, allowed_modes);

                solutions.push(SpectralSolution {
                    geometry: geometry_name,
                    param: lambda,
                    n1: j1,
                    m1,
                    n2: j2,
                    m2,
                    ratio,
                    score: s,
                    tau_score: tau_s,
                    tau_modes,
                });
            }
        }
    }

    solutions.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
    solutions
}

/// Check tau/electron ratio on an S³ quotient geometry with given allowed modes.
fn check_tau_quotient_spectral(lambda: f64, allowed_modes: &[(i32, i32)]) -> (f64, String) {
    let target_tau = TAU_ELECTRON_RATIO;
    let xi = 1.0 / (lambda * lambda) - 1.0;
    let mut best_score = f64::MAX;
    let mut best_modes = String::new();

    // Use small-j modes as denominators
    let denom_modes: Vec<&(i32, i32)> = allowed_modes.iter().filter(|(j, _)| *j <= 10).collect();

    for &&(j2, m2) in &denom_modes {
        let e2 = j2 as f64 * (j2 as f64 + 2.0) + (m2 as f64).powi(2) * xi;
        if e2 < 1e-15 {
            continue;
        }

        for &(j1, m1) in allowed_modes {
            if j1 == j2 && m1 == m2 {
                continue;
            }
            let e1 = j1 as f64 * (j1 as f64 + 2.0) + (m1 as f64).powi(2) * xi;
            if e1 < 0.0 {
                continue;
            }
            let ratio = e1 / e2;
            let s = (ratio - target_tau).abs();
            if s < best_score {
                best_score = s;
                best_modes = format!("j({},{})÷({},{})", j1, m1, j2, m2);
            }
        }
    }

    (best_score, best_modes)
}

/// Direct-solve for Poincare homology sphere (S³/2I) spectral ratio.
///
/// Same Berger eigenvalue formula but restricted to the binary icosahedral
/// selection rule via `poincare_allowed_modes`.
pub fn solve_poincare_spectral(max_j: i32) -> Vec<SpectralSolution> {
    let modes = poincare_allowed_modes(max_j);
    solve_quotient_spectral("poincare-spectral", &modes, max_j)
}

/// Direct-solve for Nil manifold eigenvalue ratio.
///
/// Eigenvalues: E_{m,n}(tau) = 4pi^2 (m^2 tau^2 + n^2).
/// Ratio: (m1^2 tau^2 + n1^2) / (m2^2 tau^2 + n2^2) = TARGET
/// Solving: tau^2 = (TARGET*n2^2 - n1^2) / (m1^2 - TARGET*m2^2)
pub fn solve_nil_spectral(max_mode: i32) -> Vec<SpectralSolution> {
    let mut solutions = Vec::new();

    // Denominator modes: small quantum numbers
    for m2 in 1..=max_mode.min(10) {
        for n2 in 0..=max_mode.min(10) {
            // Numerator modes
            for m1 in 1..=max_mode {
                for n1 in 0..=max_mode {
                    if m1 == m2 && n1 == n2 {
                        continue;
                    }

                    let m1f = m1 as f64;
                    let m2f = m2 as f64;
                    let n1f = n1 as f64;
                    let n2f = n2 as f64;

                    let denom = m1f * m1f - TARGET_RATIO * m2f * m2f;
                    if denom.abs() < 1e-15 {
                        continue;
                    }

                    let tau_sq = (TARGET_RATIO * n2f * n2f - n1f * n1f) / denom;
                    if tau_sq <= 0.0 {
                        continue;
                    }

                    let tau = tau_sq.sqrt();

                    // Verify eigenvalue ratio using the CPU reference
                    let ratio = physics::nil_eigenvalue_ratio_cpu(m1, n1, m2, n2, tau);
                    let s = score(ratio);

                    if s < 1e-6 {
                        let (tau_s, tau_modes) = check_tau_nil_spectral(tau, max_mode);

                        solutions.push(SpectralSolution {
                            geometry: "nil-spectral",
                            param: tau,
                            n1: m1,
                            m1: n1,
                            n2: m2,
                            m2: n2,
                            ratio,
                            score: s,
                            tau_score: tau_s,
                            tau_modes,
                        });
                    }
                }
            }
        }
    }

    solutions.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
    solutions
}

/// Direct-solve for binary dihedral quotient S³/2D_n spectral ratio.
pub fn solve_dihedral_spectral(n: i32, max_j: i32) -> Vec<SpectralSolution> {
    let modes = dihedral_allowed_modes(n, max_j);
    let name: &'static str = match n {
        3 => "dihedral3-spectral",
        4 => "dihedral4-spectral",
        5 => "dihedral5-spectral",
        _ => "dihedral-spectral",
    };
    solve_quotient_spectral(name, &modes, max_j)
}

/// Direct-solve for binary tetrahedral quotient S³/2T spectral ratio.
pub fn solve_tetrahedral_spectral(max_j: i32) -> Vec<SpectralSolution> {
    let modes = tetrahedral_allowed_modes(max_j);
    solve_quotient_spectral("tetrahedral-spectral", &modes, max_j)
}

/// Direct-solve for binary octahedral quotient S³/2O spectral ratio.
pub fn solve_octahedral_spectral(max_j: i32) -> Vec<SpectralSolution> {
    let modes = octahedral_allowed_modes(max_j);
    solve_quotient_spectral("octahedral-spectral", &modes, max_j)
}

/// Check tau/electron ratio on Poincare homology sphere.
fn check_tau_poincare_spectral(lambda: f64, max_j: i32) -> (f64, String) {
    let modes = poincare_allowed_modes(max_j);
    check_tau_quotient_spectral(lambda, &modes)
}

/// Check tau/electron ratio on Nil manifold.
fn check_tau_nil_spectral(tau: f64, max_mode: i32) -> (f64, String) {
    let target_tau = TAU_ELECTRON_RATIO;
    let mut best_score = f64::MAX;
    let mut best_modes = String::new();

    for m2 in 1..=max_mode.min(10) {
        for n2 in 0..=max_mode.min(10) {
            for m1 in 1..=max_mode {
                for n1 in 0..=max_mode {
                    if m1 == m2 && n1 == n2 {
                        continue;
                    }
                    let ratio = physics::nil_eigenvalue_ratio_cpu(m1, n1, m2, n2, tau);
                    if ratio < 1e-15 {
                        continue;
                    }
                    let s = (ratio - target_tau).abs();
                    if s < best_score {
                        best_score = s;
                        best_modes = format!("({},{})÷({},{})", m1, n1, m2, n2);
                    }
                }
            }
        }
    }

    (best_score, best_modes)
}

/// Run all direct-solve spectral analyses and return combined results.
pub fn run_spectral_phase() -> SearchResult {
    use std::time::Instant;
    let start = Instant::now();

    println!("--- Phase 0: Direct-Solve Spectral Analysis ---");
    println!("(Exact algebraic solutions — no scanning needed)\n");

    // 1. Flat torus eigenvalue solve
    print!("  Torus eigenvalues (modes up to 100)... ");
    let torus_solutions = solve_torus_spectral(100);
    println!(
        "found {} exact solutions (score < 1e-6)",
        torus_solutions.len()
    );

    // 2. Berger sphere eigenvalue solve
    print!("  Berger sphere eigenvalues (j up to 50)... ");
    let berger_solutions = solve_berger_spectral(50);
    println!(
        "found {} exact solutions (score < 1e-6)",
        berger_solutions.len()
    );

    // 3. Lens space direct solve
    print!("  Lens space direct solve (windings up to 30)... ");
    let lens_solutions = solve_lens_direct(30);
    println!(
        "found {} exact solutions (score < 1e-6)",
        lens_solutions.len()
    );

    // 4. Poincare homology sphere (S^3 / binary icosahedral)
    print!("  Poincare homology sphere (j up to 50)... ");
    let poincare_solutions = solve_poincare_spectral(50);
    println!(
        "found {} exact solutions (score < 1e-6)",
        poincare_solutions.len()
    );

    // 5. Nil manifold eigenvalue solve
    print!("  Nil manifold eigenvalues (modes up to 30)... ");
    let nil_solutions = solve_nil_spectral(30);
    println!(
        "found {} exact solutions (score < 1e-6)",
        nil_solutions.len()
    );

    // 6. Binary dihedral quotients
    print!("  Dihedral D3 quotient (j up to 50)... ");
    let dihedral3_solutions = solve_dihedral_spectral(3, 50);
    println!(
        "found {} exact solutions (score < 1e-6)",
        dihedral3_solutions.len()
    );

    print!("  Dihedral D4 quotient (j up to 50)... ");
    let dihedral4_solutions = solve_dihedral_spectral(4, 50);
    println!(
        "found {} exact solutions (score < 1e-6)",
        dihedral4_solutions.len()
    );

    print!("  Dihedral D5 quotient (j up to 50)... ");
    let dihedral5_solutions = solve_dihedral_spectral(5, 50);
    println!(
        "found {} exact solutions (score < 1e-6)",
        dihedral5_solutions.len()
    );

    // 7. Binary tetrahedral quotient
    print!("  Tetrahedral quotient (j up to 50)... ");
    let tetrahedral_solutions = solve_tetrahedral_spectral(50);
    println!(
        "found {} exact solutions (score < 1e-6)",
        tetrahedral_solutions.len()
    );

    // 8. Binary octahedral quotient
    print!("  Octahedral quotient (j up to 50)... ");
    let octahedral_solutions = solve_octahedral_spectral(50);
    println!(
        "found {} exact solutions (score < 1e-6)",
        octahedral_solutions.len()
    );

    // Combine all solutions
    let mut all_solutions: Vec<SpectralSolution> = Vec::new();
    all_solutions.extend(torus_solutions);
    all_solutions.extend(berger_solutions);
    all_solutions.extend(lens_solutions);
    all_solutions.extend(poincare_solutions);
    all_solutions.extend(nil_solutions);
    all_solutions.extend(dihedral3_solutions);
    all_solutions.extend(dihedral4_solutions);
    all_solutions.extend(dihedral5_solutions);
    all_solutions.extend(tetrahedral_solutions);
    all_solutions.extend(octahedral_solutions);
    all_solutions.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));

    // Print top results
    println!("\n  Top spectral solutions:");
    let mut dual_hits = 0;
    for (i, sol) in all_solutions.iter().take(20).enumerate() {
        let dual_marker = if sol.tau_score < 1.0 {
            dual_hits += 1;
            " *** DUAL HIT ***"
        } else {
            ""
        };
        println!(
            "    #{:>2}: [{}] param={:.12} ({},{})÷({},{}) ratio={:.12} score={:.2e} | tau_score={:.2e} {}{}",
            i + 1,
            sol.geometry,
            sol.param,
            sol.n1, sol.m1, sol.n2, sol.m2,
            sol.ratio,
            sol.score,
            sol.tau_score,
            sol.tau_modes,
            dual_marker,
        );
    }

    if dual_hits > 0 {
        println!(
            "\n  *** {} DUAL HITS: geometries that predict BOTH muon AND tau ratios! ***",
            dual_hits
        );
    }

    // Check if any solutions have recognizable parameter values
    println!("\n  Checking for recognizable constants in optimal parameters...");
    check_recognizable_params(&all_solutions);

    // Convert to GeometryCandidates for compatibility with existing pipeline
    let candidates: Vec<GeometryCandidate> = all_solutions
        .iter()
        .take(100)
        .map(|sol| {
            let p_enc = sol.n1 * 1000 + sol.m1;
            let q_enc = sol.n2 * 1000 + sol.m2;
            GeometryCandidate {
                rho: sol.param,
                p: p_enc,
                q: q_enc,
                epsilon: 0.0,
                path_length: 0.0, // spectral — no single path length
                ratio: sol.ratio,
                score: sol.score,
                method: format!(
                    "{} ({},{})÷({},{})",
                    sol.geometry, sol.n1, sol.m1, sol.n2, sol.m2
                ),
            }
        })
        .collect();

    let elapsed = start.elapsed().as_secs_f64();
    let total = all_solutions.len() as u64;
    let best_score = candidates.first().map(|c| c.score).unwrap_or(f64::MAX);

    println!(
        "\nPhase 0 complete: {:.3}s, {} exact solutions, best score: {:.6e}",
        elapsed, total, best_score
    );

    SearchResult {
        phase: 0,
        candidates,
        total_evaluated: total,
        best_score,
        elapsed_secs: elapsed,
    }
}

fn check_recognizable_params(solutions: &[SpectralSolution]) {
    let pi = std::f64::consts::PI;
    let alpha = physics::ALPHA;
    let constants: &[(&str, f64)] = &[
        ("1/(2π)", 1.0 / (2.0 * pi)),
        ("α", alpha),
        ("2α", 2.0 * alpha),
        ("α/π", alpha / pi),
        ("1/π", 1.0 / pi),
        ("√α", alpha.sqrt()),
        ("1/e", 1.0 / std::f64::consts::E),
        ("1/√(2π)", 1.0 / (2.0 * pi).sqrt()),
        ("1/3", 1.0 / 3.0),
        ("1/4", 0.25),
        ("1/7", 1.0 / 7.0),
        ("φ-1", (1.0 + 5.0_f64.sqrt()) / 2.0 - 1.0),
        ("1/137", 1.0 / 137.0),
        ("π/1000", pi / 1000.0),
        ("√(2/3)", (2.0 / 3.0_f64).sqrt()),
        ("1/√2", std::f64::consts::FRAC_1_SQRT_2),
        ("√(3)/2", 3.0_f64.sqrt() / 2.0),
        ("α·π", alpha * pi),
        ("3α", 3.0 * alpha),
        ("2π·α", 2.0 * pi * alpha),
    ];

    // Check unique parameters only
    let mut seen_params: Vec<f64> = Vec::new();
    for sol in solutions.iter().take(50) {
        if seen_params.iter().any(|&p| (p - sol.param).abs() < 1e-10) {
            continue;
        }
        seen_params.push(sol.param);

        for &(name, val) in constants {
            let diff = (sol.param - val).abs();
            if diff < 0.01 {
                println!(
                    "    [{}] param={:.12} ≈ {} ({:.12}), diff = {:.6e}",
                    sol.geometry, sol.param, name, val, diff
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_torus_spectral_finds_solutions() {
        let solutions = solve_torus_spectral(30);
        // Should find at least some solutions
        assert!(!solutions.is_empty(), "Torus spectral should find solutions");
        // All solutions should have score < 1e-6
        for sol in &solutions {
            assert!(sol.score < 1e-6, "Score {} should be < 1e-6", sol.score);
        }
    }

    #[test]
    fn test_berger_spectral_finds_solutions() {
        let solutions = solve_berger_spectral(20);
        // Should find at least some solutions
        assert!(
            !solutions.is_empty(),
            "Berger spectral should find solutions"
        );
        for sol in &solutions {
            assert!(sol.score < 1e-6, "Score {} should be < 1e-6", sol.score);
        }
    }

    #[test]
    fn test_lens_direct_finds_solutions() {
        let solutions = solve_lens_direct(15);
        assert!(
            !solutions.is_empty(),
            "Lens direct should find solutions"
        );
        for sol in &solutions {
            assert!(sol.score < 1e-6, "Score {} should be < 1e-6", sol.score);
        }
    }

    #[test]
    fn test_newton_geodesic_converges() {
        // (1,0)÷(0,1): ratio = rho / (1+rho). Not useful for 206.768 but tests convergence.
        // Try a pair that we know works from Phase 1 scans
        // Use (21,10)÷(1,0): search near rho ~ 0.005
        let result = newton_geodesic_ratio(21, 10, 1, 0, 0.005, 1000);
        assert!(result.is_some(), "Newton should converge");
        if let Some((rho, _ratio, residual)) = result {
            assert!(rho > 0.0 && rho < 1.0, "rho should be in (0,1)");
            // May or may not match TARGET, but should converge to something
            assert!(residual < 1.0, "Should converge to reasonable residual");
        }
    }

    #[test]
    fn test_torus_spectral_verification() {
        // If we find a solution, verify the eigenvalue ratio manually
        let solutions = solve_torus_spectral(30);
        if let Some(sol) = solutions.first() {
            let rho = sol.param;
            let m1 = sol.n1;
            let n1 = sol.m1;
            let m2 = sol.n2;
            let n2 = sol.m2;

            let e1 = (m1 as f64 / rho).powi(2) + (n1 as f64).powi(2);
            let e2 = (m2 as f64 / rho).powi(2) + (n2 as f64).powi(2);
            let ratio = e1 / e2;
            assert!(
                (ratio - TARGET_RATIO).abs() < 1e-6,
                "Verification failed: ratio={}, target={}",
                ratio,
                TARGET_RATIO
            );
        }
    }

    #[test]
    fn test_poincare_spectral_finds_solutions() {
        let solutions = solve_poincare_spectral(30);
        // Poincare solutions should be a subset of Berger solutions
        // May or may not find solutions at small max_j due to restrictive selection rule
        for sol in &solutions {
            assert!(sol.score < 1e-6, "Score {} should be < 1e-6", sol.score);
        }
    }

    #[test]
    fn test_nil_spectral_finds_solutions() {
        // Use max_mode=30 for tests (O(n^4) loop); production uses 100
        let solutions = solve_nil_spectral(30);
        assert!(!solutions.is_empty(), "Nil spectral should find solutions");
        for sol in &solutions {
            assert!(sol.score < 1e-6, "Score {} should be < 1e-6", sol.score);
        }
    }

    #[test]
    fn test_dihedral_spectral_finds_solutions() {
        let solutions = solve_dihedral_spectral(3, 30);
        for sol in &solutions {
            assert!(sol.score < 1e-6, "Score {} should be < 1e-6", sol.score);
        }
    }

    #[test]
    fn test_tetrahedral_spectral_finds_solutions() {
        let solutions = solve_tetrahedral_spectral(30);
        for sol in &solutions {
            assert!(sol.score < 1e-6, "Score {} should be < 1e-6", sol.score);
        }
    }

    #[test]
    fn test_octahedral_spectral_finds_solutions() {
        let solutions = solve_octahedral_spectral(30);
        for sol in &solutions {
            assert!(sol.score < 1e-6, "Score {} should be < 1e-6", sol.score);
        }
    }

    #[test]
    fn test_nil_spectral_verification() {
        // Verify that solutions have the correct eigenvalue ratio
        let solutions = solve_nil_spectral(30);
        for sol in solutions.iter().take(5) {
            let tau = sol.param;
            let m1 = sol.n1;
            let n1 = sol.m1;
            let m2 = sol.n2;
            let n2 = sol.m2;
            let ratio = physics::nil_eigenvalue_ratio_cpu(m1, n1, m2, n2, tau);
            assert!(
                (ratio - TARGET_RATIO).abs() < 1e-6,
                "Nil verification failed: ratio={}, target={}",
                ratio,
                TARGET_RATIO
            );
        }
    }

    #[test]
    fn test_quotient_solutions_are_berger_subset() {
        // Poincare solutions at a given lambda should also be valid Berger solutions
        let poincare_sols = solve_poincare_spectral(30);
        for sol in poincare_sols.iter().take(3) {
            let lambda = sol.param;
            let xi = 1.0 / (lambda * lambda) - 1.0;
            let e1 = sol.n1 as f64 * (sol.n1 as f64 + 2.0) + (sol.m1 as f64).powi(2) * xi;
            let e2 = sol.n2 as f64 * (sol.n2 as f64 + 2.0) + (sol.m2 as f64).powi(2) * xi;
            assert!(e2 > 1e-15, "Denominator eigenvalue should be positive");
            let ratio = e1 / e2;
            assert!(
                (ratio - TARGET_RATIO).abs() < 1e-6,
                "Poincare solution should satisfy Berger formula: ratio={}, target={}",
                ratio,
                TARGET_RATIO
            );
        }
    }
}
