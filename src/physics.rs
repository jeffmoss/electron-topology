use serde::Serialize;

// Fine-structure constant
pub const ALPHA: f64 = 1.0 / 137.035999084;

// Compton wavelength of the electron (meters)
pub const COMPTON_WAVELENGTH: f64 = 2.4263102175e-12;

// Muon/electron mass ratio target
pub const TARGET_RATIO: f64 = 206.7682843;
pub const TARGET_UNCERTAINTY: f64 = 0.0000052;

// Particle masses in MeV/c^2
pub const ELECTRON_MASS: f64 = 0.51099895;
pub const MUON_MASS: f64 = 105.6583755;
pub const TAU_MASS: f64 = 1776.86;

// Tau/electron mass ratio
pub const TAU_ELECTRON_RATIO: f64 = 3477.48;

// Aspden cavity resonance value
pub const ASPDEN_VALUE: f64 = 206.7683078;

#[derive(Clone, Debug, Serialize)]
pub struct GeometryCandidate {
    pub rho: f64,
    pub p: i32,
    pub q: i32,
    pub epsilon: f64,
    pub path_length: f64,
    pub ratio: f64,
    pub score: f64,
    pub method: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct SearchResult {
    pub phase: u32,
    pub candidates: Vec<GeometryCandidate>,
    pub total_evaluated: u64,
    pub best_score: f64,
    pub elapsed_secs: f64,
}

/// Score a computed ratio against the target muon/electron mass ratio.
/// Lower is better.
pub fn score(computed_ratio: f64) -> f64 {
    (computed_ratio - TARGET_RATIO).abs()
}

/// Koide formula check: Q = (m_e + m_mu + m_tau) / (sqrt(m_e) + sqrt(m_mu) + sqrt(m_tau))^2
/// The empirical value should be close to 2/3.
pub fn koide_check(m_e: f64, m_mu: f64, m_tau: f64) -> f64 {
    let numerator = m_e + m_mu + m_tau;
    let denominator = m_e.sqrt() + m_mu.sqrt() + m_tau.sqrt();
    numerator / (denominator * denominator)
}

/// CPU reference implementation of torus knot path length using trapezoidal rule.
///
/// Uses the torus metric convention (matching the GPU kernel in common.cuh):
///   ds^2 = rho^2 * dtheta^2  +  (1 + rho*cos(theta))^2 * dphi^2
///
/// For a (p,q) curve: theta = p*t, phi = q*t, t in [0, 2pi].
///   ds/dt = sqrt( rho^2 * p^2  +  (1 + rho*cos(p*t))^2 * q^2 )
pub fn path_length_cpu(p: i32, q: i32, rho: f64, num_points: usize) -> f64 {
    let n = num_points;
    let dt = 2.0 * std::f64::consts::PI / n as f64;
    let pf = p as f64;
    let qf = q as f64;

    let integrand = |t: f64| -> f64 {
        let cos_pt = (pf * t).cos();
        let term_theta = rho * pf;
        let term_phi = (1.0 + rho * cos_pt) * qf;
        (term_theta * term_theta + term_phi * term_phi).sqrt()
    };

    // Trapezoidal rule
    let mut sum = 0.5 * (integrand(0.0) + integrand(2.0 * std::f64::consts::PI));
    for i in 1..n {
        sum += integrand(i as f64 * dt);
    }
    sum * dt
}

/// CPU reference implementation of Berger sphere path length using trapezoidal rule.
///
/// The Berger sphere is S^3 with the Hopf fiber scaled by parameter lambda.
/// A (p,q) closed curve has path length:
///   L(p, q, lambda) = integral_0^{2pi} sqrt( q^2 + lambda^2*p^2 + 2*lambda^2*p*q*cos(q*t) ) dt
///
/// Special cases:
///   L(1, 0, lambda) = 2*pi*lambda   (pure Hopf fiber)
///   L(0, 1, lambda) = 2*pi          (base S^2 great circle)
pub fn berger_path_length_cpu(p: i32, q: i32, lambda: f64, num_points: usize) -> f64 {
    let n = num_points;
    let dt = 2.0 * std::f64::consts::PI / n as f64;
    let pf = p as f64;
    let qf = q as f64;
    let lam2 = lambda * lambda;

    let integrand = |t: f64| -> f64 {
        let cos_qt = (qf * t).cos();
        (qf * qf + lam2 * pf * pf + 2.0 * lam2 * pf * qf * cos_qt).sqrt()
    };

    // Trapezoidal rule
    let mut sum = 0.5 * (integrand(0.0) + integrand(2.0 * std::f64::consts::PI));
    for i in 1..n {
        sum += integrand(i as f64 * dt);
    }
    sum * dt
}

/// CPU reference implementation of lens space L(n,1) path-length ratio.
///
/// On a lens space with shape parameter sigma, closed geodesics with
/// winding numbers (a,b) have length proportional to:
///   sqrt(a^2 * sigma^2 + b^2 * (1 - sigma^2))
///
/// The ratio L1/L2 is independent of n (the common 2*pi/n factor cancels).
pub fn lens_path_ratio_cpu(a1: i32, b1: i32, a2: i32, b2: i32, sigma: f64) -> f64 {
    let s2 = sigma * sigma;
    let l1 = ((a1 as f64).powi(2) * s2 + (b1 as f64).powi(2) * (1.0 - s2)).sqrt();
    let l2 = ((a2 as f64).powi(2) * s2 + (b2 as f64).powi(2) * (1.0 - s2)).sqrt();
    if l2 > 1e-15 { l1 / l2 } else { 0.0 }
}

fn gcd(a: i32, b: i32) -> i32 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Generate all coprime (p, q) pairs with 1 <= q < p <= max_winding,
/// plus the degenerate cases (1, 0) and (0, 1).
pub fn coprime_pairs(max_winding: i32) -> Vec<(i32, i32)> {
    let mut pairs = vec![(1, 0), (0, 1)];
    for p in 2..=max_winding {
        for q in 1..p {
            if gcd(p, q) == 1 {
                pairs.push((p, q));
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score() {
        assert!((score(TARGET_RATIO) - 0.0).abs() < 1e-15);
        assert!((score(207.0) - (207.0 - TARGET_RATIO)).abs() < 1e-12);
    }

    #[test]
    fn test_koide() {
        let q = koide_check(ELECTRON_MASS, MUON_MASS, TAU_MASS);
        assert!((q - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_coprime_pairs_includes_basics() {
        let pairs = coprime_pairs(5);
        assert!(pairs.contains(&(1, 0)));
        assert!(pairs.contains(&(0, 1)));
        assert!(pairs.contains(&(2, 1)));
        assert!(pairs.contains(&(3, 1)));
        assert!(pairs.contains(&(3, 2)));
        // (4, 2) should NOT be included — gcd is 2
        assert!(!pairs.contains(&(4, 2)));
    }

    #[test]
    fn test_path_length_poloidal() {
        // (1,0) knot: one loop around the tube (poloidal), path = 2*pi*rho
        let rho = 0.5;
        let l = path_length_cpu(1, 0, rho, 1000);
        assert!((l - 2.0 * std::f64::consts::PI * rho).abs() < 0.01);
    }

    #[test]
    fn test_path_length_toroidal() {
        // (0,1) knot: one loop around the hole (toroidal) at theta=0
        // path = 2*pi*(1 + rho)
        let rho = 0.3;
        let l = path_length_cpu(0, 1, rho, 1000);
        assert!((l - 2.0 * std::f64::consts::PI * (1.0 + rho)).abs() < 0.01);
    }

    #[test]
    fn test_berger_path_length_fiber() {
        // (1,0): pure Hopf fiber, length = 2*pi*lambda
        let lambda = 0.5;
        let l = berger_path_length_cpu(1, 0, lambda, 1000);
        assert!((l - 2.0 * std::f64::consts::PI * lambda).abs() < 0.01);
    }

    #[test]
    fn test_berger_path_length_base() {
        // (0,1): base great circle, length = 2*pi
        let lambda = 2.0;
        let l = berger_path_length_cpu(0, 1, lambda, 1000);
        assert!((l - 2.0 * std::f64::consts::PI).abs() < 0.01);
    }

    #[test]
    fn test_lens_round_sphere() {
        // On round S^3 (sigma = 1/sqrt(2)), L(a,b) = sqrt((a^2+b^2)/2)
        let sigma = std::f64::consts::FRAC_1_SQRT_2;
        let ratio = lens_path_ratio_cpu(10, 3, 1, 1, sigma);
        let expected = (100.0 + 9.0_f64).sqrt() / (1.0 + 1.0_f64).sqrt();
        assert!((ratio - expected).abs() < 1e-10);
    }

    #[test]
    fn test_lens_degenerate() {
        // sigma -> 1: only a matters, ratio = a1/a2
        let ratio = lens_path_ratio_cpu(207, 0, 1, 0, 0.9999);
        assert!((ratio - 207.0).abs() < 0.01);
    }
}
