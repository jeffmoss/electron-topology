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

// Physical constants (SI)
pub const C_LIGHT: f64 = 299_792_458.0;           // m/s
pub const H_PLANCK: f64 = 6.62607015e-34;         // J*s
pub const HBAR: f64 = 1.054571817e-34;            // J*s
pub const EPSILON_0: f64 = 8.8541878128e-12;      // F/m
pub const ELEMENTARY_CHARGE: f64 = 1.602176634e-19; // C

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

/// Result from the constrained Williamson/van der Mark search.
#[derive(Clone, Debug, Serialize)]
pub struct WilliamsonCandidate {
    pub p_electron: i32,         // electron poloidal winding (q_e=2 always)
    pub p_muon: i32,             // muon poloidal winding
    pub q_muon: i32,             // muon toroidal winding
    pub rho: f64,                // torus aspect ratio at solution
    pub ratio: f64,              // L_muon / L_electron
    pub score: f64,              // |ratio - TARGET|
    pub l_electron: f64,         // electron normalized path length
    pub l_muon: f64,             // muon normalized path length
    pub physical_r_major: f64,   // major radius (m)
    pub physical_r_tube: f64,    // tube radius (m)
    pub model_charge_ratio: f64, // q_model / e
    pub g_factor: f64,           // gyromagnetic ratio
    pub p_tau: i32,              // best tau mode poloidal
    pub q_tau: i32,              // best tau mode toroidal
    pub tau_ratio: f64,          // L_tau / L_electron
    pub tau_score: f64,          // |tau_ratio - TAU_ELECTRON_RATIO|
}

/// Williamson model charge from the toroidal geometry.
///
/// The paper derives q² = ε₀·h·c / (4π) from the toroidal boundary conditions,
/// giving q/e = √(ε₀·h·c / (4π)) / e = √(α·ℏ·c / (ε₀ · 4π · e²) · ε₀·h·c / (4π)) / e
/// Simplification: q/e = √(α/(2)) ≈ 0.0515...
/// However the relevant ratio for the model is the effective coupling:
/// α' = (q/e)² · α where q/e comes from the geometric constraint.
///
/// We compute q = √(ε₀·h·c·3) / (2π) per the plan specification.
/// This gives q/e ≈ 2.28, representing the bare geometric charge before renormalization.
pub fn williamson_charge_ratio() -> f64 {
    let q_model = (1.0 / (2.0 * std::f64::consts::PI))
        * (3.0 * EPSILON_0 * H_PLANCK * C_LIGHT).sqrt();
    q_model / ELEMENTARY_CHARGE
}

/// Williamson effective fine-structure constant: alpha' = (q/e)^2 * alpha
pub fn williamson_alpha_prime() -> f64 {
    let qr = williamson_charge_ratio();
    qr * qr * ALPHA
}

/// Williamson a-factor: a = 1 + alpha'/(2*pi)
pub fn williamson_a_factor() -> f64 {
    1.0 + williamson_alpha_prime() / (2.0 * std::f64::consts::PI)
}

/// Williamson g-factor: g = 2 * a
pub fn williamson_g_factor() -> f64 {
    2.0 * williamson_a_factor()
}

/// Williamson major radius: R = a * lambda_C / (4*pi)
pub fn williamson_major_radius() -> f64 {
    williamson_a_factor() * COMPTON_WAVELENGTH / (4.0 * std::f64::consts::PI)
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

/// Allowed (j, m) mode pairs for Poincaré homology sphere S³/2I.
/// j must be even and satisfy the binary icosahedral selection rule.
/// m has same parity as j with 0 <= m <= j.
pub fn poincare_allowed_modes(max_j: i32) -> Vec<(i32, i32)> {
    // Allowed even j below 60 (binary icosahedral group, order 120).
    // For j >= 60, ALL even j are allowed.
    const SMALL_ALLOWED: &[i32] = &[
        0, 12, 20, 24, 30, 32, 36, 40, 42, 44, 48, 50, 52, 54, 56,
    ];

    let mut modes = Vec::new();
    let mut j = 0;
    while j <= max_j {
        let allowed = if j >= 60 {
            j % 2 == 0
        } else {
            SMALL_ALLOWED.contains(&j)
        };
        if allowed {
            // m has same parity as j, 0 <= m <= j
            let mut m = 0;
            while m <= j {
                modes.push((j, m));
                m += 2;
            }
        }
        j += 1;
    }
    modes
}

/// Allowed (j, m) mode pairs for binary dihedral quotient S³/2D_n.
/// Selection rule: j even, m ≡ 0 (mod n).
pub fn dihedral_allowed_modes(n: i32, max_j: i32) -> Vec<(i32, i32)> {
    let mut modes = Vec::new();
    let mut j = 0;
    while j <= max_j {
        if j % 2 == 0 {
            let mut m = 0;
            while m <= j {
                if m % n == 0 {
                    modes.push((j, m));
                }
                m += 1;
            }
        }
        j += 1;
    }
    modes
}

/// Allowed (j, m) mode pairs for binary tetrahedral quotient S³/2T (order 24).
/// j must be even. Gaps below j=12: j=2, 4, 10 are NOT allowed.
/// For j >= 12, all even j are allowed.
pub fn tetrahedral_allowed_modes(max_j: i32) -> Vec<(i32, i32)> {
    const SMALL_DISALLOWED: &[i32] = &[2, 4, 10];

    let mut modes = Vec::new();
    let mut j = 0;
    while j <= max_j {
        if j % 2 == 0 {
            let allowed = if j >= 12 {
                true
            } else {
                !SMALL_DISALLOWED.contains(&j)
            };
            if allowed {
                let mut m = 0;
                while m <= j {
                    modes.push((j, m));
                    m += 2;
                }
            }
        }
        j += 1;
    }
    modes
}

/// Allowed (j, m) mode pairs for binary octahedral quotient S³/2O (order 48).
/// j must be even. Gaps below j=24: j=2, 4, 6, 10, 14, 22 are NOT allowed.
/// For j >= 24, all even j are allowed.
pub fn octahedral_allowed_modes(max_j: i32) -> Vec<(i32, i32)> {
    const SMALL_DISALLOWED: &[i32] = &[2, 4, 6, 10, 14, 22];

    let mut modes = Vec::new();
    let mut j = 0;
    while j <= max_j {
        if j % 2 == 0 {
            let allowed = if j >= 24 {
                true
            } else {
                !SMALL_DISALLOWED.contains(&j)
            };
            if allowed {
                let mut m = 0;
                while m <= j {
                    modes.push((j, m));
                    m += 2;
                }
            }
        }
        j += 1;
    }
    modes
}

/// CPU reference: Nil manifold abelian eigenvalue ratio.
/// E_{m,n}(tau) = (2*pi*m*tau)^2 + (2*pi*n)^2
/// Returns E1/E2 for mode pairs (m1,n1) and (m2,n2).
pub fn nil_eigenvalue_ratio_cpu(m1: i32, n1: i32, m2: i32, n2: i32, tau: f64) -> f64 {
    let pi2 = 2.0 * std::f64::consts::PI;
    let e1 = (pi2 * m1 as f64 * tau).powi(2) + (pi2 * n1 as f64).powi(2);
    let e2 = (pi2 * m2 as f64 * tau).powi(2) + (pi2 * n2 as f64).powi(2);
    if e2 > 1e-15 { e1 / e2 } else { 0.0 }
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

    #[test]
    fn test_poincare_allowed_modes() {
        let modes = poincare_allowed_modes(60);
        // j=0 should be present (trivial rep)
        assert!(modes.contains(&(0, 0)));
        // j=12 should be present (first non-trivial)
        assert!(modes.iter().any(|&(j, _)| j == 12));
        // j=2, j=4, j=6, j=8, j=10 should NOT be present
        assert!(!modes.iter().any(|&(j, _)| j == 2));
        assert!(!modes.iter().any(|&(j, _)| j == 4));
        assert!(!modes.iter().any(|&(j, _)| j == 6));
        assert!(!modes.iter().any(|&(j, _)| j == 8));
        assert!(!modes.iter().any(|&(j, _)| j == 10));
        // j=20 should be present
        assert!(modes.iter().any(|&(j, _)| j == 20));
    }

    #[test]
    fn test_dihedral_allowed_modes() {
        let modes = dihedral_allowed_modes(3, 20);
        // j must be even, m must be divisible by 3
        for &(j, m) in &modes {
            assert!(j % 2 == 0, "j={} should be even", j);
            assert!(m % 3 == 0, "m={} should be divisible by 3", m);
        }
    }

    #[test]
    fn test_williamson_charge_ratio() {
        let qr = williamson_charge_ratio();
        // Geometric charge from q = sqrt(3*eps0*h*c)/(2pi) gives q/e ~ 2.28
        assert!(qr > 1.0 && qr < 5.0, "Charge ratio {} out of expected range", qr);
    }

    #[test]
    fn test_williamson_g_factor() {
        let g = williamson_g_factor();
        // g = 2*(1 + alpha'/(2pi)), should be slightly above 2.0
        assert!(g > 2.0 && g < 2.1, "g-factor {} out of expected range", g);
    }

    #[test]
    fn test_williamson_major_radius() {
        let r = williamson_major_radius();
        // Should be on order of 1e-13 meters (sub-Compton)
        assert!(r > 1e-14 && r < 1e-11, "Major radius {} out of expected range", r);
    }

    #[test]
    fn test_nil_eigenvalue_ratio() {
        // At tau=1, E_{m,n} = (2*pi*m)^2 + (2*pi*n)^2 = 4*pi^2*(m^2+n^2)
        // Ratio (2,0)/(1,0) should be 4
        let ratio = nil_eigenvalue_ratio_cpu(2, 0, 1, 0, 1.0);
        assert!((ratio - 4.0).abs() < 1e-10);

        // Ratio (1,1)/(1,0) at tau=1 should be 2
        let ratio2 = nil_eigenvalue_ratio_cpu(1, 1, 1, 0, 1.0);
        assert!((ratio2 - 2.0).abs() < 1e-10);
    }
}
