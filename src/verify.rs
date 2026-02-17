use crate::physics::{
    berger_path_length_cpu, dihedral_allowed_modes, nil_eigenvalue_ratio_cpu,
    octahedral_allowed_modes, path_length_cpu, poincare_allowed_modes,
    tetrahedral_allowed_modes, williamson_charge_ratio, williamson_g_factor,
    ASPDEN_VALUE,
};

/// Test known analytical results for torus knot path lengths.
///
/// Returns true if all checks pass within tolerance.
pub fn verify_known_values() -> bool {
    let tol = 1e-3;
    let pi2 = 2.0 * std::f64::consts::PI;
    let n = 10_000; // quadrature points for verification

    // (1,0) torus knot: one poloidal loop around the tube, path = 2*pi*rho
    let rho_10 = 0.5;
    let l_10 = path_length_cpu(1, 0, rho_10, n);
    let expected_10 = pi2 * rho_10;
    let ok_10 = (l_10 - expected_10).abs() < tol;
    if !ok_10 {
        eprintln!(
            "VERIFY FAIL: (1,0) rho={} path = {:.8}, expected {:.8}",
            rho_10, l_10, expected_10
        );
    }

    // (0,1) torus knot: one toroidal loop at theta=0, path = 2*pi*(1+rho)
    let rho = 0.3;
    let l_01 = path_length_cpu(0, 1, rho, n);
    let expected_01 = pi2 * (1.0 + rho);
    let ok_01 = (l_01 - expected_01).abs() < tol;
    if !ok_01 {
        eprintln!(
            "VERIFY FAIL: (0,1) rho={} path = {:.8}, expected {:.8}",
            rho, l_01, expected_01
        );
    }

    // (1,1) thin torus (rho -> 0): path -> 2*pi*sqrt(rho^2 + 1) ~ 2*pi
    let rho_thin = 0.001;
    let l_11 = path_length_cpu(1, 1, rho_thin, n);
    let expected_11 = pi2 * (rho_thin * rho_thin + 1.0).sqrt();
    let ok_11 = (l_11 - expected_11).abs() < tol;
    if !ok_11 {
        eprintln!(
            "VERIFY FAIL: (1,1) thin rho={} path = {:.8}, expected {:.8}",
            rho_thin, l_11, expected_11
        );
    }

    ok_10 && ok_01 && ok_11
}

/// Test known analytical results for Berger sphere path lengths.
///
/// Returns true if all checks pass within tolerance.
pub fn verify_berger_values() -> bool {
    let tol = 1e-3;
    let pi2 = 2.0 * std::f64::consts::PI;
    let n = 10_000;
    let mut all_ok = true;

    // (1,0): pure Hopf fiber, length = 2*pi*lambda for several lambda values
    for &lambda in &[0.25, 0.5, 1.0, 2.0, 5.0] {
        let l = berger_path_length_cpu(1, 0, lambda, n);
        let expected = pi2 * lambda;
        if (l - expected).abs() > tol {
            eprintln!(
                "VERIFY FAIL: Berger (1,0) lambda={} path = {:.8}, expected {:.8}",
                lambda, l, expected
            );
            all_ok = false;
        }
    }

    // (0,1): base great circle, length = 2*pi for several lambda values
    for &lambda in &[0.25, 0.5, 1.0, 2.0, 5.0] {
        let l = berger_path_length_cpu(0, 1, lambda, n);
        let expected = pi2;
        if (l - expected).abs() > tol {
            eprintln!(
                "VERIFY FAIL: Berger (0,1) lambda={} path = {:.8}, expected {:.8}",
                lambda, l, expected
            );
            all_ok = false;
        }
    }

    all_ok
}

/// High-precision CPU reference path length using 1000-point quadrature.
pub fn cpu_reference_check(rho: f64, p: i32, q: i32) -> f64 {
    path_length_cpu(p, q, rho, 1000)
}

/// Check a sample of GPU results against CPU reference.
///
/// Each tuple is (rho, p, q, gpu_path_length).
/// Returns (num_checked, num_mismatches, max_relative_error).
pub fn gpu_cpu_consistency(
    gpu_results: &[(f64, i32, i32, f64)],
    sample_fraction: f64,
) -> (usize, usize, f64) {
    let step = (1.0 / sample_fraction).max(1.0) as usize;
    let mut checked = 0usize;
    let mut mismatches = 0usize;
    let mut max_error = 0.0f64;

    for (i, &(rho, p, q, gpu_path)) in gpu_results.iter().enumerate() {
        if i % step != 0 {
            continue;
        }
        let cpu_path = cpu_reference_check(rho, p, q);
        let rel_error = if cpu_path.abs() > 1e-15 {
            (gpu_path - cpu_path).abs() / cpu_path
        } else {
            (gpu_path - cpu_path).abs()
        };
        checked += 1;
        if rel_error > max_error {
            max_error = rel_error;
        }
        // Tolerance: 1% relative error (GPU uses fewer quadrature points)
        if rel_error > 0.01 {
            mismatches += 1;
        }
    }

    (checked, mismatches, max_error)
}

/// Verify Poincaré homology sphere selection rule.
///
/// Checks that allowed j values match the known binary icosahedral pattern
/// and that all modes are a strict subset of all even-j Berger modes.
pub fn verify_poincare_selection_rule() -> bool {
    let modes = poincare_allowed_modes(60);
    let mut all_ok = true;

    // All j must be even
    for &(j, m) in &modes {
        if j % 2 != 0 {
            eprintln!("VERIFY FAIL: Poincaré j={} is odd", j);
            all_ok = false;
        }
        // m must have same parity as j (both even for even j)
        if m % 2 != j % 2 {
            eprintln!("VERIFY FAIL: Poincaré j={} m={} parity mismatch", j, m);
            all_ok = false;
        }
        if m > j {
            eprintln!("VERIFY FAIL: Poincaré m={} > j={}", m, j);
            all_ok = false;
        }
    }

    // Known allowed j values below 60
    let known_allowed = [0, 12, 20, 24, 30, 32, 36, 40, 42, 44, 48, 50, 52, 54, 56];
    for &j in &known_allowed {
        if !modes.iter().any(|&(jj, _)| jj == j) {
            eprintln!("VERIFY FAIL: Poincaré j={} should be allowed", j);
            all_ok = false;
        }
    }

    // Known disallowed j values
    let known_disallowed = [2, 4, 6, 8, 10, 14, 16, 18, 22, 26, 28, 34, 38, 46];
    for &j in &known_disallowed {
        if modes.iter().any(|&(jj, _)| jj == j) {
            eprintln!("VERIFY FAIL: Poincaré j={} should NOT be allowed", j);
            all_ok = false;
        }
    }

    all_ok
}

/// Verify Nil manifold abelian eigenvalue formula.
pub fn verify_nil_eigenvalues() -> bool {
    let mut all_ok = true;
    let tol = 1e-10;

    // At tau=1: E_{m,n} = 4pi^2 (m^2 + n^2)
    // Ratio (2,0)/(1,0) = 4/1 = 4
    let r1 = nil_eigenvalue_ratio_cpu(2, 0, 1, 0, 1.0);
    if (r1 - 4.0).abs() > tol {
        eprintln!("VERIFY FAIL: Nil (2,0)/(1,0) at tau=1: got {}, expected 4.0", r1);
        all_ok = false;
    }

    // Ratio (1,1)/(1,0) at tau=1 = 2
    let r2 = nil_eigenvalue_ratio_cpu(1, 1, 1, 0, 1.0);
    if (r2 - 2.0).abs() > tol {
        eprintln!("VERIFY FAIL: Nil (1,1)/(1,0) at tau=1: got {}, expected 2.0", r2);
        all_ok = false;
    }

    // At tau=2: E_{1,0} = (2pi*2)^2 = 16pi^2, E_{0,1} = (2pi)^2 = 4pi^2
    // Ratio (1,0)/(0,1) at tau=2 = 16pi^2 / 4pi^2 = 4
    let r3 = nil_eigenvalue_ratio_cpu(1, 0, 0, 1, 2.0);
    if (r3 - 4.0).abs() > tol {
        eprintln!("VERIFY FAIL: Nil (1,0)/(0,1) at tau=2: got {}, expected 4.0", r3);
        all_ok = false;
    }

    // Ratio (3,0)/(1,0) at any tau = 9 (tau cancels when n=0)
    let r4 = nil_eigenvalue_ratio_cpu(3, 0, 1, 0, 0.5);
    if (r4 - 9.0).abs() > tol {
        eprintln!("VERIFY FAIL: Nil (3,0)/(1,0) at tau=0.5: got {}, expected 9.0", r4);
        all_ok = false;
    }

    all_ok
}

/// Verify polyhedral selection rules produce proper subsets.
///
/// The subgroup chain 2T < 2O < 2I in SU(2) implies the containment of
/// allowed j-value sets: Poincaré(2I) ⊆ Octahedral(2O) ⊆ Tetrahedral(2T).
/// A larger group is more restrictive, so its quotient has fewer surviving modes.
pub fn verify_polyhedral_subset() -> bool {
    let max_j = 50;
    let mut all_ok = true;

    let poincare = poincare_allowed_modes(max_j);
    let dihedral3 = dihedral_allowed_modes(3, max_j);
    let tetrahedral = tetrahedral_allowed_modes(max_j);
    let octahedral = octahedral_allowed_modes(max_j);

    // All polyhedral modes should only contain even j
    for &(j, _m) in poincare.iter().chain(dihedral3.iter()).chain(tetrahedral.iter()).chain(octahedral.iter()) {
        if j % 2 != 0 {
            eprintln!("VERIFY FAIL: polyhedral mode j={} is odd", j);
            all_ok = false;
        }
    }

    // j-value set containment: Poincaré ⊆ Octahedral ⊆ Tetrahedral
    let poincare_j: std::collections::HashSet<i32> =
        poincare.iter().map(|&(j, _)| j).collect();
    let octahedral_j: std::collections::HashSet<i32> =
        octahedral.iter().map(|&(j, _)| j).collect();
    let tetrahedral_j: std::collections::HashSet<i32> =
        tetrahedral.iter().map(|&(j, _)| j).collect();

    for &j in &poincare_j {
        if !octahedral_j.contains(&j) {
            eprintln!("VERIFY FAIL: Poincaré j={} not in octahedral set", j);
            all_ok = false;
        }
    }
    for &j in &octahedral_j {
        if !tetrahedral_j.contains(&j) {
            eprintln!("VERIFY FAIL: Octahedral j={} not in tetrahedral set", j);
            all_ok = false;
        }
    }

    // Strict subset: Poincaré should have fewer modes than octahedral
    if poincare.len() >= octahedral.len() {
        eprintln!(
            "VERIFY FAIL: Poincaré ({}) should have fewer modes than octahedral ({})",
            poincare.len(), octahedral.len()
        );
        all_ok = false;
    }

    // Octahedral should have fewer modes than tetrahedral
    if octahedral.len() >= tetrahedral.len() {
        eprintln!(
            "VERIFY FAIL: Octahedral ({}) should have fewer modes than tetrahedral ({})",
            octahedral.len(), tetrahedral.len()
        );
        all_ok = false;
    }

    all_ok
}

/// Check whether a computed value reproduces Aspden's cavity resonance prediction.
pub fn check_aspden_reproduction(value: f64) -> bool {
    (value - ASPDEN_VALUE).abs() < 0.0001
}

/// Verify Williamson model charge: q/e should be a finite positive number.
/// The geometric charge q = sqrt(3*eps0*h*c)/(2pi) gives q/e ~ 2.28.
pub fn verify_williamson_charge() -> bool {
    let qr = williamson_charge_ratio();
    if !qr.is_finite() || qr <= 0.0 || qr > 10.0 {
        eprintln!(
            "VERIFY FAIL: Williamson charge ratio q/e = {:.6}, expected finite positive",
            qr
        );
        return false;
    }
    true
}

/// Verify Williamson g-factor: g = 2*(1 + alpha'/(2pi)), should be slightly above 2.0.
pub fn verify_williamson_g_factor() -> bool {
    let g = williamson_g_factor();
    if g < 2.0 || g > 2.1 {
        eprintln!(
            "VERIFY FAIL: Williamson g-factor = {:.6}, expected slightly above 2.0",
            g
        );
        return false;
    }
    true
}

/// Verify (1,2) double-loop path length in the thin torus limit.
/// L(1,2,rho->0) -> 2*pi*sqrt(1 + 4) ~ 2*pi*sqrt(5)
/// More precisely: L(p,q,rho) = integral_0^{2pi} sqrt(rho^2*p^2 + (1+rho*cos(pt))^2*q^2) dt
/// For rho->0: L(p,q,0) = 2*pi*|q| (the (1+0)^2*q^2 dominates)
pub fn verify_double_loop_limit() -> bool {
    let rho = 0.001;
    let n = 10_000;
    let l = path_length_cpu(1, 2, rho, n);
    // At small rho, L(1,2,rho) ~ 2*pi*sqrt(rho^2 + 4)
    // since the integrand ~ sqrt(rho^2 + (1+rho*cos(t))^2 * 4)
    // ~ sqrt(rho^2 + 4 + 8*rho*cos(t) + 4*rho^2*cos^2(t))
    // The integral is dominated by q=2, giving ~ 2*pi*2 = 4*pi for rho=0
    let expected = 4.0 * std::f64::consts::PI;
    let tol = 0.1; // 1% tolerance at rho=0.001
    if (l - expected).abs() > tol {
        eprintln!(
            "VERIFY FAIL: L(1,2,{}) = {:.8}, expected ~{:.8}",
            rho, l, expected
        );
        return false;
    }
    true
}

/// Verify path-length ratio symmetry: L(p,q,rho)/L(p',q',rho) computed
/// two ways should agree.
pub fn verify_path_length_ratio_symmetry() -> bool {
    let rho = 0.3;
    let n = 10_000;
    let l1 = path_length_cpu(5, 2, rho, n);
    let l2 = path_length_cpu(1, 2, rho, n);
    let ratio_12 = l1 / l2;

    // Compute independently with higher precision
    let l1_hp = path_length_cpu(5, 2, rho, 100_000);
    let l2_hp = path_length_cpu(1, 2, rho, 100_000);
    let ratio_hp = l1_hp / l2_hp;

    let rel_err = (ratio_12 - ratio_hp).abs() / ratio_hp;
    if rel_err > 0.001 {
        eprintln!(
            "VERIFY FAIL: ratio 10k={:.10} vs 100k={:.10}, rel_err={:.6e}",
            ratio_12, ratio_hp, rel_err
        );
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_known_values() {
        assert!(verify_known_values());
    }

    #[test]
    fn test_cpu_reference_check() {
        let pi2 = 2.0 * std::f64::consts::PI;
        // (1,0) at rho=0.5: path = 2*pi*0.5 = pi
        let l = cpu_reference_check(0.5, 1, 0);
        assert!((l - pi2 * 0.5).abs() < 0.01);
    }

    #[test]
    fn test_verify_berger_values() {
        assert!(verify_berger_values());
    }

    #[test]
    fn test_aspden_exact() {
        assert!(check_aspden_reproduction(206.7683078));
    }

    #[test]
    fn test_aspden_close() {
        assert!(check_aspden_reproduction(206.7683));
    }

    #[test]
    fn test_aspden_far() {
        assert!(!check_aspden_reproduction(207.0));
    }

    #[test]
    fn test_gpu_cpu_consistency_empty() {
        let (checked, mismatches, _) = gpu_cpu_consistency(&[], 0.1);
        assert_eq!(checked, 0);
        assert_eq!(mismatches, 0);
    }

    #[test]
    fn test_verify_poincare_selection_rule() {
        assert!(verify_poincare_selection_rule());
    }

    #[test]
    fn test_verify_nil_eigenvalues() {
        assert!(verify_nil_eigenvalues());
    }

    #[test]
    fn test_verify_polyhedral_subset() {
        assert!(verify_polyhedral_subset());
    }

    #[test]
    fn test_verify_williamson_charge() {
        assert!(verify_williamson_charge());
    }

    #[test]
    fn test_verify_williamson_g_factor() {
        assert!(verify_williamson_g_factor());
    }

    #[test]
    fn test_verify_double_loop_limit() {
        assert!(verify_double_loop_limit());
    }

    #[test]
    fn test_verify_path_length_ratio_symmetry() {
        assert!(verify_path_length_ratio_symmetry());
    }
}
