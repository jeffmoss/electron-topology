use crate::physics::{path_length_cpu, ASPDEN_VALUE};

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

/// Check whether a computed value reproduces Aspden's cavity resonance prediction.
pub fn check_aspden_reproduction(value: f64) -> bool {
    (value - ASPDEN_VALUE).abs() < 0.0001
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
}
