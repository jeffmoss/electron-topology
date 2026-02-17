//! Williamson/van der Mark constrained toroidal search.
//!
//! The electron is a (p_e, 2) torus knot (double-loop forced by periodic
//! boundary conditions). The muon is a different mode on the SAME torus.
//! We search for torus geometries where L_muon / L_electron = 206.7682843.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use std::time::Instant;

use crate::gpu;
use crate::physics::{
    self, path_length_cpu, score, WilliamsonCandidate, ALPHA, COMPTON_WAVELENGTH,
    TARGET_RATIO, TAU_ELECTRON_RATIO,
};
use crate::results::{format_number, ResultCollector};

const THREADS_PER_BLOCK: u32 = 256;

/// GPU CandidateResult layout matching kernels/common.cuh
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuCandidate {
    score: f64,
    rho: f64,
    p: i32,
    q: i32,
    path_length: f64,
    ratio: f64,
}

unsafe impl cudarc::driver::ValidAsZeroBits for GpuCandidate {}
unsafe impl cudarc::driver::DeviceRepr for GpuCandidate {}

/// Generate all muon candidate (p, q) modes with 1 <= p, q <= max_winding.
/// Includes non-coprime modes (standing-wave patterns may be physical).
pub fn generate_muon_modes(max_winding: i32) -> (Vec<i32>, Vec<i32>) {
    let mut p_vals = Vec::new();
    let mut q_vals = Vec::new();

    for p in 1..=max_winding {
        for q in 1..=max_winding {
            // Exclude pure electron modes (any_p, 2) since those ARE the electron
            // Actually keep them — different p values give different path lengths
            p_vals.push(p);
            q_vals.push(q);
        }
    }

    (p_vals, q_vals)
}

// ---------------------------------------------------------------------------
// Phase W0: Spectral Direct-Solve
// ---------------------------------------------------------------------------

/// Geodesic path-length solve: finds rho where L_mu/L_e = TARGET on the curved torus.
///
/// Strategy: the geodesic ratio L_mu/L_e approaches q_mu/q_e = q_mu/2 as rho->0,
/// so we need q_mu >= 2*TARGET ~ 414. We scan promising (p_mu, q_mu) pairs and
/// bisect to find rho where the true geodesic ratio equals TARGET.
///
/// Uses path_length_cpu with the correct torus metric:
///   L(p,q,rho) = integral_0^{2pi} sqrt(rho^2 * p^2 + (1 + rho*cos(p*t))^2 * q^2) dt
pub fn solve_williamson_spectral(
    max_p_e: i32,
    max_winding: i32,
) -> Vec<(i32, i32, i32, f64, f64, f64)> {
    // Returns: (p_e, p_mu, q_mu, rho, ratio, score)
    let mut solutions = Vec::new();
    let quad_pts = 1000;
    let num_rho_samples = 200;

    // The thin-torus limit gives ratio ~ q_mu/2, so q_mu must be >= ~2*TARGET = ~414.
    // Modes with smaller q_mu can still work if p_mu is large enough, but the ratio
    // is dominated by q_mu/q_e. We focus on q_mu in [q_min..max_winding].
    let q_min = ((TARGET_RATIO * 2.0).ceil() as i32 - 10).max(1);

    for p_e in [1, 3, 5, 7, 9, 11, 13, 15] {
        if p_e > max_p_e {
            break;
        }

        for q_mu in q_min..=max_winding {
            for p_mu in 1..=max_winding.min(50) {
                if p_mu == p_e && q_mu == 2 {
                    continue;
                }

                // Quick check: at rho->0, ratio -> q_mu/2. At rho->1, ratio is smaller.
                // Only worth checking if q_mu/2 >= TARGET (ratio decreases with rho for q-dominated modes).
                // But for high p_mu, the ratio can increase. Do a quick 2-point check.
                let r_lo = physics::path_length_cpu(p_mu, q_mu, 0.002, quad_pts)
                    / physics::path_length_cpu(p_e, 2, 0.002, quad_pts);
                let r_hi = physics::path_length_cpu(p_mu, q_mu, 0.998, quad_pts)
                    / physics::path_length_cpu(p_e, 2, 0.998, quad_pts);

                let (min_r, max_r) = if r_lo < r_hi { (r_lo, r_hi) } else { (r_hi, r_lo) };
                if TARGET_RATIO < min_r || TARGET_RATIO > max_r {
                    continue; // TARGET not in range — skip
                }

                // Bracket and bisect
                let mut prev_rho = 0.0;
                let mut prev_f = f64::NAN;

                for i in 1..=num_rho_samples {
                    let rho = i as f64 / (num_rho_samples as f64 + 1.0);
                    let l_e = physics::path_length_cpu(p_e, 2, rho, quad_pts);
                    if l_e < 1e-15 {
                        prev_rho = rho;
                        prev_f = f64::NAN;
                        continue;
                    }
                    let l_mu = physics::path_length_cpu(p_mu, q_mu, rho, quad_pts);
                    let f_val = l_mu / l_e - TARGET_RATIO;

                    if !prev_f.is_nan() && prev_f.signum() != f_val.signum() {
                        // Sign change — bisect
                        let mut lo = prev_rho;
                        let mut hi = rho;
                        for _ in 0..60 {
                            let mid = (lo + hi) * 0.5;
                            let le = physics::path_length_cpu(p_e, 2, mid, quad_pts);
                            let lm = physics::path_length_cpu(p_mu, q_mu, mid, quad_pts);
                            if le < 1e-15 { lo = mid; continue; }
                            let fm = lm / le - TARGET_RATIO;
                            if fm.signum() == prev_f.signum() {
                                lo = mid;
                            } else {
                                hi = mid;
                            }
                        }
                        let rho_root = (lo + hi) * 0.5;
                        let le = physics::path_length_cpu(p_e, 2, rho_root, quad_pts);
                        let lm = physics::path_length_cpu(p_mu, q_mu, rho_root, quad_pts);
                        let ratio = lm / le;
                        let s = score(ratio);
                        if s < 1e-6 {
                            solutions.push((p_e, p_mu, q_mu, rho_root, ratio, s));
                        }
                    }

                    prev_rho = rho;
                    prev_f = f_val;
                }
            }
        }
    }

    solutions.sort_by(|a, b| a.5.partial_cmp(&b.5).unwrap_or(std::cmp::Ordering::Equal));
    solutions
}

// ---------------------------------------------------------------------------
// Phase W1: GPU Constrained Scan
// ---------------------------------------------------------------------------

/// Run the williamson_scan GPU kernel for a single p_electron value.
fn run_williamson_gpu_band(
    gpu: &gpu::GpuContext,
    func: &cudarc::driver::CudaFunction,
    p_electron: i32,
    p_muon_gpu: &cudarc::driver::CudaSlice<i32>,
    q_muon_gpu: &cudarc::driver::CudaSlice<i32>,
    num_muon_modes: i32,
    rho_min: f64,
    rho_max: f64,
    num_rho: u32,
    collector: &mut ResultCollector,
) -> Result<u64, Box<dyn std::error::Error>> {
    let rho_step = (rho_max - rho_min) / (num_rho as f64 - 1.0);
    let mut total_evaluated = 0u64;

    let max_threads_per_launch: u64 = 256 * 1024 * 1024;
    let batch_size_rho =
        (max_threads_per_launch / num_muon_modes as u64).min(num_rho as u64) as u32;

    let mut rho_offset = 0u32;

    while rho_offset < num_rho {
        let batch_rho = batch_size_rho.min(num_rho - rho_offset);
        let batch_rho_min = rho_min + rho_offset as f64 * rho_step;
        let batch_rho_i32 = batch_rho as i32;

        let threads = batch_rho as u64 * num_muon_modes as u64;
        let num_blocks = ((threads as u32) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        let mut block_results = gpu.alloc_zeros::<GpuCandidate>(num_blocks as usize)?;

        let cfg = LaunchConfig {
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            grid_dim: (num_blocks, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            gpu.stream
                .launch_builder(func)
                .arg(&batch_rho_i32)
                .arg(&batch_rho_min)
                .arg(&rho_step)
                .arg(&p_electron)
                .arg(p_muon_gpu)
                .arg(q_muon_gpu)
                .arg(&num_muon_modes)
                .arg(&mut block_results)
                .launch(cfg)?;
        }

        let results = gpu.dtoh(&block_results)?;

        for r in &results {
            if r.score < 1.0 && r.score < 1.0e29 {
                collector.add(physics::GeometryCandidate {
                    rho: r.rho,
                    p: r.p,
                    q: r.q,
                    epsilon: 0.0,
                    path_length: r.path_length,
                    ratio: r.ratio,
                    score: r.score,
                    method: format!("williamson p_e={}", p_electron),
                });
            }
        }

        total_evaluated += threads;
        rho_offset += batch_rho;
    }

    Ok(total_evaluated)
}

// ---------------------------------------------------------------------------
// Phase W2: Newton Refinement
// ---------------------------------------------------------------------------

/// Newton's method to find exact rho where L(p_mu, q_mu, rho) / L(p_e, 2, rho) = TARGET.
pub fn newton_williamson_ratio(
    p_e: i32,
    p_mu: i32,
    q_mu: i32,
    rho_init: f64,
) -> Option<(f64, f64, f64)> {
    let quadrature_points = 10_000;
    let mut rho = rho_init;
    let max_iter = 50;
    let tol = 1e-14;

    for _ in 0..max_iter {
        if rho <= 1e-15 || rho >= 1.0 {
            return None;
        }

        let l_e = path_length_cpu(p_e, 2, rho, quadrature_points);
        let l_mu = path_length_cpu(p_mu, q_mu, rho, quadrature_points);
        if l_e < 1e-15 {
            return None;
        }
        let ratio = l_mu / l_e;
        let f_val = ratio - TARGET_RATIO;

        if f_val.abs() < tol {
            return Some((rho, ratio, f_val.abs()));
        }

        // Numerical derivative
        let h = rho * 1e-8;
        let rho_plus = rho + h;
        let rho_minus = (rho - h).max(1e-15);
        let actual_h = rho_plus - rho_minus;

        let le_p = path_length_cpu(p_e, 2, rho_plus, quadrature_points);
        let lmu_p = path_length_cpu(p_mu, q_mu, rho_plus, quadrature_points);
        let le_m = path_length_cpu(p_e, 2, rho_minus, quadrature_points);
        let lmu_m = path_length_cpu(p_mu, q_mu, rho_minus, quadrature_points);

        let f_plus = if le_p > 1e-15 {
            lmu_p / le_p - TARGET_RATIO
        } else {
            return None;
        };
        let f_minus = if le_m > 1e-15 {
            lmu_m / le_m - TARGET_RATIO
        } else {
            return None;
        };

        let df = (f_plus - f_minus) / actual_h;
        if df.abs() < 1e-30 {
            return None;
        }

        let delta = f_val / df;
        rho -= delta;

        if rho <= 0.0 {
            rho = rho_init * 0.5;
        }
    }

    // Return best even if not converged
    let l_e = path_length_cpu(p_e, 2, rho, quadrature_points);
    let l_mu = path_length_cpu(p_mu, q_mu, rho, quadrature_points);
    if l_e > 1e-15 {
        let ratio = l_mu / l_e;
        Some((rho, ratio, (ratio - TARGET_RATIO).abs()))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Phase W3: Physical Validation
// ---------------------------------------------------------------------------

/// Search for a tau mode on the same torus (same rho).
fn search_tau_mode(
    p_e: i32,
    rho: f64,
    max_winding: i32,
) -> (f64, i32, i32, f64) {
    // Returns (score, p_tau, q_tau, ratio)
    let quadrature_points = 10_000;
    let l_e = path_length_cpu(p_e, 2, rho, quadrature_points);
    if l_e < 1e-15 {
        return (f64::MAX, 0, 0, 0.0);
    }

    let mut best_score = f64::MAX;
    let mut best_p = 0;
    let mut best_q = 0;
    let mut best_ratio = 0.0;

    for p in 1..=max_winding {
        for q in 1..=max_winding {
            if p == p_e && q == 2 {
                continue; // skip electron mode
            }
            let l_tau = path_length_cpu(p, q, rho, quadrature_points);
            let ratio = l_tau / l_e;
            let s = (ratio - TAU_ELECTRON_RATIO).abs();
            if s < best_score {
                best_score = s;
                best_p = p;
                best_q = q;
                best_ratio = ratio;
            }
        }
    }

    (best_score, best_p, best_q, best_ratio)
}

/// Build a fully validated WilliamsonCandidate.
fn validate_williamson_candidate(
    p_e: i32,
    p_mu: i32,
    q_mu: i32,
    rho: f64,
) -> WilliamsonCandidate {
    let quadrature_points = 10_000;
    let l_e = path_length_cpu(p_e, 2, rho, quadrature_points);
    let l_mu = path_length_cpu(p_mu, q_mu, rho, quadrature_points);
    let ratio = if l_e > 1e-15 { l_mu / l_e } else { 0.0 };
    let s = score(ratio);

    let charge_ratio = physics::williamson_charge_ratio();
    let g = physics::williamson_g_factor();
    let r_major = physics::williamson_major_radius();
    let r_tube = r_major * rho;

    let (tau_score, p_tau, q_tau, tau_ratio) = search_tau_mode(p_e, rho, 50);

    WilliamsonCandidate {
        p_electron: p_e,
        p_muon: p_mu,
        q_muon: q_mu,
        rho,
        ratio,
        score: s,
        l_electron: l_e,
        l_muon: l_mu,
        physical_r_major: r_major,
        physical_r_tube: r_tube,
        model_charge_ratio: charge_ratio,
        g_factor: g,
        p_tau,
        q_tau,
        tau_ratio,
        tau_score,
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Run the complete Williamson/van der Mark constrained search.
pub fn run_williamson_search(
    gpu: &gpu::GpuContext,
    running: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    quick: bool,
) -> Result<Vec<WilliamsonCandidate>, Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(60));
    if quick {
        println!("=== Williamson/van der Mark Constrained Search (QUICK MODE) ===");
    } else {
        println!("=== Williamson/van der Mark Constrained Search ===");
    }
    println!("{}", "=".repeat(60));
    println!("Electron = (p_e, 2) double-loop | Target ratio: {:.10}", TARGET_RATIO);
    println!();

    let overall_start = Instant::now();

    // Print Williamson model constants
    let charge_ratio = physics::williamson_charge_ratio();
    let g = physics::williamson_g_factor();
    let r_major = physics::williamson_major_radius();
    let alpha_prime = physics::williamson_alpha_prime();
    println!("Williamson model constants:");
    println!("  Model charge q/e = {:.6} (paper: ~0.91)", charge_ratio);
    println!("  Alpha' = {:.6e} (alpha = {:.6e})", alpha_prime, ALPHA);
    println!("  g-factor = {:.6} (QED: 2.00231930436)", g);
    println!("  Major radius R = {:.6e} m", r_major);
    println!("  Compton wavelength = {:.6e} m", COMPTON_WAVELENGTH);
    println!();

    // ===== Phase W0: Spectral Direct-Solve =====
    let w0_start = Instant::now();
    println!("--- Phase W0: Spectral Direct-Solve ---");
    let spectral_max_winding = if quick { 500 } else { 500 };
    let spectral_solutions = solve_williamson_spectral(if quick { 1 } else { 15 }, spectral_max_winding);
    let w0_time = w0_start.elapsed().as_secs_f64();
    println!(
        "  Found {} geodesic solutions (score < 1e-6) in {:.3}s",
        spectral_solutions.len(),
        w0_time
    );

    if !spectral_solutions.is_empty() {
        println!("  Top spectral solutions:");
        for (i, (p_e, p_mu, q_mu, rho, ratio, s)) in spectral_solutions.iter().take(10).enumerate() {
            println!(
                "    #{:>2}: e=({},2) mu=({},{}) rho={:.10} ratio={:.10} score={:.2e}",
                i + 1, p_e, p_mu, q_mu, rho, ratio, s
            );
        }
    }
    println!();

    if !running.load(std::sync::atomic::Ordering::SeqCst) {
        return Ok(Vec::new());
    }

    // ===== Phase W1: GPU Constrained Scan =====
    let w1_start = Instant::now();
    println!("--- Phase W1: GPU Constrained Scan ---");

    // Generate muon modes
    let max_muon_winding = if quick { 500 } else { 500 };
    let (p_muon_vals, q_muon_vals) = generate_muon_modes(max_muon_winding);
    let num_muon_modes = p_muon_vals.len() as i32;
    println!("  Muon candidate modes: {} (all (p,q) with 1 <= p,q <= {})",
             format_number(num_muon_modes as u64), max_muon_winding);

    // Compile kernel
    let func = gpu.compile_kernel("kernels/williamson_scan.cu", "williamson_scan")?;
    println!("  Williamson kernel compiled");

    // Upload muon modes
    let p_muon_gpu = gpu.htod(&p_muon_vals)?;
    let q_muon_gpu = gpu.htod(&q_muon_vals)?;

    let mut collector = ResultCollector::new(200);
    let mut total_gpu_evaluated = 0u64;

    // Rho bands — quick mode uses fewer rho points
    let bands: Vec<(f64, f64, u32, &str)> = if quick {
        vec![
            (0.001, 0.05, 10_000, "thin torus"),
            (0.05, 0.2, 10_000, "moderate"),
            (0.2, 0.5, 10_000, "thick"),
            (0.5, 0.999, 10_000, "near-sphere"),
        ]
    } else {
        vec![
            (0.001, 0.05, 500_000, "thin torus"),
            (0.05, 0.2, 500_000, "moderate"),
            (0.2, 0.5, 200_000, "thick"),
            (0.5, 0.999, 200_000, "near-sphere"),
        ]
    };

    // Quick mode: only p_e=1; full mode: all 8 values
    let p_electron_values: Vec<i32> = if quick { vec![1] } else { vec![1, 3, 5, 7, 9, 11, 13, 15] };

    for &p_e in &p_electron_values {
        if !running.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }

        let pe_start = Instant::now();
        let mut pe_evaluated = 0u64;

        for (band_idx, &(rho_min, rho_max, num_rho, desc)) in bands.iter().enumerate() {
            if !running.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }

            let band_start = Instant::now();
            let evals = run_williamson_gpu_band(
                gpu, &func, p_e,
                &p_muon_gpu, &q_muon_gpu, num_muon_modes,
                rho_min, rho_max, num_rho,
                &mut collector,
            )?;
            pe_evaluated += evals;
            let best_so_far = collector.top_n(1).first().map(|c| c.score).unwrap_or(f64::MAX);
            println!(
                "    p_e={:>2} band {}/{} ({}): {} evals, best: {:.6e} ({:.2}s)",
                p_e, band_idx + 1, bands.len(), desc,
                format_number(evals), best_so_far,
                band_start.elapsed().as_secs_f64(),
            );
        }

        total_gpu_evaluated += pe_evaluated;
        let best_so_far = collector.top_n(1).first().map(|c| c.score).unwrap_or(f64::MAX);
        println!(
            "  p_e={:>2}: {} evals across {} bands, best: {:.6e} ({:.2}s)",
            p_e,
            format_number(pe_evaluated),
            bands.len(),
            best_so_far,
            pe_start.elapsed().as_secs_f64(),
        );
    }

    // Phase W1b: Fine refinement around GPU hits
    let refine_count = if quick { 5 } else { 20 };
    if running.load(std::sync::atomic::Ordering::SeqCst) {
        println!("\n  Fine refinement around top {} GPU hits...", refine_count);
        let top_hits = collector.top_n(30);

        for (i, hit) in top_hits.iter().take(refine_count).enumerate() {
            if !running.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }

            let (p_e_enc, _q_e) = (hit.p / 1000, hit.p % 1000);
            let (p_mu, q_mu) = (hit.q / 1000, hit.q % 1000);

            // Fine band around the hit
            let fine_width = if hit.score < 1e-4 { 1e-5 } else { 1e-4 };
            let fine_min = (hit.rho - fine_width / 2.0).max(0.001);
            let fine_max = (hit.rho + fine_width / 2.0).min(0.999);
            let fine_num: u32 = if quick { 10_000 } else { 1_000_000 };

            // Create small muon mode array with just this mode and neighbors
            let mut fine_p_mu = vec![p_mu];
            let mut fine_q_mu = vec![q_mu];
            // Add neighbors
            for dp in -2..=2i32 {
                for dq in -2..=2i32 {
                    let np = p_mu + dp;
                    let nq = q_mu + dq;
                    if np >= 1 && nq >= 1 && !(np == p_mu && nq == q_mu) {
                        fine_p_mu.push(np);
                        fine_q_mu.push(nq);
                    }
                }
            }
            let fine_num_modes = fine_p_mu.len() as i32;
            let fp_gpu = gpu.htod(&fine_p_mu)?;
            let fq_gpu = gpu.htod(&fine_q_mu)?;

            let evals = run_williamson_gpu_band(
                gpu, &func, p_e_enc,
                &fp_gpu, &fq_gpu, fine_num_modes,
                fine_min, fine_max, fine_num,
                &mut collector,
            )?;
            total_gpu_evaluated += evals;
            let best_now = collector.top_n(1).first().map(|c| c.score).unwrap_or(f64::MAX);
            print!("    refine {}/{}: e=({},{}) mu=({},{}) best={:.6e}", i+1, refine_count, p_e_enc, _q_e, p_mu, q_mu, best_now);

            if i < 5 {
                // Ultra-deep for top 5
                let ud_width = 1e-6;
                let ud_min = (hit.rho - ud_width / 2.0).max(0.001);
                let ud_max = (hit.rho + ud_width / 2.0).min(0.999);
                let ud_num: u32 = if quick { 100_000 } else { 10_000_000 };

                let ud_p = vec![p_mu];
                let ud_q = vec![q_mu];
                let udp_gpu = gpu.htod(&ud_p)?;
                let udq_gpu = gpu.htod(&ud_q)?;

                let evals = run_williamson_gpu_band(
                    gpu, &func, p_e_enc,
                    &udp_gpu, &udq_gpu, 1,
                    ud_min, ud_max, ud_num,
                    &mut collector,
                )?;
                total_gpu_evaluated += evals;
                print!(" (ultra-deep done)");
            }
            println!();
        }
    }

    collector.dedup();

    let w1_time = w1_start.elapsed().as_secs_f64();
    let gpu_top = collector.top_n(50);
    let gpu_best = gpu_top.first().map(|c| c.score).unwrap_or(f64::MAX);
    println!(
        "\nPhase W1 complete: {:.3}s, {} evaluations, best score: {:.6e}",
        w1_time, format_number(total_gpu_evaluated), gpu_best
    );

    if !running.load(std::sync::atomic::Ordering::SeqCst) {
        return Ok(Vec::new());
    }

    // ===== Phase W2: Newton Refinement =====
    let w2_start = Instant::now();
    println!("\n--- Phase W2: Newton Refinement ---");

    // Collect unique candidates from spectral + GPU
    let mut newton_inputs: Vec<(i32, i32, i32, f64)> = Vec::new();

    // From spectral
    for &(p_e, p_mu, q_mu, rho, _, _) in spectral_solutions.iter().take(30) {
        newton_inputs.push((p_e, p_mu, q_mu, rho));
    }

    // From GPU
    for c in gpu_top.iter().take(50) {
        let p_e = c.p / 1000;
        let p_mu = c.q / 1000;
        let q_mu = c.q % 1000;
        if !newton_inputs.iter().any(|&(pe, pm, qm, r)| {
            pe == p_e && pm == p_mu && qm == q_mu && (r - c.rho).abs() < 1e-6
        }) {
            newton_inputs.push((p_e, p_mu, q_mu, c.rho));
        }
    }

    println!("  Refining {} candidates with Newton's method...", newton_inputs.len());

    let mut newton_results: Vec<(i32, i32, i32, f64, f64, f64)> = Vec::new();

    for &(p_e, p_mu, q_mu, rho_init) in &newton_inputs {
        if let Some((rho, ratio, residual)) = newton_williamson_ratio(p_e, p_mu, q_mu, rho_init) {
            newton_results.push((p_e, p_mu, q_mu, rho, ratio, residual));
        }
    }

    newton_results.sort_by(|a, b| a.5.partial_cmp(&b.5).unwrap_or(std::cmp::Ordering::Equal));

    let w2_time = w2_start.elapsed().as_secs_f64();
    if !newton_results.is_empty() {
        println!("  Best Newton residual: {:.6e}", newton_results[0].5);
        for (i, &(p_e, p_mu, q_mu, rho, ratio, residual)) in newton_results.iter().take(10).enumerate() {
            println!(
                "    #{:>2}: e=({},2) mu=({},{}) rho={:.14} ratio={:.14} residual={:.6e}",
                i + 1, p_e, p_mu, q_mu, rho, ratio, residual
            );
        }
    }
    println!("  Newton refinement: {:.3}s", w2_time);

    // ===== Phase W3: Physical Validation =====
    let w3_start = Instant::now();
    println!("\n--- Phase W3: Physical Validation ---");

    // Validate top candidates
    let mut validated: Vec<WilliamsonCandidate> = Vec::new();

    // From Newton results
    for &(p_e, p_mu, q_mu, rho, _, _) in newton_results.iter().take(30) {
        let c = validate_williamson_candidate(p_e, p_mu, q_mu, rho);
        validated.push(c);
    }

    // From spectral (if not already covered by Newton)
    for &(p_e, p_mu, q_mu, rho, _, _) in spectral_solutions.iter().take(20) {
        if !validated.iter().any(|c| {
            c.p_electron == p_e && c.p_muon == p_mu && c.q_muon == q_mu
                && (c.rho - rho).abs() < 1e-10
        }) {
            let c = validate_williamson_candidate(p_e, p_mu, q_mu, rho);
            validated.push(c);
        }
    }

    validated.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));

    let w3_time = w3_start.elapsed().as_secs_f64();
    println!("  Validated {} candidates in {:.3}s", validated.len(), w3_time);

    // Print comprehensive report
    let total_time = overall_start.elapsed().as_secs_f64();
    println!("\n{}", "=".repeat(60));
    println!("=== Williamson Constrained Search Results ===");
    println!("{}", "=".repeat(60));
    println!("Total time: {:.3}s", total_time);
    println!(
        "  Phase W0 (spectral):  {:.3}s, {} solutions",
        w0_time,
        spectral_solutions.len()
    );
    println!(
        "  Phase W1 (GPU):       {:.3}s, {} evaluations",
        w1_time,
        format_number(total_gpu_evaluated)
    );
    println!("  Phase W2 (Newton):    {:.3}s, {} refined", w2_time, newton_results.len());
    println!("  Phase W3 (validate):  {:.3}s, {} validated", w3_time, validated.len());
    println!();

    println!("Williamson Model Constants:");
    println!("  Charge ratio q/e  = {:.6}", charge_ratio);
    println!("  g-factor          = {:.6}", g);
    println!("  Major radius R    = {:.6e} m", r_major);
    println!("  alpha'            = {:.6e}", alpha_prime);
    println!();

    if !validated.is_empty() {
        println!("Top 10 Williamson Candidates:");
        println!("{:-<120}", "");
        println!(
            "{:>3} {:>6} {:>8} {:>6} {:>16} {:>16} {:>12} {:>12} {:>10} {:>6}",
            "#", "e(p,2)", "mu(p,q)", "rho", "ratio", "score",
            "L_electron", "L_muon", "tau_score", "tau"
        );
        println!("{:-<120}", "");

        for (i, c) in validated.iter().take(10).enumerate() {
            println!(
                "{:>3} ({:>2},2) ({:>3},{:>3}) {:.10} {:.12} {:.6e} {:.8} {:.8} {:.2e} ({},{})",
                i + 1,
                c.p_electron,
                c.p_muon,
                c.q_muon,
                c.rho,
                c.ratio,
                c.score,
                c.l_electron,
                c.l_muon,
                c.tau_score,
                c.p_tau,
                c.q_tau,
            );
        }

        // Detailed printout for best candidate
        if let Some(best) = validated.first() {
            println!("\n=== Best Williamson Candidate ===");
            println!("  Electron mode:  ({}, 2)", best.p_electron);
            println!("  Muon mode:      ({}, {})", best.p_muon, best.q_muon);
            println!("  Torus rho:      {:.14}", best.rho);
            println!("  L_electron:     {:.10}", best.l_electron);
            println!("  L_muon:         {:.10}", best.l_muon);
            println!("  Ratio L_mu/L_e: {:.14}", best.ratio);
            println!("  Score:          {:.6e}", best.score);
            println!("  Physical R:     {:.6e} m", best.physical_r_major);
            println!("  Physical r:     {:.6e} m (tube radius)", best.physical_r_tube);
            println!("  Charge q/e:     {:.6}", best.model_charge_ratio);
            println!("  g-factor:       {:.6}", best.g_factor);

            if best.tau_score < 10.0 {
                println!("  Tau mode:       ({}, {}) with ratio {:.6}, score {:.6e}",
                         best.p_tau, best.q_tau, best.tau_ratio, best.tau_score);

                // Koide check
                let m_e = 1.0; // normalized
                let m_mu = best.ratio;
                let m_tau = best.tau_ratio;
                let koide = physics::koide_check(m_e, m_mu, m_tau);
                println!("  Koide Q:        {:.8} (expected 2/3 = {:.8})", koide, 2.0 / 3.0);
            } else {
                println!("  Tau mode:       none found (best score {:.2e})", best.tau_score);
            }
        }
    } else {
        println!("No validated candidates found.");
    }

    // Save results to JSON
    let json = serde_json::to_string_pretty(&validated)?;
    std::fs::write("williamson_results.json", &json)?;
    println!("\nResults saved to williamson_results.json");

    Ok(validated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_muon_modes() {
        let (p, q) = generate_muon_modes(10);
        assert_eq!(p.len(), 100); // 10 * 10
        assert_eq!(q.len(), 100);
    }

    #[test]
    fn test_geodesic_ratio_landscape() {
        // Diagnostic: what ratios L_mu/L_e are achievable?
        let quad_pts = 1000;
        println!("Geodesic ratio landscape for e=(1,2):");
        println!("\n  --- p_mu=1, varying q_mu ---");
        for q_mu in [10, 50, 100, 200, 400, 414, 415, 500] {
            let mut min_ratio = f64::MAX;
            let mut max_ratio = 0.0_f64;
            let mut best_rho = 0.0;
            let mut best_score = f64::MAX;
            for i in 1..=500 {
                let rho = i as f64 / 501.0;
                let l_e = physics::path_length_cpu(1, 2, rho, quad_pts);
                let l_mu = physics::path_length_cpu(1, q_mu, rho, quad_pts);
                let ratio = l_mu / l_e;
                min_ratio = min_ratio.min(ratio);
                max_ratio = max_ratio.max(ratio);
                let s = (ratio - TARGET_RATIO).abs();
                if s < best_score { best_score = s; best_rho = rho; }
            }
            let hit = if best_score < 1.0 { " <-- HIT" } else { "" };
            println!("  mu=(1,{:>3}): range [{:.2}, {:.2}], closest={:.4} at rho={:.4}{}",
                q_mu, min_ratio, max_ratio, TARGET_RATIO - best_score * best_score.signum(), best_rho, hit);
        }
        println!("\n  --- varying p_mu with q_mu=2 (same family as electron) ---");
        for p_mu in [3, 5, 10, 50, 100, 200] {
            let mut min_ratio = f64::MAX;
            let mut max_ratio = 0.0_f64;
            let mut best_rho = 0.0;
            let mut best_score = f64::MAX;
            for i in 1..=500 {
                let rho = i as f64 / 501.0;
                let l_e = physics::path_length_cpu(1, 2, rho, quad_pts);
                let l_mu = physics::path_length_cpu(p_mu, 2, rho, quad_pts);
                let ratio = l_mu / l_e;
                min_ratio = min_ratio.min(ratio);
                max_ratio = max_ratio.max(ratio);
                let s = (ratio - TARGET_RATIO).abs();
                if s < best_score { best_score = s; best_rho = rho; }
            }
            let hit = if best_score < 1.0 { " <-- HIT" } else { "" };
            println!("  mu=({:>3},2): range [{:.2}, {:.2}], closest={:.4} at rho={:.4}{}",
                p_mu, min_ratio, max_ratio, TARGET_RATIO - best_score * best_score.signum(), best_rho, hit);
        }
        println!("\n  --- mixed modes ---");
        for &(p_mu, q_mu) in &[(3, 100), (5, 50), (10, 30), (2, 414), (3, 414), (1, 414)] {
            let mut min_ratio = f64::MAX;
            let mut max_ratio = 0.0_f64;
            let mut best_rho = 0.0;
            let mut best_score = f64::MAX;
            for i in 1..=500 {
                let rho = i as f64 / 501.0;
                let l_e = physics::path_length_cpu(1, 2, rho, quad_pts);
                let l_mu = physics::path_length_cpu(p_mu, q_mu, rho, quad_pts);
                let ratio = l_mu / l_e;
                min_ratio = min_ratio.min(ratio);
                max_ratio = max_ratio.max(ratio);
                let s = (ratio - TARGET_RATIO).abs();
                if s < best_score { best_score = s; best_rho = rho; }
            }
            let hit = if best_score < 1.0 { " <-- HIT" } else { "" };
            println!("  mu=({:>3},{:>3}): range [{:.2}, {:.2}], closest={:.4} at rho={:.4}{}",
                p_mu, q_mu, min_ratio, max_ratio, TARGET_RATIO - best_score * best_score.signum(), best_rho, hit);
        }
    }

    #[test]
    fn test_spectral_solve_finds_solutions() {
        // Need winding >= 414 for geodesic ratio to reach 206.77
        let solutions = solve_williamson_spectral(1, 500);
        assert!(!solutions.is_empty(), "Should find geodesic spectral solutions (need q_mu >= 414)");
        println!("Found {} geodesic solutions for p_e<=1, winding<=500", solutions.len());
        for &(p_e, p_mu, q_mu, rho, ratio, s) in solutions.iter().take(10) {
            println!("  e=({},2) mu=({},{}) rho={:.6} ratio={:.6} score={:.2e}", p_e, p_mu, q_mu, rho, ratio, s);
            assert!(s < 1e-6, "Score {} should be < 1e-6", s);
        }
    }

    #[test]
    fn test_spectral_solve_rho_range() {
        let solutions = solve_williamson_spectral(1, 500);
        for &(_, _, _, rho, _, _) in &solutions {
            assert!(rho > 0.0 && rho < 1.0, "rho {} should be in (0,1)", rho);
        }
    }

    #[test]
    fn test_newton_converges_on_spectral_hit() {
        let solutions = solve_williamson_spectral(1, 500);
        if let Some(&(p_e, p_mu, q_mu, rho, _, _)) = solutions.first() {
            println!("Testing Newton on spectral hit: e=({},2) mu=({},{}) rho={:.10}", p_e, p_mu, q_mu, rho);
            let result = newton_williamson_ratio(p_e, p_mu, q_mu, rho);
            assert!(result.is_some(), "Newton should converge near geodesic spectral solution");
            if let Some((rho_refined, ratio, residual)) = result {
                println!("  Newton: rho={:.14} ratio={:.14} residual={:.2e}", rho_refined, ratio, residual);
                assert!(residual < 1e-8, "Newton residual {} should be < 1e-8", residual);
            }
        }
    }

    #[test]
    fn test_double_loop_limit() {
        // L(1, 2, rho->0) should approach 2*sqrt(rho^2 + 4)*pi ~ 4*pi
        let rho = 0.001;
        let l = path_length_cpu(1, 2, rho, 10_000);
        let expected = 2.0 * std::f64::consts::PI * (rho * rho + 4.0_f64).sqrt();
        assert!(
            (l - expected).abs() < 0.01,
            "L(1,2,{}) = {}, expected {}",
            rho, l, expected
        );
    }
}
