mod gpu;
mod physics;
mod results;
mod search;
mod spectral;
mod verify;
mod williamson;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use std::time::Instant;

use physics::{
    coprime_pairs, dihedral_allowed_modes, nil_eigenvalue_ratio_cpu, octahedral_allowed_modes,
    poincare_allowed_modes, score, tetrahedral_allowed_modes, GeometryCandidate, SearchResult,
    TARGET_RATIO,
};
use results::{format_number, print_summary, ResultCollector};

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

/// Topology classification for dispatch.
enum Topology {
    Geodesic,
    Berger,
    Lens,
    Poincare,
    Nil,
    Dihedral,
    Tetrahedral,
    Octahedral,
    Other,
}

fn classify_topology(method: &str) -> Topology {
    if method.starts_with("geodesic") {
        Topology::Geodesic
    } else if method.starts_with("berger") {
        Topology::Berger
    } else if method.starts_with("lens") {
        Topology::Lens
    } else if method.starts_with("poincare") {
        Topology::Poincare
    } else if method.starts_with("nil") {
        Topology::Nil
    } else if method.starts_with("dihedral") {
        Topology::Dihedral
    } else if method.starts_with("tetrahedral") {
        Topology::Tetrahedral
    } else if method.starts_with("octahedral") {
        Topology::Octahedral
    } else {
        Topology::Other
    }
}

/// Descriptor for a GPU topology scan. All topologies share the same
/// kernel signature: (num_param, param_min, param_step, p1, q1, p2, q2, num_pairs, out).
struct TopologyScan {
    name: &'static str,
    kernel_path: &'static str,
    kernel_func: &'static str,
    param_name: &'static str,
    method_prefix: &'static str,
    bands: Vec<(f64, f64, u32, &'static str)>,
    /// Pre-computed mode pairs (p1, q1, p2, q2). If None, uses generate_winding_pairs.
    mode_pairs: Option<(Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>)>,
    max_winding: i32,
}

/// Decode encoded pair: p_encoded = p*1000 + q
fn decode_pair(encoded: i32) -> (i32, i32) {
    (encoded / 1000, encoded % 1000)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check for --williamson-only flag to skip existing phases
    let williamson_only = std::env::args().any(|a| a == "--williamson-only" || a == "--williamson");
    let williamson_quick = std::env::args().any(|a| a == "--williamson-quick");

    if williamson_only || williamson_quick {
        println!("=== Williamson/van der Mark Constrained Search (standalone) ===");
    } else {
        println!("=== Topological Resonance Simulation ===");
        println!("=== Multi-Topology Framework: Torus + S³ Quotients + Nil ===");
    }
    println!("Target muon/electron mass ratio: {:.10}", TARGET_RATIO);
    println!();

    // Verify CPU references
    print!("Verifying CPU reference (torus)... ");
    if verify::verify_known_values() {
        println!("OK");
    } else {
        println!("FAILED (continuing anyway)");
    }
    print!("Verifying CPU reference (Berger)... ");
    if verify::verify_berger_values() {
        println!("OK");
    } else {
        println!("FAILED (continuing anyway)");
    }
    print!("Verifying Poincaré selection rule... ");
    if verify::verify_poincare_selection_rule() {
        println!("OK");
    } else {
        println!("FAILED (continuing anyway)");
    }
    print!("Verifying Nil eigenvalues... ");
    if verify::verify_nil_eigenvalues() {
        println!("OK");
    } else {
        println!("FAILED (continuing anyway)");
    }
    print!("Verifying polyhedral subset chain... ");
    if verify::verify_polyhedral_subset() {
        println!("OK");
    } else {
        println!("FAILED (continuing anyway)");
    }
    print!("Verifying Williamson model... ");
    if verify::verify_williamson_charge() && verify::verify_williamson_g_factor() {
        println!("OK");
    } else {
        println!("FAILED (continuing anyway)");
    }

    let gpu = gpu::GpuContext::new()?;
    println!("GPU initialized: device 0\n");

    // Ctrl+C handler
    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, std::sync::atomic::Ordering::SeqCst);
    })
    .ok();

    let overall_start = Instant::now();

    // If --williamson-only, skip straight to the Williamson search
    if williamson_only || williamson_quick {
        match williamson::run_williamson_search(&gpu, &running, williamson_quick) {
            Ok(w_results) => {
                if !w_results.is_empty() {
                    println!(
                        "\nWilliamson search found {} candidates, best score: {:.6e}",
                        w_results.len(),
                        w_results[0].score,
                    );
                }
            }
            Err(e) => {
                eprintln!("Williamson search error: {}", e);
            }
        }
        let total_time = overall_start.elapsed().as_secs_f64();
        println!("\nTotal time: {:.3}s", total_time);
        return Ok(());
    }

    // ===== Phase 0: Direct-Solve Spectral Analysis (CPU, exact) =====
    let phase0_start = Instant::now();
    let phase0 = spectral::run_spectral_phase();
    let phase0_time = phase0_start.elapsed().as_secs_f64();
    println!();

    if !running.load(std::sync::atomic::Ordering::SeqCst) {
        save_results(&[phase0])?;
        return Ok(());
    }

    // ===== Phase 1: GPU Coarse Scans (all topologies) =====
    let mut phase1_results: Vec<SearchResult> = Vec::new();
    let mut phase1_timings: Vec<(&str, f64)> = Vec::new();

    // --- Phase 1a: Torus Geodesic ---
    let scan = TopologyScan {
        name: "Torus Geodesic",
        kernel_path: "kernels/geodesic_path.cu",
        kernel_func: "geodesic_scan",
        param_name: "rho",
        method_prefix: "geodesic",
        bands: vec![
            (0.0001, 0.001, 200_000, "ultra-thin"),
            (0.001, 0.01, 200_000, "thin"),
            (0.01, 0.1, 100_000, "moderate"),
            (0.1, 0.999, 50_000, "thick"),
            (0.003, 0.007, 1_000_000, "target region (rho ~ 1/207)"),
            (0.005, 0.012, 500_000, "alpha region"),
            (0.0001, 0.001, 500_000, "tau hint region"),
        ],
        mode_pairs: None,
        max_winding: 30,
    };
    let t = Instant::now();
    let result = run_topology_scan(&gpu, &running, &scan)?;
    phase1_timings.push(("Torus geodesic", t.elapsed().as_secs_f64()));
    print_summary(&result);
    println!();
    phase1_results.push(result);

    if !running.load(std::sync::atomic::Ordering::SeqCst) {
        let mut all = vec![phase0];
        all.extend(phase1_results);
        save_results(&all)?;
        return Ok(());
    }

    // GPU-CPU consistency spot-check on torus candidates
    println!("GPU-CPU consistency check (torus)...");
    let spot_checks: Vec<(f64, i32, i32, f64)> = phase1_results[0]
        .candidates
        .iter()
        .filter(|c| c.method.starts_with("geodesic"))
        .take(10)
        .map(|c| {
            let (p1, q1) = decode_pair(c.p);
            (c.rho, p1, q1, c.path_length)
        })
        .collect();
    let (checked, mismatches, max_err) = verify::gpu_cpu_consistency(&spot_checks, 1.0);
    println!(
        "  Checked {}, mismatches: {}, max relative error: {:.6e}",
        checked, mismatches, max_err
    );
    println!();

    // --- Phase 1b: Berger Sphere ---
    let scan = TopologyScan {
        name: "Berger Sphere",
        kernel_path: "kernels/berger_sphere.cu",
        kernel_func: "berger_scan",
        param_name: "λ",
        method_prefix: "berger",
        bands: vec![
            (0.001, 0.1, 200_000, "thin fibers"),
            (0.1, 1.0, 200_000, "sub-round"),
            (1.0, 10.0, 200_000, "super-round"),
            (0.01, 0.05, 500_000, "target region"),
            (0.003, 0.01, 500_000, "ultra-thin fibers"),
        ],
        mode_pairs: None,
        max_winding: 30,
    };
    let t = Instant::now();
    let result = run_topology_scan(&gpu, &running, &scan)?;
    phase1_timings.push(("Berger sphere", t.elapsed().as_secs_f64()));
    print_summary(&result);
    println!();
    phase1_results.push(result);

    if !running.load(std::sync::atomic::Ordering::SeqCst) {
        let mut all = vec![phase0];
        all.extend(phase1_results);
        save_results(&all)?;
        return Ok(());
    }

    // --- Phase 1c: Lens Space ---
    let scan = TopologyScan {
        name: "Lens Space",
        kernel_path: "kernels/lens_space.cu",
        kernel_func: "lens_scan",
        param_name: "σ",
        method_prefix: "lens",
        bands: vec![
            (0.001, 0.999, 1_000_000, "full range"),
            (0.001, 0.01, 500_000, "extreme thin"),
            (0.99, 0.999, 500_000, "extreme thick"),
            (0.003, 0.007, 1_000_000, "target region"),
        ],
        mode_pairs: None,
        max_winding: 30,
    };
    let t = Instant::now();
    let result = run_topology_scan(&gpu, &running, &scan)?;
    phase1_timings.push(("Lens space", t.elapsed().as_secs_f64()));
    print_summary(&result);
    println!();
    phase1_results.push(result);

    if !running.load(std::sync::atomic::Ordering::SeqCst) {
        let mut all = vec![phase0];
        all.extend(phase1_results);
        save_results(&all)?;
        return Ok(());
    }

    // --- Phase 1d: Poincaré Homology Sphere (S³/2I) ---
    let poincare_pairs = generate_quotient_pairs(&poincare_allowed_modes(60));
    let scan = TopologyScan {
        name: "Poincaré Homology Sphere",
        kernel_path: "kernels/poincare_scan.cu",
        kernel_func: "poincare_scan",
        param_name: "λ",
        method_prefix: "poincare",
        bands: vec![
            (0.001, 0.1, 200_000, "thin fibers"),
            (0.1, 1.0, 200_000, "sub-round"),
            (1.0, 10.0, 200_000, "super-round"),
            (0.01, 0.05, 500_000, "target region"),
            (0.003, 0.01, 500_000, "ultra-thin fibers"),
        ],
        mode_pairs: Some(poincare_pairs),
        max_winding: 0,
    };
    let t = Instant::now();
    let result = run_topology_scan(&gpu, &running, &scan)?;
    phase1_timings.push(("Poincaré S³/2I", t.elapsed().as_secs_f64()));
    print_summary(&result);
    println!();
    phase1_results.push(result);

    if !running.load(std::sync::atomic::Ordering::SeqCst) {
        let mut all = vec![phase0];
        all.extend(phase1_results);
        save_results(&all)?;
        return Ok(());
    }

    // --- Phase 1e: Nil Manifold ---
    let nil_pairs = generate_nil_pairs(30);
    let scan = TopologyScan {
        name: "Nil Manifold",
        kernel_path: "kernels/nil_scan.cu",
        kernel_func: "nil_scan",
        param_name: "τ",
        method_prefix: "nil",
        bands: vec![
            (0.001, 1.0, 500_000, "sub-unity tau"),
            (1.0, 10.0, 500_000, "super-unity tau"),
            (0.01, 0.1, 500_000, "target region"),
            (0.001, 0.01, 500_000, "extreme thin tau"),
        ],
        mode_pairs: Some(nil_pairs),
        max_winding: 0,
    };
    let t = Instant::now();
    let result = run_topology_scan(&gpu, &running, &scan)?;
    phase1_timings.push(("Nil manifold", t.elapsed().as_secs_f64()));
    print_summary(&result);
    println!();
    phase1_results.push(result);

    if !running.load(std::sync::atomic::Ordering::SeqCst) {
        let mut all = vec![phase0];
        all.extend(phase1_results);
        save_results(&all)?;
        return Ok(());
    }

    // --- Phase 1f: Binary Polyhedral Quotients (reuse poincare_scan kernel) ---
    // Tetrahedral S³/2T
    let tet_pairs = generate_quotient_pairs(&tetrahedral_allowed_modes(60));
    let scan = TopologyScan {
        name: "Tetrahedral S³/2T",
        kernel_path: "kernels/poincare_scan.cu",
        kernel_func: "poincare_scan",
        param_name: "λ",
        method_prefix: "tetrahedral",
        bands: vec![
            (0.001, 0.1, 200_000, "thin fibers"),
            (0.1, 1.0, 200_000, "sub-round"),
            (1.0, 10.0, 200_000, "super-round"),
            (0.01, 0.05, 500_000, "target region"),
        ],
        mode_pairs: Some(tet_pairs),
        max_winding: 0,
    };
    let t = Instant::now();
    let result = run_topology_scan(&gpu, &running, &scan)?;
    phase1_timings.push(("Tetrahedral S³/2T", t.elapsed().as_secs_f64()));
    print_summary(&result);
    println!();
    phase1_results.push(result);

    // Octahedral S³/2O
    if running.load(std::sync::atomic::Ordering::SeqCst) {
        let oct_pairs = generate_quotient_pairs(&octahedral_allowed_modes(60));
        let scan = TopologyScan {
            name: "Octahedral S³/2O",
            kernel_path: "kernels/poincare_scan.cu",
            kernel_func: "poincare_scan",
            param_name: "λ",
            method_prefix: "octahedral",
            bands: vec![
                (0.001, 0.1, 200_000, "thin fibers"),
                (0.1, 1.0, 200_000, "sub-round"),
                (1.0, 10.0, 200_000, "super-round"),
                (0.01, 0.05, 500_000, "target region"),
            ],
            mode_pairs: Some(oct_pairs),
            max_winding: 0,
        };
        let t = Instant::now();
        let result = run_topology_scan(&gpu, &running, &scan)?;
        phase1_timings.push(("Octahedral S³/2O", t.elapsed().as_secs_f64()));
        print_summary(&result);
        println!();
        phase1_results.push(result);
    }

    // Dihedral S³/2D_3
    if running.load(std::sync::atomic::Ordering::SeqCst) {
        let d3_pairs = generate_quotient_pairs(&dihedral_allowed_modes(3, 60));
        let scan = TopologyScan {
            name: "Dihedral S³/2D₃",
            kernel_path: "kernels/poincare_scan.cu",
            kernel_func: "poincare_scan",
            param_name: "λ",
            method_prefix: "dihedral",
            bands: vec![
                (0.001, 0.1, 200_000, "thin fibers"),
                (0.1, 1.0, 200_000, "sub-round"),
                (1.0, 10.0, 200_000, "super-round"),
                (0.01, 0.05, 500_000, "target region"),
            ],
            mode_pairs: Some(d3_pairs),
            max_winding: 0,
        };
        let t = Instant::now();
        let result = run_topology_scan(&gpu, &running, &scan)?;
        phase1_timings.push(("Dihedral S³/2D₃", t.elapsed().as_secs_f64()));
        print_summary(&result);
        println!();
        phase1_results.push(result);
    }

    // Merge all Phase 1 results
    let mut merged_phase1 = SearchResult {
        phase: 1,
        candidates: Vec::new(),
        total_evaluated: 0,
        best_score: f64::MAX,
        elapsed_secs: 0.0,
    };
    for r in &phase1_results {
        merged_phase1
            .candidates
            .extend(r.candidates.iter().cloned());
        merged_phase1.total_evaluated += r.total_evaluated;
        if r.best_score < merged_phase1.best_score {
            merged_phase1.best_score = r.best_score;
        }
    }
    merged_phase1
        .candidates
        .sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
    merged_phase1.candidates.truncate(100);

    // ===== Phase 2: Fine-grain refinement + Helmholtz cross-check =====
    let phase2_start = Instant::now();
    let phase2 = run_phase2(&gpu, &running, &merged_phase1)?;
    let phase2_time = phase2_start.elapsed().as_secs_f64();
    print_summary(&phase2);
    println!();

    if !running.load(std::sync::atomic::Ordering::SeqCst) {
        save_results(&[phase0.clone(), merged_phase1, phase2])?;
        return Ok(());
    }

    // ===== Phase 3: High-precision CPU analysis =====
    let phase3_start = Instant::now();
    let phase3 = run_phase3(&phase2)?;
    let phase3_time = phase3_start.elapsed().as_secs_f64();
    print_summary(&phase3);
    println!();

    save_results(&[phase0, merged_phase1, phase2, phase3])?;

    // ===== Williamson/van der Mark Constrained Search =====
    let williamson_start = Instant::now();
    if running.load(std::sync::atomic::Ordering::SeqCst) {
        match williamson::run_williamson_search(&gpu, &running, false) {
            Ok(w_results) => {
                if !w_results.is_empty() {
                    println!(
                        "\nWilliamson search found {} candidates, best score: {:.6e}",
                        w_results.len(),
                        w_results[0].score,
                    );
                }
            }
            Err(e) => {
                eprintln!("Williamson search error: {}", e);
            }
        }
    }
    let williamson_time = williamson_start.elapsed().as_secs_f64();

    // Timing breakdown
    let total_time = overall_start.elapsed().as_secs_f64();
    println!("\n=== Timing Breakdown ===");
    println!(
        "  Phase 0 (spectral solve): {:.3}s ({:.1}%)",
        phase0_time,
        phase0_time / total_time * 100.0
    );
    for (name, t) in &phase1_timings {
        println!(
            "  Phase 1 ({}): {:.3}s ({:.1}%)",
            name,
            t,
            t / total_time * 100.0
        );
    }
    println!(
        "  Phase 2 (refinement):     {:.3}s ({:.1}%)",
        phase2_time,
        phase2_time / total_time * 100.0
    );
    println!(
        "  Phase 3 (CPU analysis):   {:.3}s ({:.1}%)",
        phase3_time,
        phase3_time / total_time * 100.0
    );
    println!(
        "  Williamson search:        {:.3}s ({:.1}%)",
        williamson_time,
        williamson_time / total_time * 100.0
    );
    println!("  Total:                    {:.3}s", total_time);

    Ok(())
}

// ---------------------------------------------------------------------------
// Mode pair generators
// ---------------------------------------------------------------------------

/// Generate all ordered pairs of coprime (p,q) winding numbers.
/// Returns (p1,q1,p2,q2) where L(p1,q1)/L(p2,q2) could be > 1.
fn generate_winding_pairs(max_winding: i32) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let modes = coprime_pairs(max_winding);
    let mut p1s = Vec::new();
    let mut q1s = Vec::new();
    let mut p2s = Vec::new();
    let mut q2s = Vec::new();

    for (i, &(pa, qa)) in modes.iter().enumerate() {
        for (j, &(pb, qb)) in modes.iter().enumerate() {
            if i == j {
                continue;
            }
            let winding_a = (pa as f64).hypot(qa as f64);
            let winding_b = (pb as f64).hypot(qb as f64);
            if winding_a > winding_b {
                p1s.push(pa);
                q1s.push(qa);
                p2s.push(pb);
                q2s.push(qb);
            }
        }
    }

    (p1s, q1s, p2s, q2s)
}

/// Generate mode-pair arrays for an S³ quotient topology from allowed (j,m) modes.
/// The poincare_scan kernel encodes my_p = p1*1000+q1, my_q = p2*1000+q2,
/// so we pass (j,m) values directly as the winding pair arrays.
fn generate_quotient_pairs(
    modes: &[(i32, i32)],
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut p1s = Vec::new();
    let mut q1s = Vec::new();
    let mut p2s = Vec::new();
    let mut q2s = Vec::new();

    // Denominator: small-j modes (ground state candidates)
    let denom_modes: Vec<&(i32, i32)> = modes.iter().filter(|(j, _)| *j <= 10).collect();

    for &&(j2, m2) in &denom_modes {
        for &(j1, m1) in modes {
            if j1 == j2 && m1 == m2 {
                continue;
            }
            // Numerator should have larger quantum numbers for ratio > 1
            let e1_approx = j1 as f64 * (j1 as f64 + 2.0) + m1 as f64 * m1 as f64;
            let e2_approx = j2 as f64 * (j2 as f64 + 2.0) + m2 as f64 * m2 as f64;
            if e1_approx > e2_approx {
                p1s.push(j1);
                q1s.push(m1);
                p2s.push(j2);
                q2s.push(m2);
            }
        }
    }

    (p1s, q1s, p2s, q2s)
}

/// Generate mode-pair arrays for Nil manifold scan.
/// Mode pairs are (m1,n1)/(m2,n2) integer quantum numbers.
fn generate_nil_pairs(max_mode: i32) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut m1s = Vec::new();
    let mut n1s = Vec::new();
    let mut m2s = Vec::new();
    let mut n2s = Vec::new();

    for m2 in 1..=max_mode.min(10) {
        for n2 in 0..=max_mode.min(10) {
            for m1 in 1..=max_mode {
                for n1 in 0..=max_mode {
                    if m1 == m2 && n1 == n2 {
                        continue;
                    }
                    // Numerator should produce larger eigenvalue at tau=1
                    let e1 = (m1 * m1 + n1 * n1) as f64;
                    let e2 = (m2 * m2 + n2 * n2) as f64;
                    if e1 > e2 {
                        m1s.push(m1);
                        n1s.push(n1);
                        m2s.push(m2);
                        n2s.push(n2);
                    }
                }
            }
        }
    }

    (m1s, n1s, m2s, n2s)
}

// ---------------------------------------------------------------------------
// Generic GPU topology scan
// ---------------------------------------------------------------------------

/// Run a GPU coarse scan for any topology. All kernels share the signature:
/// (num_param, param_min, param_step, p1, q1, p2, q2, num_pairs, block_results)
fn run_topology_scan(
    gpu: &gpu::GpuContext,
    running: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    scan: &TopologyScan,
) -> Result<SearchResult, Box<dyn std::error::Error>> {
    println!("--- Phase 1: {} Scan ---", scan.name);
    let start = Instant::now();

    let func = gpu.compile_kernel(scan.kernel_path, scan.kernel_func)?;
    println!("{} kernel compiled", scan.name);

    let (p1s, q1s, p2s, q2s) = match &scan.mode_pairs {
        Some((p1, q1, p2, q2)) => (p1.clone(), q1.clone(), p2.clone(), q2.clone()),
        None => generate_winding_pairs(scan.max_winding),
    };
    let num_pairs = p1s.len();
    println!("Using {} mode pairs", format_number(num_pairs as u64));

    let p1_gpu = gpu.htod(&p1s)?;
    let q1_gpu = gpu.htod(&q1s)?;
    let p2_gpu = gpu.htod(&p2s)?;
    let q2_gpu = gpu.htod(&q2s)?;

    let mut collector = ResultCollector::new(200);
    let mut total_evaluated = 0u64;
    let mut band_num = 0u32;

    for (param_min, param_max, num_param, desc) in &scan.bands {
        if !running.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }

        let param_step = (param_max - param_min) / (*num_param as f64 - 1.0);
        let num_pairs_i32 = num_pairs as i32;
        let total_expected = *num_param as u64 * num_pairs as u64;

        let max_threads_per_launch: u64 = 256 * 1024 * 1024;
        let batch_size_param =
            (max_threads_per_launch / num_pairs as u64).min(*num_param as u64) as u32;

        let mut param_offset = 0u32;
        let band_start = Instant::now();
        let mut last_progress_print = start.elapsed().as_secs_f64();

        while param_offset < *num_param {
            if !running.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }

            let batch_param = batch_size_param.min(*num_param - param_offset);
            let batch_param_min = *param_min + param_offset as f64 * param_step;
            let batch_param_i32 = batch_param as i32;

            let threads = batch_param as u64 * num_pairs as u64;
            let num_blocks = ((threads as u32) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            let mut block_results = gpu.alloc_zeros::<GpuCandidate>(num_blocks as usize)?;

            let cfg = LaunchConfig {
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                grid_dim: (num_blocks, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                gpu.stream
                    .launch_builder(&func)
                    .arg(&batch_param_i32)
                    .arg(&batch_param_min)
                    .arg(&param_step)
                    .arg(&p1_gpu)
                    .arg(&q1_gpu)
                    .arg(&p2_gpu)
                    .arg(&q2_gpu)
                    .arg(&num_pairs_i32)
                    .arg(&mut block_results)
                    .launch(cfg)?;
            }

            let results = gpu.dtoh(&block_results)?;

            for r in &results {
                if r.score < 1.0 && r.score < 1.0e29 {
                    let (p1, q1) = decode_pair(r.p);
                    let (p2, q2) = decode_pair(r.q);
                    collector.add(GeometryCandidate {
                        rho: r.rho,
                        p: r.p,
                        q: r.q,
                        epsilon: 0.0,
                        path_length: r.path_length,
                        ratio: r.ratio,
                        score: r.score,
                        method: format!(
                            "{} ({},{})÷({},{})",
                            scan.method_prefix, p1, q1, p2, q2
                        ),
                    });
                }
            }

            total_evaluated += threads;
            param_offset += batch_param;

            let now = start.elapsed().as_secs_f64();
            if now - last_progress_print > 5.0 {
                let band_evaluated = param_offset as u64 * num_pairs as u64;
                let pct = (band_evaluated as f64 / total_expected as f64) * 100.0;
                let best_so_far = collector
                    .top_n(1)
                    .first()
                    .map(|c| c.score)
                    .unwrap_or(f64::MAX);
                println!(
                    "    Progress: {:.1}% ({} evals, best: {:.6e})",
                    pct,
                    format_number(band_evaluated),
                    best_so_far
                );
                last_progress_print = now;
            }
        }

        band_num += 1;
        let best_so_far = collector
            .top_n(1)
            .first()
            .map(|c| c.score)
            .unwrap_or(f64::MAX);

        println!(
            "  Band {}/{} [{}]: {}=[{:.4},{:.4}] x {} pts x {} pairs = {} evals | best: {:.6e} ({:.2}s)",
            band_num,
            scan.bands.len(),
            desc,
            scan.param_name,
            param_min,
            param_max,
            format_number(*num_param as u64),
            format_number(num_pairs as u64),
            format_number(*num_param as u64 * num_pairs as u64),
            best_so_far,
            band_start.elapsed().as_secs_f64(),
        );

        // Intermediate persistence
        let partial = SearchResult {
            phase: 1,
            candidates: collector.top_n(50),
            total_evaluated,
            best_score: best_so_far,
            elapsed_secs: start.elapsed().as_secs_f64(),
        };
        if let Ok(json) = serde_json::to_string_pretty(&[&partial]) {
            let _ = std::fs::write("results_partial.json", &json);
        }
    }

    collector.dedup();

    let elapsed = start.elapsed().as_secs_f64();
    let top = collector.top_n(50);
    let best_score = top.first().map(|c| c.score).unwrap_or(f64::MAX);

    println!(
        "\n{} scan complete: {:.3}s, {} total evaluations, best score: {:.6e}",
        scan.name,
        elapsed,
        format_number(total_evaluated),
        best_score
    );

    if let Some(best) = top.first() {
        let (p1, q1) = decode_pair(best.p);
        let (p2, q2) = decode_pair(best.q);
        println!(
            "  BEST: ({},{})÷({},{}) {}={:.10} ratio={:.10} score={:.6e}",
            p1, q1, p2, q2, scan.param_name, best.rho, best.ratio, best.score
        );
    }

    Ok(SearchResult {
        phase: 1,
        candidates: top,
        total_evaluated,
        best_score,
        elapsed_secs: elapsed,
    })
}

// ---------------------------------------------------------------------------
// Phase 2: Ultra-fine refinement
// ---------------------------------------------------------------------------

/// Phase 2: Ultra-fine refinement around Phase 1 hits + Helmholtz cross-check.
fn run_phase2(
    gpu: &gpu::GpuContext,
    running: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    phase1: &SearchResult,
) -> Result<SearchResult, Box<dyn std::error::Error>> {
    println!("--- Phase 2: Ultra-Fine Refinement ---");
    let start = Instant::now();

    if phase1.candidates.is_empty() {
        println!("No Phase 1 candidates to refine.");
        return Ok(SearchResult {
            phase: 2,
            candidates: vec![],
            total_evaluated: 0,
            best_score: f64::MAX,
            elapsed_secs: 0.0,
        });
    }

    // Compile all topology kernels needed for refinement
    let geodesic_fn = gpu.compile_kernel("kernels/geodesic_path.cu", "geodesic_scan")?;
    let berger_fn = gpu.compile_kernel("kernels/berger_sphere.cu", "berger_scan")?;
    let lens_fn = gpu.compile_kernel("kernels/lens_space.cu", "lens_scan")?;
    let poincare_fn = gpu.compile_kernel("kernels/poincare_scan.cu", "poincare_scan")?;
    let nil_fn = gpu.compile_kernel("kernels/nil_scan.cu", "nil_scan")?;

    let mut collector = ResultCollector::new(200);
    let mut total_evaluated = 0u64;

    let top_hits: Vec<GeometryCandidate> = phase1.candidates.iter().take(20).cloned().collect();

    let mut unique_regions: Vec<(i32, i32, f64, f64, &str)> = Vec::new();
    for hit in &top_hits {
        if !unique_regions
            .iter()
            .any(|k| k.0 == hit.p && k.1 == hit.q && (k.2 - hit.rho).abs() < 0.001)
        {
            unique_regions.push((hit.p, hit.q, hit.rho, hit.score, &hit.method));
        }
    }

    let mut region_best: Vec<(i32, i32, f64, f64, String)> = Vec::new();

    for (idx, &(p_enc, q_enc, param_center, hit_score, ref method)) in
        unique_regions.iter().enumerate()
    {
        if !running.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }

        let region_start = Instant::now();
        let topo = classify_topology(method);
        let (p1, q1) = decode_pair(p_enc);
        let (p2, q2) = decode_pair(q_enc);

        let mut p1s = vec![p1];
        let mut q1s = vec![q1];
        let mut p2s = vec![p2];
        let mut q2s = vec![q2];

        // Add neighboring mode numbers for basic topologies
        match topo {
            Topology::Geodesic | Topology::Berger | Topology::Lens => {
                for dp1 in -1..=1i32 {
                    for dq1 in -1..=1i32 {
                        for dp2 in -1..=1i32 {
                            for dq2 in -1..=1i32 {
                                let np1 = p1 + dp1;
                                let nq1 = q1 + dq1;
                                let np2 = p2 + dp2;
                                let nq2 = q2 + dq2;
                                if np1 >= 0
                                    && nq1 >= 0
                                    && np2 >= 0
                                    && nq2 >= 0
                                    && (np1 > 0 || nq1 > 0)
                                    && (np2 > 0 || nq2 > 0)
                                {
                                    p1s.push(np1);
                                    q1s.push(nq1);
                                    p2s.push(np2);
                                    q2s.push(nq2);
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                // Quotient/Nil topologies: just use the winning pair
            }
        }

        let num_pairs = p1s.len() as i32;
        let p1_gpu = gpu.htod(&p1s)?;
        let q1_gpu = gpu.htod(&q1s)?;
        let p2_gpu = gpu.htod(&p2s)?;
        let q2_gpu = gpu.htod(&q2s)?;

        let param_width = if hit_score < 1e-4 {
            0.00001
        } else if hit_score < 0.01 {
            0.0001
        } else {
            0.001
        };

        let param_min = (param_center - param_width / 2.0).max(0.0001);
        let param_max = param_center + param_width / 2.0;
        let num_param: i32 = 10_000_000;
        let param_step = (param_max - param_min) / (num_param - 1) as f64;

        let max_threads_per_launch: u64 = 256 * 1024 * 1024;
        let batch_size_param =
            (max_threads_per_launch / num_pairs as u64).min(num_param as u64) as i32;

        let kernel_fn = match topo {
            Topology::Geodesic => &geodesic_fn,
            Topology::Berger => &berger_fn,
            Topology::Lens => &lens_fn,
            Topology::Poincare
            | Topology::Dihedral
            | Topology::Tetrahedral
            | Topology::Octahedral => &poincare_fn,
            Topology::Nil => &nil_fn,
            Topology::Other => &geodesic_fn,
        };

        let method_prefix = match topo {
            Topology::Geodesic => "geodesic-fine",
            Topology::Berger => "berger-fine",
            Topology::Lens => "lens-fine",
            Topology::Poincare => "poincare-fine",
            Topology::Nil => "nil-fine",
            Topology::Dihedral => "dihedral-fine",
            Topology::Tetrahedral => "tetrahedral-fine",
            Topology::Octahedral => "octahedral-fine",
            Topology::Other => "other-fine",
        };

        let param_name = match topo {
            Topology::Geodesic => "rho",
            Topology::Berger
            | Topology::Poincare
            | Topology::Dihedral
            | Topology::Tetrahedral
            | Topology::Octahedral => "λ",
            Topology::Lens => "σ",
            Topology::Nil => "τ",
            Topology::Other => "rho",
        };

        let mut param_offset: i32 = 0;
        let mut local_best = f64::MAX;
        let mut local_best_param = param_center;

        while param_offset < num_param {
            if !running.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }

            let batch_param = batch_size_param.min(num_param - param_offset);
            let batch_param_min = param_min + param_offset as f64 * param_step;

            let threads = batch_param as u64 * num_pairs as u64;
            let num_blocks = ((threads as u32) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            let mut block_results = gpu.alloc_zeros::<GpuCandidate>(num_blocks as usize)?;

            let cfg = LaunchConfig {
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                grid_dim: (num_blocks, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                gpu.stream
                    .launch_builder(kernel_fn)
                    .arg(&batch_param)
                    .arg(&batch_param_min)
                    .arg(&param_step)
                    .arg(&p1_gpu)
                    .arg(&q1_gpu)
                    .arg(&p2_gpu)
                    .arg(&q2_gpu)
                    .arg(&num_pairs)
                    .arg(&mut block_results)
                    .launch(cfg)?;
            }

            let results = gpu.dtoh(&block_results)?;

            for r in &results {
                if r.score < 0.001 && r.score < 1.0e29 {
                    if r.score < local_best {
                        local_best = r.score;
                        local_best_param = r.rho;
                    }
                    let (rp1, rq1) = decode_pair(r.p);
                    let (rp2, rq2) = decode_pair(r.q);
                    collector.add(GeometryCandidate {
                        rho: r.rho,
                        p: r.p,
                        q: r.q,
                        epsilon: 0.0,
                        path_length: r.path_length,
                        ratio: r.ratio,
                        score: r.score,
                        method: format!(
                            "{} ({},{})÷({},{})",
                            method_prefix, rp1, rq1, rp2, rq2
                        ),
                    });
                }
            }

            total_evaluated += threads;
            param_offset += batch_param;
        }

        region_best.push((
            p_enc,
            q_enc,
            local_best_param,
            local_best,
            method_prefix.to_string(),
        ));

        println!(
            "  Refined {}/{}: ({},{})÷({},{}) {}~{:.8}, width={:.1e}, 10M pts, best: {:.6e} ({:.2}s) [{}]",
            idx + 1,
            unique_regions.len(),
            p1, q1, p2, q2,
            param_name,
            param_center,
            param_width,
            local_best,
            region_start.elapsed().as_secs_f64(),
            method_prefix,
        );
    }

    // Phase 2a: Ultra-deep pass on top 5 regions
    region_best.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
    let ultra_deep_regions: Vec<_> = region_best.into_iter().take(5).collect();

    if !ultra_deep_regions.is_empty() && running.load(std::sync::atomic::Ordering::SeqCst) {
        println!(
            "  Ultra-deep refinement on top {} regions (100M pts each)...",
            ultra_deep_regions.len()
        );
    }

    for (ud_idx, (p_enc, q_enc, best_param, prev_score, ref method_prefix)) in
        ultra_deep_regions.iter().enumerate()
    {
        if !running.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }

        let ud_start = Instant::now();
        let topo = classify_topology(method_prefix);
        let (p1, q1) = decode_pair(*p_enc);
        let (p2, q2) = decode_pair(*q_enc);

        let p1s = vec![p1];
        let q1s = vec![q1];
        let p2s = vec![p2];
        let q2s = vec![q2];
        let num_pairs_ud = 1i32;

        let p1_gpu = gpu.htod(&p1s)?;
        let q1_gpu = gpu.htod(&q1s)?;
        let p2_gpu = gpu.htod(&p2s)?;
        let q2_gpu = gpu.htod(&q2s)?;

        let ud_width = 1e-6;
        let ud_param_min = (best_param - ud_width / 2.0).max(0.0001);
        let ud_param_max = best_param + ud_width / 2.0;
        let ud_num_param: i32 = 100_000_000;
        let ud_param_step = (ud_param_max - ud_param_min) / (ud_num_param - 1) as f64;

        let max_threads_per_launch: u64 = 256 * 1024 * 1024;
        let ud_batch_size =
            (max_threads_per_launch / num_pairs_ud as u64).min(ud_num_param as u64) as i32;

        let kernel_fn = match topo {
            Topology::Geodesic => &geodesic_fn,
            Topology::Berger => &berger_fn,
            Topology::Lens => &lens_fn,
            Topology::Poincare
            | Topology::Dihedral
            | Topology::Tetrahedral
            | Topology::Octahedral => &poincare_fn,
            Topology::Nil => &nil_fn,
            Topology::Other => &geodesic_fn,
        };

        let ud_method = match topo {
            Topology::Geodesic => "geodesic-ultradeep",
            Topology::Berger => "berger-ultradeep",
            Topology::Lens => "lens-ultradeep",
            Topology::Poincare => "poincare-ultradeep",
            Topology::Nil => "nil-ultradeep",
            Topology::Dihedral => "dihedral-ultradeep",
            Topology::Tetrahedral => "tetrahedral-ultradeep",
            Topology::Octahedral => "octahedral-ultradeep",
            Topology::Other => "other-ultradeep",
        };

        let mut ud_param_offset: i32 = 0;
        let mut ud_best = *prev_score;

        while ud_param_offset < ud_num_param {
            if !running.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }

            let batch_param = ud_batch_size.min(ud_num_param - ud_param_offset);
            let batch_param_min = ud_param_min + ud_param_offset as f64 * ud_param_step;

            let threads = batch_param as u64 * num_pairs_ud as u64;
            let num_blocks = ((threads as u32) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            let mut block_results = gpu.alloc_zeros::<GpuCandidate>(num_blocks as usize)?;

            let cfg = LaunchConfig {
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                grid_dim: (num_blocks, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                gpu.stream
                    .launch_builder(kernel_fn)
                    .arg(&batch_param)
                    .arg(&batch_param_min)
                    .arg(&ud_param_step)
                    .arg(&p1_gpu)
                    .arg(&q1_gpu)
                    .arg(&p2_gpu)
                    .arg(&q2_gpu)
                    .arg(&num_pairs_ud)
                    .arg(&mut block_results)
                    .launch(cfg)?;
            }

            let results = gpu.dtoh(&block_results)?;

            for r in &results {
                if r.score < 0.001 && r.score < 1.0e29 {
                    if r.score < ud_best {
                        ud_best = r.score;
                    }
                    let (rp1, rq1) = decode_pair(r.p);
                    let (rp2, rq2) = decode_pair(r.q);
                    collector.add(GeometryCandidate {
                        rho: r.rho,
                        p: r.p,
                        q: r.q,
                        epsilon: 0.0,
                        path_length: r.path_length,
                        ratio: r.ratio,
                        score: r.score,
                        method: format!(
                            "{} ({},{})÷({},{})",
                            ud_method, rp1, rq1, rp2, rq2
                        ),
                    });
                }
            }

            total_evaluated += threads;
            ud_param_offset += batch_param;
        }

        println!(
            "    Ultra-deep {}/{}: ({},{})÷({},{}) param~{:.12}, 100M pts, score: {:.6e} -> {:.6e} ({:.2}s) [{}]",
            ud_idx + 1,
            ultra_deep_regions.len(),
            p1, q1, p2, q2,
            best_param,
            prev_score,
            ud_best,
            ud_start.elapsed().as_secs_f64(),
            ud_method,
        );
    }

    // Phase 2b: Helmholtz cross-check
    if running.load(std::sync::atomic::Ordering::SeqCst) {
        println!("  Running Helmholtz eigenvalue cross-check...");
        if let Ok(count) = run_helmholtz_check(gpu, &mut collector) {
            total_evaluated += count;
        }
    }

    // Phase 2c: Cavity resonance cross-check
    if running.load(std::sync::atomic::Ordering::SeqCst) {
        println!("  Running cavity resonance cross-check...");
        if let Ok(count) = run_cavity_check(gpu, &mut collector) {
            total_evaluated += count;
        }
    }

    collector.dedup();

    let elapsed = start.elapsed().as_secs_f64();
    let top = collector.top_n(50);
    let best_score = top.first().map(|c| c.score).unwrap_or(f64::MAX);

    println!(
        "Phase 2 complete: {:.3}s, {} evaluations, best score: {:.6e}",
        elapsed,
        format_number(total_evaluated),
        best_score
    );

    Ok(SearchResult {
        phase: 2,
        candidates: top,
        total_evaluated,
        best_score,
        elapsed_secs: elapsed,
    })
}

/// Helmholtz eigenvalue cross-check.
fn run_helmholtz_check(
    gpu: &gpu::GpuContext,
    collector: &mut ResultCollector,
) -> Result<u64, Box<dyn std::error::Error>> {
    let func = gpu.compile_kernel("kernels/toroidal_helmholtz.cu", "helmholtz_scan")?;

    let mut m1_vals = Vec::new();
    let mut n1_vals = Vec::new();
    let mut m2_vals = Vec::new();
    let mut n2_vals = Vec::new();

    for m2 in 1..=5i32 {
        for n2 in 0..=5i32 {
            for m1 in 1..=30i32 {
                for n1 in 0..=30i32 {
                    if m1 > m2 || (m1 == m2 && n1 > n2) {
                        m1_vals.push(m1);
                        n1_vals.push(n1);
                        m2_vals.push(m2);
                        n2_vals.push(n2);
                    }
                }
            }
        }
    }

    let num_pairs = m1_vals.len() as i32;
    let num_rho: i32 = 50_000;
    let rho_min = 0.001f64;
    let rho_step = (0.999 - rho_min) / (num_rho - 1) as f64;

    let m1_gpu = gpu.htod(&m1_vals)?;
    let n1_gpu = gpu.htod(&n1_vals)?;
    let m2_gpu = gpu.htod(&m2_vals)?;
    let n2_gpu = gpu.htod(&n2_vals)?;

    let total_threads = num_rho as u64 * num_pairs as u64;
    let capped = total_threads.min(256 * 1024 * 1024) as u32;
    let num_blocks = (capped + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    let mut block_results = gpu.alloc_zeros::<GpuCandidate>(num_blocks as usize)?;

    let cfg = LaunchConfig {
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        grid_dim: (num_blocks, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        gpu.stream
            .launch_builder(&func)
            .arg(&num_rho)
            .arg(&rho_min)
            .arg(&rho_step)
            .arg(&m1_gpu)
            .arg(&n1_gpu)
            .arg(&m2_gpu)
            .arg(&n2_gpu)
            .arg(&num_pairs)
            .arg(&mut block_results)
            .launch(cfg)?;
    }

    let results = gpu.dtoh(&block_results)?;

    let mut hits = 0;
    for r in &results {
        if r.score < 1.0 && r.score < 1.0e29 {
            hits += 1;
            collector.add(GeometryCandidate {
                rho: r.rho,
                p: r.p,
                q: r.q,
                epsilon: 0.0,
                path_length: r.path_length,
                ratio: r.ratio,
                score: r.score,
                method: "helmholtz".to_string(),
            });
        }
    }

    println!(
        "  Helmholtz: {} rho x {} mode pairs = {} evals, {} hits",
        format_number(num_rho as u64),
        format_number(num_pairs as u64),
        format_number(total_threads.min(256 * 1024 * 1024)),
        hits
    );

    Ok(total_threads)
}

/// Cavity resonance cross-check (Aspden parameterized + direct mode scan).
fn run_cavity_check(
    gpu: &gpu::GpuContext,
    collector: &mut ResultCollector,
) -> Result<u64, Box<dyn std::error::Error>> {
    let funcs = gpu.compile_kernel_multi(
        "kernels/cavity_resonance.cu",
        &["cavity_scan", "cavity_mode_scan"],
    )?;
    let cavity_scan_fn = &funcs[0];
    let cavity_mode_fn = &funcs[1];

    let mut total_evaluated = 0u64;

    // --- Part A: Aspden parameterized cavity_scan ---
    {
        let c1_list: Vec<f64> = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let c2_list: Vec<f64> = vec![-1.0, -0.5, 0.0, 0.5, 1.0];

        let mut c1_vals = Vec::new();
        let mut c2_vals = Vec::new();
        for &c1 in &c1_list {
            for &c2 in &c2_list {
                c1_vals.push(c1);
                c2_vals.push(c2);
            }
        }
        let num_c = c1_vals.len() as i32;

        let num_rho: i32 = 100_000;
        let rho_min = 0.001f64;
        let rho_max = 0.5f64;
        let rho_step = (rho_max - rho_min) / (num_rho - 1) as f64;

        let g1 = 0.5f64;
        let g2 = 9.0f64 / 8.0;

        let c1_gpu = gpu.htod(&c1_vals)?;
        let c2_gpu = gpu.htod(&c2_vals)?;

        let total_threads = num_rho as u64 * num_c as u64;
        let capped = total_threads.min(256 * 1024 * 1024) as u32;
        let num_blocks = (capped + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        let mut block_results = gpu.alloc_zeros::<GpuCandidate>(num_blocks as usize)?;

        let cfg = LaunchConfig {
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            grid_dim: (num_blocks, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            gpu.stream
                .launch_builder(cavity_scan_fn)
                .arg(&num_rho)
                .arg(&rho_min)
                .arg(&rho_step)
                .arg(&c1_gpu)
                .arg(&c2_gpu)
                .arg(&num_c)
                .arg(&g1)
                .arg(&g2)
                .arg(&mut block_results)
                .launch(cfg)?;
        }

        let results = gpu.dtoh(&block_results)?;
        let mut hits = 0;
        for r in &results {
            if r.score < 1.0 && r.score < 1.0e29 {
                hits += 1;
                collector.add(GeometryCandidate {
                    rho: r.rho,
                    p: r.p,
                    q: r.q,
                    epsilon: 0.0,
                    path_length: r.path_length,
                    ratio: r.ratio,
                    score: r.score,
                    method: "cavity".to_string(),
                });
            }
        }

        total_evaluated += total_threads;
        println!(
            "  Cavity (Aspden): {} rho x {} (c1,c2) pairs = {} evals, {} hits",
            format_number(num_rho as u64),
            format_number(num_c as u64),
            format_number(total_threads.min(256 * 1024 * 1024)),
            hits
        );
    }

    // --- Part B: Direct mode scan cavity_mode_scan ---
    {
        let mut n_vals: Vec<i32> = Vec::new();
        let mut j_vals: Vec<i32> = Vec::new();
        for n in 195..=210i32 {
            for j in 0..=5i32 {
                n_vals.push(n);
                j_vals.push(j);
            }
        }
        let num_nj = n_vals.len() as i32;

        let num_rho: i32 = 100_000;
        let rho_min = 0.001f64;
        let rho_max = 0.5f64;
        let rho_step = (rho_max - rho_min) / (num_rho - 1) as f64;

        let n_gpu = gpu.htod(&n_vals)?;
        let j_gpu = gpu.htod(&j_vals)?;

        let total_threads = num_rho as u64 * num_nj as u64;
        let capped = total_threads.min(256 * 1024 * 1024) as u32;
        let num_blocks = (capped + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        let mut block_results = gpu.alloc_zeros::<GpuCandidate>(num_blocks as usize)?;

        let cfg = LaunchConfig {
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            grid_dim: (num_blocks, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            gpu.stream
                .launch_builder(cavity_mode_fn)
                .arg(&num_rho)
                .arg(&rho_min)
                .arg(&rho_step)
                .arg(&n_gpu)
                .arg(&j_gpu)
                .arg(&num_nj)
                .arg(&mut block_results)
                .launch(cfg)?;
        }

        let results = gpu.dtoh(&block_results)?;
        let mut hits = 0;
        for r in &results {
            if r.score < 1.0 && r.score < 1.0e29 {
                hits += 1;
                collector.add(GeometryCandidate {
                    rho: r.rho,
                    p: r.p,
                    q: r.q,
                    epsilon: 0.0,
                    path_length: r.path_length,
                    ratio: r.ratio,
                    score: r.score,
                    method: "cavity-mode".to_string(),
                });
            }
        }

        total_evaluated += total_threads;
        println!(
            "  Cavity (mode): {} rho x {} (n,j) pairs = {} evals, {} hits",
            format_number(num_rho as u64),
            format_number(num_nj as u64),
            format_number(total_threads.min(256 * 1024 * 1024)),
            hits
        );
    }

    Ok(total_evaluated)
}

// ---------------------------------------------------------------------------
// Phase 3: High-precision CPU analysis
// ---------------------------------------------------------------------------

/// Phase 3: High-precision CPU analysis of best candidates.
fn run_phase3(phase2: &SearchResult) -> Result<SearchResult, Box<dyn std::error::Error>> {
    println!("--- Phase 3: High-Precision CPU Analysis ---");
    let start = Instant::now();

    if phase2.candidates.is_empty() {
        println!("No Phase 2 candidates to analyze.");
        return Ok(SearchResult {
            phase: 3,
            candidates: vec![],
            total_evaluated: 0,
            best_score: f64::MAX,
            elapsed_secs: 0.0,
        });
    }

    use rayon::prelude::*;

    let quadrature_points = 10_000;

    let mut candidates: Vec<GeometryCandidate> = phase2
        .candidates
        .par_iter()
        .filter_map(|c| {
            let topo = classify_topology(&c.method);
            let (p1, q1) = decode_pair(c.p);
            let (p2, q2) = decode_pair(c.q);

            let param_width = 1e-6;
            let param_min = (c.rho - param_width / 2.0).max(1e-10);
            let param_max = c.rho + param_width / 2.0;
            let n_pts = 10_000usize;
            let param_step = (param_max - param_min) / (n_pts - 1) as f64;

            let mut best_param = c.rho;
            let mut best_score = f64::MAX;
            let mut best_ratio = 0.0;
            let mut best_l1 = 0.0;

            match topo {
                Topology::Geodesic => {
                    for i in 0..n_pts {
                        let rho = param_min + i as f64 * param_step;
                        let l1 = physics::path_length_cpu(p1, q1, rho, quadrature_points);
                        let l2 = physics::path_length_cpu(p2, q2, rho, quadrature_points);
                        let ratio = if l2 > 1e-15 { l1 / l2 } else { 0.0 };
                        let s = score(ratio);
                        if s < best_score {
                            best_score = s;
                            best_param = rho;
                            best_ratio = ratio;
                            best_l1 = l1;
                        }
                    }
                    Some(GeometryCandidate {
                        rho: best_param,
                        p: c.p,
                        q: c.q,
                        epsilon: 0.0,
                        path_length: best_l1,
                        ratio: best_ratio,
                        score: best_score,
                        method: format!("geodesic-f64 ({},{})÷({},{})", p1, q1, p2, q2),
                    })
                }
                Topology::Berger => {
                    for i in 0..n_pts {
                        let lambda = param_min + i as f64 * param_step;
                        let l1 =
                            physics::berger_path_length_cpu(p1, q1, lambda, quadrature_points);
                        let l2 =
                            physics::berger_path_length_cpu(p2, q2, lambda, quadrature_points);
                        let ratio = if l2 > 1e-15 { l1 / l2 } else { 0.0 };
                        let s = score(ratio);
                        if s < best_score {
                            best_score = s;
                            best_param = lambda;
                            best_ratio = ratio;
                            best_l1 = l1;
                        }
                    }
                    Some(GeometryCandidate {
                        rho: best_param,
                        p: c.p,
                        q: c.q,
                        epsilon: 0.0,
                        path_length: best_l1,
                        ratio: best_ratio,
                        score: best_score,
                        method: format!("berger-f64 ({},{})÷({},{})", p1, q1, p2, q2),
                    })
                }
                Topology::Lens => {
                    for i in 0..n_pts {
                        let sigma = param_min + i as f64 * param_step;
                        let ratio = physics::lens_path_ratio_cpu(p1, q1, p2, q2, sigma);
                        let s = score(ratio);
                        if s < best_score {
                            best_score = s;
                            best_param = sigma;
                            best_ratio = ratio;
                            let s2 = sigma * sigma;
                            best_l1 = ((p1 as f64).powi(2) * s2
                                + (q1 as f64).powi(2) * (1.0 - s2))
                            .sqrt();
                        }
                    }
                    Some(GeometryCandidate {
                        rho: best_param,
                        p: c.p,
                        q: c.q,
                        epsilon: 0.0,
                        path_length: best_l1,
                        ratio: best_ratio,
                        score: best_score,
                        method: format!("lens-f64 ({},{})÷({},{})", p1, q1, p2, q2),
                    })
                }
                Topology::Poincare
                | Topology::Dihedral
                | Topology::Tetrahedral
                | Topology::Octahedral => {
                    // S³ quotient topologies use Berger path length
                    let prefix = match topo {
                        Topology::Poincare => "poincare-f64",
                        Topology::Dihedral => "dihedral-f64",
                        Topology::Tetrahedral => "tetrahedral-f64",
                        Topology::Octahedral => "octahedral-f64",
                        _ => unreachable!(),
                    };
                    for i in 0..n_pts {
                        let lambda = param_min + i as f64 * param_step;
                        let l1 =
                            physics::berger_path_length_cpu(p1, q1, lambda, quadrature_points);
                        let l2 =
                            physics::berger_path_length_cpu(p2, q2, lambda, quadrature_points);
                        let ratio = if l2 > 1e-15 { l1 / l2 } else { 0.0 };
                        let s = score(ratio);
                        if s < best_score {
                            best_score = s;
                            best_param = lambda;
                            best_ratio = ratio;
                            best_l1 = l1;
                        }
                    }
                    Some(GeometryCandidate {
                        rho: best_param,
                        p: c.p,
                        q: c.q,
                        epsilon: 0.0,
                        path_length: best_l1,
                        ratio: best_ratio,
                        score: best_score,
                        method: format!("{} ({},{})÷({},{})", prefix, p1, q1, p2, q2),
                    })
                }
                Topology::Nil => {
                    for i in 0..n_pts {
                        let tau = param_min + i as f64 * param_step;
                        let ratio = nil_eigenvalue_ratio_cpu(p1, q1, p2, q2, tau);
                        let s = score(ratio);
                        if s < best_score {
                            best_score = s;
                            best_param = tau;
                            best_ratio = ratio;
                            let pi2 = 2.0 * std::f64::consts::PI;
                            best_l1 = (pi2 * p1 as f64 * tau).powi(2)
                                + (pi2 * q1 as f64).powi(2);
                        }
                    }
                    Some(GeometryCandidate {
                        rho: best_param,
                        p: c.p,
                        q: c.q,
                        epsilon: 0.0,
                        path_length: best_l1,
                        ratio: best_ratio,
                        score: best_score,
                        method: format!("nil-f64 ({},{})÷({},{})", p1, q1, p2, q2),
                    })
                }
                Topology::Other => Some(c.clone()),
            }
        })
        .collect();

    // Newton's method refinement
    println!("  Newton refinement on top candidates...");
    let mut newton_refined: Vec<GeometryCandidate> = Vec::new();
    for c in candidates.iter().take(20) {
        let topo = classify_topology(&c.method);
        let (p1, q1) = decode_pair(c.p);
        let (p2, q2) = decode_pair(c.q);

        match topo {
            Topology::Geodesic => {
                if let Some((rho, ratio, residual)) =
                    spectral::newton_geodesic_ratio(p1, q1, p2, q2, c.rho, quadrature_points)
                {
                    let l1 = physics::path_length_cpu(p1, q1, rho, quadrature_points);
                    newton_refined.push(GeometryCandidate {
                        rho,
                        p: c.p,
                        q: c.q,
                        epsilon: 0.0,
                        path_length: l1,
                        ratio,
                        score: residual,
                        method: format!("geodesic-newton ({},{})÷({},{})", p1, q1, p2, q2),
                    });
                }
            }
            Topology::Berger
            | Topology::Poincare
            | Topology::Dihedral
            | Topology::Tetrahedral
            | Topology::Octahedral => {
                let prefix = match topo {
                    Topology::Berger => "berger-newton",
                    Topology::Poincare => "poincare-newton",
                    Topology::Dihedral => "dihedral-newton",
                    Topology::Tetrahedral => "tetrahedral-newton",
                    Topology::Octahedral => "octahedral-newton",
                    _ => unreachable!(),
                };
                if let Some((lambda, ratio, residual)) =
                    spectral::newton_berger_ratio(p1, q1, p2, q2, c.rho, quadrature_points)
                {
                    let l1 =
                        physics::berger_path_length_cpu(p1, q1, lambda, quadrature_points);
                    newton_refined.push(GeometryCandidate {
                        rho: lambda,
                        p: c.p,
                        q: c.q,
                        epsilon: 0.0,
                        path_length: l1,
                        ratio,
                        score: residual,
                        method: format!("{} ({},{})÷({},{})", prefix, p1, q1, p2, q2),
                    });
                }
            }
            _ => {}
        }
    }

    if !newton_refined.is_empty() {
        println!(
            "    Newton refined {} candidates, best residual: {:.6e}",
            newton_refined.len(),
            newton_refined
                .iter()
                .map(|c| c.score)
                .fold(f64::MAX, f64::min)
        );
    }
    candidates.extend(newton_refined);
    candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    if let Some(best) = candidates.first() {
        let (p1, q1) = decode_pair(best.p);
        let (p2, q2) = decode_pair(best.q);
        println!(
            "Best: ({},{})÷({},{}) rho={:.12} ratio={:.12} score={:.6e}",
            p1, q1, p2, q2, best.rho, best.ratio, best.score
        );

        let koide_q = physics::koide_check(
            physics::ELECTRON_MASS,
            physics::MUON_MASS,
            physics::TAU_MASS,
        );
        println!(
            "Koide Q = {:.8} (expected 2/3 = {:.8})",
            koide_q,
            2.0 / 3.0
        );

        check_recognizable_constants(best.rho);
    }

    // Tau mass search
    println!("\nChecking tau/electron ratio for best geometries...");
    let tau_target = physics::TAU_ELECTRON_RATIO;
    let modes = coprime_pairs(30);
    let quadrature_tau = 10_000;
    let mut tau_hits: Vec<(f64, i32, i32, i32, i32, f64, f64, &str)> = Vec::new();

    for c in candidates.iter().take(10) {
        let topo = classify_topology(&c.method);
        let topo_name = match topo {
            Topology::Geodesic => "torus",
            Topology::Berger => "berger",
            Topology::Lens => "lens",
            Topology::Poincare => "poincare",
            Topology::Nil => "nil",
            Topology::Dihedral => "dihedral",
            Topology::Tetrahedral => "tetrahedral",
            Topology::Octahedral => "octahedral",
            Topology::Other => "other",
        };

        for (i, &(pa, qa)) in modes.iter().enumerate() {
            for &(pb, qb) in modes.iter().skip(i + 1) {
                let (ratio_ab, ratio_ba) = match topo {
                    Topology::Geodesic => {
                        let la = physics::path_length_cpu(pa, qa, c.rho, quadrature_tau);
                        let lb = physics::path_length_cpu(pb, qb, c.rho, quadrature_tau);
                        if lb < 1e-15 || la < 1e-15 {
                            continue;
                        }
                        (la / lb, lb / la)
                    }
                    Topology::Berger
                    | Topology::Poincare
                    | Topology::Dihedral
                    | Topology::Tetrahedral
                    | Topology::Octahedral => {
                        let la =
                            physics::berger_path_length_cpu(pa, qa, c.rho, quadrature_tau);
                        let lb =
                            physics::berger_path_length_cpu(pb, qb, c.rho, quadrature_tau);
                        if lb < 1e-15 || la < 1e-15 {
                            continue;
                        }
                        (la / lb, lb / la)
                    }
                    Topology::Lens => {
                        let r = physics::lens_path_ratio_cpu(pa, qa, pb, qb, c.rho);
                        if r < 1e-15 {
                            continue;
                        }
                        (r, 1.0 / r)
                    }
                    Topology::Nil => {
                        let r = nil_eigenvalue_ratio_cpu(pa, qa, pb, qb, c.rho);
                        if r < 1e-15 {
                            continue;
                        }
                        (r, 1.0 / r)
                    }
                    Topology::Other => {
                        continue;
                    }
                };

                let score_ab = (ratio_ab - tau_target).abs();
                let score_ba = (ratio_ba - tau_target).abs();
                if score_ab < 1.0 {
                    tau_hits.push((c.rho, pa, qa, pb, qb, ratio_ab, score_ab, topo_name));
                }
                if score_ba < 1.0 {
                    tau_hits.push((c.rho, pb, qb, pa, qa, ratio_ba, score_ba, topo_name));
                }
            }
        }
    }

    tau_hits.sort_by(|a, b| a.6.partial_cmp(&b.6).unwrap_or(std::cmp::Ordering::Equal));
    if tau_hits.is_empty() {
        println!("  No tau/electron ratio matches found (score < 1.0)");
    } else {
        println!("  Found {} tau/electron ratio candidates:", tau_hits.len());
        for (param, p1, q1, p2, q2, ratio, s, tname) in tau_hits.iter().take(10) {
            println!(
                "    param={:.10} ({},{})÷({},{}) ratio={:.6} score={:.6e} [{}]",
                param, p1, q1, p2, q2, ratio, s, tname
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total = candidates.len() as u64;
    let best_score = candidates.first().map(|c| c.score).unwrap_or(f64::MAX);
    candidates.truncate(50);

    println!(
        "Phase 3 complete: {:.3}s, {} candidates refined",
        elapsed, total
    );

    Ok(SearchResult {
        phase: 3,
        candidates,
        total_evaluated: total,
        best_score,
        elapsed_secs: elapsed,
    })
}

fn check_recognizable_constants(rho: f64) {
    let pi = std::f64::consts::PI;
    let constants: &[(&str, f64)] = &[
        ("1/(2*pi)", 1.0 / (2.0 * pi)),
        ("alpha", physics::ALPHA),
        ("2*alpha", 2.0 * physics::ALPHA),
        ("alpha/pi", physics::ALPHA / pi),
        ("1/pi", 1.0 / pi),
        ("sqrt(alpha)", physics::ALPHA.sqrt()),
        ("1/e", 1.0 / std::f64::consts::E),
        ("1/sqrt(2*pi)", 1.0 / (2.0 * pi).sqrt()),
        ("1/3", 1.0 / 3.0),
        ("1/4", 0.25),
        ("1/7", 1.0 / 7.0),
        ("phi - 1", (1.0 + 5.0_f64.sqrt()) / 2.0 - 1.0),
        ("1/137", 1.0 / 137.0),
        ("pi/1000", pi / 1000.0),
    ];

    println!("Checking rho = {:.12} against known constants:", rho);
    for &(name, val) in constants {
        let diff = (rho - val).abs();
        if diff < 0.01 {
            println!("  ~ {} ({:.12}): diff = {:.6e}", name, val, diff);
        }
    }
}

fn save_results(results: &[SearchResult]) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(results)?;
    std::fs::write("results.json", &json)?;
    println!("Results saved to results.json");

    println!("\n=== Final Summary ===");
    for result in results {
        println!(
            "Phase {}: {} evaluated, best score = {:.6e} ({:.3}s)",
            result.phase,
            format_number(result.total_evaluated),
            result.best_score,
            result.elapsed_secs,
        );
        for (i, c) in result.candidates.iter().take(10).enumerate() {
            let (p1, q1) = decode_pair(c.p);
            let (p2, q2) = decode_pair(c.q);
            println!(
                "  #{:>2}: ({},{})÷({},{}) rho={:.10} ratio={:.10} score={:.6e} [{}]",
                i + 1,
                p1,
                q1,
                p2,
                q2,
                c.rho,
                c.ratio,
                c.score,
                c.method
            );
        }
    }

    Ok(())
}
