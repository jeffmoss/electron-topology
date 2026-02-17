mod gpu;
mod physics;
mod results;
mod search;
mod verify;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use std::time::Instant;

use physics::{coprime_pairs, score, GeometryCandidate, SearchResult, TARGET_RATIO};
use results::{format_number, print_candidate, print_summary, ResultCollector};

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Toroidal Resonance Simulation ===");
    println!("Target muon/electron mass ratio: {:.10}", TARGET_RATIO);
    println!();

    // Verify CPU reference
    print!("Verifying CPU reference... ");
    if verify::verify_known_values() {
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

    // ===== Phase 1: All-Pairs Geodesic Scan =====
    let phase1 = run_phase1_allpairs(&gpu, &running)?;
    print_summary(&phase1);
    println!();

    if !running.load(std::sync::atomic::Ordering::SeqCst) {
        save_results(&[phase1])?;
        return Ok(());
    }

    // ===== Phase 2: Fine-grain refinement + Helmholtz cross-check =====
    let phase2 = run_phase2(&gpu, &running, &phase1)?;
    print_summary(&phase2);
    println!();

    if !running.load(std::sync::atomic::Ordering::SeqCst) {
        save_results(&[phase1, phase2])?;
        return Ok(());
    }

    // ===== Phase 3: High-precision CPU analysis =====
    let phase3 = run_phase3(&phase2)?;
    print_summary(&phase3);
    println!();

    save_results(&[phase1, phase2, phase3])?;

    Ok(())
}

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
            // Only keep pairs where the first mode is likely to produce
            // a longer path (higher winding) — this gives ratio > 1
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

/// Phase 1: Massive all-pairs geodesic scan.
fn run_phase1_allpairs(
    gpu: &gpu::GpuContext,
    running: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<SearchResult, Box<dyn std::error::Error>> {
    println!("--- Phase 1: All-Pairs Geodesic Scan ---");
    let start = Instant::now();

    let func = gpu.compile_kernel("kernels/geodesic_path.cu", "geodesic_scan")?;
    println!("Kernel compiled (compute_89 / Ada Lovelace)");

    // Generate all-pairs winding numbers up to 20
    let (p1s, q1s, p2s, q2s) = generate_winding_pairs(20);
    let num_pairs = p1s.len();
    println!("Generated {} winding number pairs", format_number(num_pairs as u64));

    // Upload pair arrays
    let p1_gpu = gpu.htod(&p1s)?;
    let q1_gpu = gpu.htod(&q1s)?;
    let p2_gpu = gpu.htod(&p2s)?;
    let q2_gpu = gpu.htod(&q2s)?;

    let mut collector = ResultCollector::new(200);
    let mut total_evaluated = 0u64;
    let mut batch_num = 0u32;

    // Scan in multiple rho bands with increasing resolution
    let rho_bands: Vec<(f64, f64, u32, &str)> = vec![
        // (rho_min, rho_max, num_points, description)
        (0.001, 0.01, 100_000, "ultra-thin tori"),
        (0.01, 0.1, 100_000, "thin tori"),
        (0.1, 0.5, 50_000, "moderate tori"),
        (0.5, 0.999, 50_000, "thick tori"),
        // Fine scan around the theoretically interesting region 1/206.77 ~ 0.00484
        (0.003, 0.007, 500_000, "target region (rho ~ 1/207)"),
        // Fine scan around alpha ~ 0.0073
        (0.005, 0.010, 500_000, "alpha region"),
    ];

    for (rho_min, rho_max, num_rho, desc) in &rho_bands {
        if !running.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }

        let rho_step = (rho_max - rho_min) / (*num_rho as f64 - 1.0);
        let num_rho_i32 = *num_rho as i32;
        let num_pairs_i32 = num_pairs as i32;

        let total_work = *num_rho as u64 * num_pairs as u64;

        // Process in GPU-sized batches if too large
        let max_threads_per_launch: u64 = 256 * 1024 * 1024; // ~268M threads max
        let batch_size_rho =
            (max_threads_per_launch / num_pairs as u64).min(*num_rho as u64) as u32;

        let mut rho_offset = 0u32;

        while rho_offset < *num_rho {
            if !running.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }

            let batch_rho = batch_size_rho.min(*num_rho - rho_offset);
            let batch_rho_min = *rho_min + rho_offset as f64 * rho_step;
            let batch_rho_i32 = batch_rho as i32;

            let threads = batch_rho as u64 * num_pairs as u64;
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
                    .arg(&batch_rho_i32)
                    .arg(&batch_rho_min)
                    .arg(&rho_step)
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
                        method: format!("geodesic ({},{})÷({},{})", p1, q1, p2, q2),
                    });
                }
            }

            total_evaluated += threads;
            rho_offset += batch_rho;
        }

        batch_num += 1;
        let best_so_far = collector
            .top_n(1)
            .first()
            .map(|c| c.score)
            .unwrap_or(f64::MAX);

        println!(
            "  Band {}/{} [{}]: rho=[{:.4},{:.4}] x {} pts x {} pairs = {} evals | best score: {:.6e} ({:.2}s)",
            batch_num,
            rho_bands.len(),
            desc,
            rho_min,
            rho_max,
            format_number(*num_rho as u64),
            format_number(num_pairs as u64),
            format_number(*num_rho as u64 * num_pairs as u64),
            best_so_far,
            start.elapsed().as_secs_f64(),
        );
    }

    let elapsed = start.elapsed().as_secs_f64();
    let top = collector.top_n(50);
    let best_score = top.first().map(|c| c.score).unwrap_or(f64::MAX);

    println!(
        "\nPhase 1 complete: {:.3}s, {} total evaluations, {} candidates with score < 1.0",
        elapsed,
        format_number(total_evaluated),
        top.len()
    );

    if let Some(best) = top.first() {
        let (p1, q1) = decode_pair(best.p);
        let (p2, q2) = decode_pair(best.q);
        println!(
            "  BEST: ({},{})÷({},{}) rho={:.10} ratio={:.10} score={:.6e}",
            p1, q1, p2, q2, best.rho, best.ratio, best.score
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

/// Decode encoded pair: p_encoded = p*1000 + q
fn decode_pair(encoded: i32) -> (i32, i32) {
    (encoded / 1000, encoded % 1000)
}

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

    let func = gpu.compile_kernel("kernels/geodesic_path.cu", "geodesic_scan")?;

    let mut collector = ResultCollector::new(200);
    let mut total_evaluated = 0u64;

    // For each top Phase 1 hit, do an ultra-fine rho scan
    // Use the SAME pair that worked, but also nearby pairs
    let top_hits: Vec<GeometryCandidate> = phase1.candidates.iter().take(20).cloned().collect();

    // Collect unique (pair, rho_region) to avoid redundant scans
    let mut unique_regions: Vec<(i32, i32, f64)> = Vec::new();
    for hit in &top_hits {
        let key = (hit.p, hit.q, (hit.rho * 10000.0).round() / 10000.0);
        if !unique_regions.iter().any(|k| k.0 == key.0 && k.1 == key.1 && (k.2 - key.2).abs() < 0.0001) {
            unique_regions.push(key);
        }
    }

    for (idx, &(p_enc, q_enc, rho_center)) in unique_regions.iter().enumerate() {
        if !running.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }

        let (p1, q1) = decode_pair(p_enc);
        let (p2, q2) = decode_pair(q_enc);

        // Build a small set of pairs: the winning pair + nearby ones
        let mut p1s = vec![p1];
        let mut q1s = vec![q1];
        let mut p2s = vec![p2];
        let mut q2s = vec![q2];

        // Add neighboring winding numbers
        for dp1 in -1..=1i32 {
            for dq1 in -1..=1i32 {
                for dp2 in -1..=1i32 {
                    for dq2 in -1..=1i32 {
                        let np1 = p1 + dp1;
                        let nq1 = q1 + dq1;
                        let np2 = p2 + dp2;
                        let nq2 = q2 + dq2;
                        if np1 >= 0 && nq1 >= 0 && np2 >= 0 && nq2 >= 0
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

        let num_pairs = p1s.len() as i32;
        let p1_gpu = gpu.htod(&p1s)?;
        let q1_gpu = gpu.htod(&q1s)?;
        let p2_gpu = gpu.htod(&p2s)?;
        let q2_gpu = gpu.htod(&q2s)?;

        // Ultra-fine rho scan: 1M points in a narrow window
        let rho_width = 0.0001; // very narrow
        let rho_min = (rho_center - rho_width / 2.0).max(0.0001);
        let rho_max = rho_center + rho_width / 2.0;
        let num_rho: i32 = 1_000_000;
        let rho_step = (rho_max - rho_min) / (num_rho - 1) as f64;

        let total_threads = num_rho as u32 * num_pairs as u32;
        let num_blocks = (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

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
                .arg(&p1_gpu)
                .arg(&q1_gpu)
                .arg(&p2_gpu)
                .arg(&q2_gpu)
                .arg(&num_pairs)
                .arg(&mut block_results)
                .launch(cfg)?;
        }

        let results = gpu.dtoh(&block_results)?;

        let mut local_best = f64::MAX;
        for r in &results {
            if r.score < 0.001 && r.score < 1.0e29 {
                if r.score < local_best {
                    local_best = r.score;
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
                    method: format!("geodesic-fine ({},{})÷({},{})", rp1, rq1, rp2, rq2),
                });
            }
        }

        total_evaluated += total_threads as u64;

        println!(
            "  Refined {}/{}: ({},{})÷({},{}) rho~{:.8}, best score: {:.6e}",
            idx + 1,
            unique_regions.len(),
            p1, q1, p2, q2,
            rho_center,
            local_best,
        );
    }

    // Phase 2b: Helmholtz cross-check
    if running.load(std::sync::atomic::Ordering::SeqCst) {
        println!("  Running Helmholtz eigenvalue cross-check...");
        if let Ok(count) = run_helmholtz_check(gpu, &mut collector) {
            total_evaluated += count;
        }
    }

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

    let quadrature_points = 10_000; // very high precision

    let mut candidates: Vec<GeometryCandidate> = phase2
        .candidates
        .par_iter()
        .filter(|c| c.method.starts_with("geodesic"))
        .map(|c| {
            let (p1, q1) = decode_pair(c.p);
            let (p2, q2) = decode_pair(c.q);
            let l1 = physics::path_length_cpu(p1, q1, c.rho, quadrature_points);
            let l2 = physics::path_length_cpu(p2, q2, c.rho, quadrature_points);
            let ratio = if l2 > 1e-15 { l1 / l2 } else { 0.0 };
            let s = score(ratio);

            GeometryCandidate {
                rho: c.rho,
                p: c.p,
                q: c.q,
                epsilon: 0.0,
                path_length: l1,
                ratio,
                score: s,
                method: format!("geodesic-f64 ({},{})÷({},{})", p1, q1, p2, q2),
            }
        })
        .collect();

    // Also refine Helmholtz candidates
    let helmholtz_refined: Vec<GeometryCandidate> = phase2
        .candidates
        .iter()
        .filter(|c| c.method == "helmholtz")
        .map(|c| c.clone())
        .collect();

    candidates.extend(helmholtz_refined);
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

    let elapsed = start.elapsed().as_secs_f64();
    let total = candidates.len() as u64;
    let best_score = candidates.first().map(|c| c.score).unwrap_or(f64::MAX);
    candidates.truncate(50);

    println!("Phase 3 complete: {:.3}s, {} candidates refined", elapsed, total);

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
                i + 1, p1, q1, p2, q2, c.rho, c.ratio, c.score, c.method
            );
        }
    }

    Ok(())
}
