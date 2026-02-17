// Phase 2: Toroidal Helmholtz eigenfrequency computation
//
// Solves the scalar Helmholtz equation  nabla^2 psi + k^2 psi = 0  on a torus
// using an inverse-aspect-ratio expansion.  For a torus with inverse aspect
// ratio epsilon = rho (minor radius / major radius), the eigenfrequencies
// can be expanded as:
//
//   k_{m,n}^2 ~ (m^2 / rho^2) + n^2 - m*n/rho + corrections in powers of rho
//
// where m = poloidal mode number, n = toroidal mode number.
//
// We compute eigenfrequency ratios  k_{m1,n1} / k_{m2,n2}  for pairs of
// modes and score them against TARGET_RATIO.

#include "common.cuh"

// ---------------------------------------------------------------------------
// Device function: toroidal Helmholtz eigenfrequency (squared) via
// inverse-aspect-ratio expansion to 3rd order.
//
// k^2_{m,n}(rho) = (m/rho)^2 + n^2 + rho*m*n + rho^2*(m^2 - n^2)/4
//                  + rho^3 * m * n * (m^2 + n^2) / 16
//
// This expansion captures the leading curvature corrections for a thin torus.
// ---------------------------------------------------------------------------
__device__ double helmholtz_k2(int m, int n, double rho) {
    double md = (double)m;
    double nd = (double)n;
    double rho2 = rho * rho;
    double rho3 = rho2 * rho;

    double k2 = (md * md) / (rho2)          // leading poloidal
              + nd * nd                       // toroidal
              + rho * md * nd                 // first-order coupling
              + rho2 * (md*md - nd*nd) / 4.0  // second-order correction
              + rho3 * md * nd * (md*md + nd*nd) / 16.0;  // third-order

    return k2;
}

extern "C" __global__ void helmholtz_scan(
    int          num_rho,        // number of rho grid points
    double       rho_min,        // smallest rho value
    double       rho_step,       // spacing between rho values
    const int   *m1_vals,        // mode m for numerator
    const int   *n1_vals,        // mode n for numerator
    const int   *m2_vals,        // mode m for denominator
    const int   *n2_vals,        // mode n for denominator
    int          num_pairs,      // number of mode pairs
    CandidateResult *block_results
) {
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    unsigned int total_work = (unsigned int)num_rho * (unsigned int)num_pairs;

    double score       = 1.0e30;
    double rho         = 0.0;
    int    my_p        = 0;  // store m1 in p field
    int    my_q        = 0;  // store n1 in q field
    double path_length = 0.0;  // store k2_ratio_raw
    double ratio       = 0.0;

    if (gid < total_work) {
        int rho_idx  = gid / num_pairs;
        int pair_idx = gid % num_pairs;

        rho = rho_min + (double)rho_idx * rho_step;

        int m1 = m1_vals[pair_idx];
        int n1 = n1_vals[pair_idx];
        int m2 = m2_vals[pair_idx];
        int n2 = n2_vals[pair_idx];

        double k2_num = helmholtz_k2(m1, n1, rho);
        double k2_den = helmholtz_k2(m2, n2, rho);

        // Eigenfrequency ratio = sqrt(k2_num / k2_den) or k2_num/k2_den
        // depending on what physical quantity we compare.
        // Use frequency ratio = sqrt(k2_num / k2_den):
        if (k2_den > 1.0e-30 && k2_num > 0.0) {
            double k2_ratio = k2_num / k2_den;
            ratio = sqrt(k2_ratio);
            path_length = k2_ratio;  // store raw ratio-squared for diagnostics
        }

        score = fabs(ratio - TARGET_RATIO);

        // Pack mode numbers for output
        my_p = m1 * 1000 + n1;  // encode (m1, n1) as p
        my_q = m2 * 1000 + n2;  // encode (m2, n2) as q
    }

    block_reduce_min_candidate(score, rho, my_p, my_q, path_length, ratio,
                               block_results);
}
