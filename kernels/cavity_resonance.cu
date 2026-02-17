// Phase 2: Aspden-style cavity resonance computation
//
// Parameterized generalization of Aspden's approach to the muon/electron
// mass ratio.  Aspden's formula relates the ratio to cavity modes of a
// toroidal resonator, incorporating the fine structure constant alpha.
//
// Aspden's key insight: the muon is an electron confined to a toroidal
// cavity whose resonance condition involves alpha.  His published value
// for the mass ratio is 206.7683078.
//
// We parameterize:
//
//   ratio(n, j, rho) = n * (1 + alpha/pi)^j * f(rho)
//
// where:
//   n     = principal cavity mode number
//   j     = order of radiative correction (0, 1, 2, ...)
//   rho   = aspect ratio parameter
//   f(rho) = geometric correction factor for toroidal cavity
//
// The geometric factor for a toroidal cavity with inverse aspect ratio rho:
//   f(rho) = 1 + rho^2/2 + (9/8)*rho^4 + ...
//
// We also try Aspden's specific formula:
//   ratio = (3/2) * alpha_inv * (1 + alpha/(2*pi)) * g(rho)
//
// and score against both TARGET_RATIO and ASPDEN_VALUE.

#include "common.cuh"

// ---------------------------------------------------------------------------
// Aspden-style cavity resonance ratio
//
// Formula: ratio = base * correction * geometric
//
// base       = (3/2) * alpha_inv = (3/2) * 137.035999084 = 205.553998626
// correction = 1 + c1*alpha/pi + c2*(alpha/pi)^2
// geometric  = 1 + g1*rho^2 + g2*rho^4
//
// The free parameters are c1, c2 (radiative correction coefficients)
// and g1, g2 (geometric factors), plus rho.
// ---------------------------------------------------------------------------
__device__ double aspden_ratio(double rho, double c1, double c2,
                               double g1, double g2) {
    double base = 1.5 * ALPHA_INV;  // 205.553998626

    double a_over_pi = ALPHA / PI;
    double correction = 1.0 + c1 * a_over_pi + c2 * a_over_pi * a_over_pi;

    double rho2 = rho * rho;
    double rho4 = rho2 * rho2;
    double geometric = 1.0 + g1 * rho2 + g2 * rho4;

    return base * correction * geometric;
}

extern "C" __global__ void cavity_scan(
    int          num_rho,        // number of rho grid points
    double       rho_min,        // smallest rho
    double       rho_step,       // rho spacing
    const double *c1_vals,       // radiative correction c1 values
    const double *c2_vals,       // radiative correction c2 values
    int          num_c,          // number of (c1,c2) pairs
    double       g1,             // geometric coefficient (fixed per launch)
    double       g2,             // geometric coefficient (fixed per launch)
    CandidateResult *block_results
) {
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    unsigned int total_work = (unsigned int)num_rho * (unsigned int)num_c;

    double score       = 1.0e30;
    double rho         = 0.0;
    int    my_p        = 0;  // encode c-index
    int    my_q        = 0;
    double path_length = 0.0;
    double ratio       = 0.0;

    if (gid < total_work) {
        int rho_idx = gid / num_c;
        int c_idx   = gid % num_c;

        rho = rho_min + (double)rho_idx * rho_step;
        double c1 = c1_vals[c_idx];
        double c2 = c2_vals[c_idx];

        ratio = aspden_ratio(rho, c1, c2, g1, g2);

        // Score against both targets; use the better one
        double score_codata = fabs(ratio - TARGET_RATIO);
        double score_aspden = fabs(ratio - ASPDEN_VALUE);
        score = (score_codata < score_aspden) ? score_codata : score_aspden;

        // Store the parameters for diagnostics
        my_p = c_idx;
        my_q = rho_idx;
        path_length = c1 * 1000.0 + c2;  // pack c1, c2 for readback
    }

    block_reduce_min_candidate(score, rho, my_p, my_q, path_length, ratio,
                               block_results);
}

// ---------------------------------------------------------------------------
// Alternative kernel: direct mode-number scan
//
// ratio = n * (1 + alpha/(2*pi))^j * (1 + rho^2/2)
//
// Scan over integer n, integer j, and continuous rho.
// ---------------------------------------------------------------------------
extern "C" __global__ void cavity_mode_scan(
    int          num_rho,
    double       rho_min,
    double       rho_step,
    const int   *n_vals,         // principal mode numbers
    const int   *j_vals,         // radiative correction orders
    int          num_nj,         // number of (n, j) pairs
    CandidateResult *block_results
) {
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    unsigned int total_work = (unsigned int)num_rho * (unsigned int)num_nj;

    double score       = 1.0e30;
    double rho         = 0.0;
    int    my_p        = 0;
    int    my_q        = 0;
    double path_length = 0.0;
    double ratio       = 0.0;

    if (gid < total_work) {
        int rho_idx = gid / num_nj;
        int nj_idx  = gid % num_nj;

        rho = rho_min + (double)rho_idx * rho_step;
        int n = n_vals[nj_idx];
        int j = j_vals[nj_idx];

        // Radiative correction factor: (1 + alpha/(2*pi))^j
        double rad_factor = 1.0;
        double a_2pi = ALPHA / TWO_PI;
        for (int k = 0; k < j; k++) {
            rad_factor *= (1.0 + a_2pi);
        }

        // Geometric factor
        double rho2 = rho * rho;
        double geom = 1.0 + 0.5 * rho2 + (9.0/8.0) * rho2 * rho2;

        ratio = (double)n * rad_factor * geom;
        score = fabs(ratio - TARGET_RATIO);

        my_p = n;
        my_q = j;
        path_length = rad_factor * geom;
    }

    block_reduce_min_candidate(score, rho, my_p, my_q, path_length, ratio,
                               block_results);
}
