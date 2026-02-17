// Lens space L(n,1) geodesic path-length ratio scan
//
// A lens space L(n,1) = S^3 / Z_n is parameterized by shape parameter
// sigma in (0,1).  Closed geodesics with winding numbers (a,b) have length:
//
//   L(a, b, sigma, n) = (2*pi/n) * sqrt(a^2 * sigma^2 + b^2 * (1 - sigma^2))
//
// The ratio L1/L2 is independent of n (the 2*pi/n factor cancels), so we
// only scan over sigma and winding-number pairs:
//
//   ratio = sqrt(a1^2*s^2 + b1^2*(1-s^2)) / sqrt(a2^2*s^2 + b2^2*(1-s^2))
//
// This is a closed-form expression -- no quadrature needed.

#include "common.cuh"

extern "C" __global__ void lens_scan(
    int          num_sigma,     // number of sigma grid points
    double       sigma_min,     // smallest sigma value
    double       sigma_step,    // spacing between sigma values
    const int   *a1_vals,       // numerator winding number a
    const int   *b1_vals,       // numerator winding number b
    const int   *a2_vals,       // denominator winding number a
    const int   *b2_vals,       // denominator winding number b
    int          num_pairs,     // number of (a1,b1,a2,b2) pairs
    CandidateResult *block_results  // output: one per block
) {
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    unsigned int total_work = (unsigned int)num_sigma * (unsigned int)num_pairs;

    // Default: worst possible score so non-participating threads lose reduction
    double score       = 1.0e30;
    double rho         = 0.0;   // stores sigma in this field
    int    my_p        = 0;
    int    my_q        = 0;
    double path_length = 0.0;
    double ratio       = 0.0;

    if (gid < total_work) {
        // Decode linear index into (sigma_idx, pair_idx)
        int sigma_idx = gid / num_pairs;
        int pair_idx  = gid % num_pairs;

        double sigma = sigma_min + (double)sigma_idx * sigma_step;
        double s2 = sigma * sigma;
        double one_minus_s2 = 1.0 - s2;

        int a1 = a1_vals[pair_idx];
        int b1 = b1_vals[pair_idx];
        int a2 = a2_vals[pair_idx];
        int b2 = b2_vals[pair_idx];

        double L1 = sqrt((double)(a1 * a1) * s2 + (double)(b1 * b1) * one_minus_s2);
        double L2 = sqrt((double)(a2 * a2) * s2 + (double)(b2 * b2) * one_minus_s2);

        path_length = L1;  // store numerator path for diagnostics

        if (L2 > 1.0e-15) {
            ratio = L1 / L2;
        }

        score = fabs(ratio - TARGET_RATIO);

        // Encode pair info: p = a1*1000 + b1, q = a2*1000 + b2
        my_p = a1 * 1000 + b1;
        my_q = a2 * 1000 + b2;
        rho  = sigma;
    }

    // Block-level reduction to find the best candidate in this block
    block_reduce_min_candidate(score, rho, my_p, my_q, path_length, ratio,
                               block_results);
}
