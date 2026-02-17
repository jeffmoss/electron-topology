// Berger sphere (squashed S^3) path-length scan
//
// Each thread evaluates one (lambda, pair_idx) where pair_idx indexes into
// a precomputed table of (p1,q1,p2,q2) winding number pairs.
//
// On the Berger sphere with Hopf fiber scaled by lambda, a (p,q) closed curve
// has path length:
//   L(p, q, lambda) = integral_0^{2pi} sqrt( q^2 + lambda^2*p^2 + 2*lambda^2*p*q*cos(q*t) ) dt
//
// Special cases:
//   L(1, 0, lambda) = 2*pi*lambda   (pure Hopf fiber)
//   L(0, 1, lambda) = 2*pi          (base S^2 great circle)
//
// Block-level reduction selects the single best candidate per block.

#include "common.cuh"

// ---------------------------------------------------------------------------
// Device function: Berger sphere path-length integrand
//
// integrand(t) = sqrt( q^2 + lambda^2*p^2 + 2*lambda^2*p*q*cos(q*t) )
// ---------------------------------------------------------------------------
__device__ double berger_integrand(double t, int p, int q, double lambda) {
    double pf = (double)p;
    double qf = (double)q;
    double lam2 = lambda * lambda;
    double cos_qt = cos(qf * t);
    return sqrt(qf * qf + lam2 * pf * pf + 2.0 * lam2 * pf * qf * cos_qt);
}

// ---------------------------------------------------------------------------
// Device function: compute Berger sphere path length via 64-point GL quadrature
// ---------------------------------------------------------------------------
__device__ double compute_berger_path_length(int p, int q, double lambda) {
    double sum = 0.0;
    for (int i = 0; i < 64; i++) {
        sum += GL_WEIGHTS[i] * berger_integrand(GL_NODES[i], p, q, lambda);
    }
    return sum;
}

extern "C" __global__ void berger_scan(
    int          num_lambda,    // number of lambda grid points
    double       lambda_min,    // smallest lambda value
    double       lambda_step,   // spacing between lambda values
    const int   *p1_vals,       // numerator p winding
    const int   *q1_vals,       // numerator q winding
    const int   *p2_vals,       // denominator p winding
    const int   *q2_vals,       // denominator q winding
    int          num_pairs,     // number of (p1,q1,p2,q2) pairs
    CandidateResult *block_results  // output: one per block
) {
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    unsigned int total_work = (unsigned int)num_lambda * (unsigned int)num_pairs;

    // Default: worst possible score so non-participating threads lose reduction
    double score       = 1.0e30;
    double rho         = 0.0;   // stores lambda in the rho field
    int    my_p        = 0;
    int    my_q        = 0;
    double path_length = 0.0;
    double ratio       = 0.0;

    if (gid < total_work) {
        // Decode linear index into (lambda_idx, pair_idx)
        int lambda_idx = gid / num_pairs;
        int pair_idx   = gid % num_pairs;

        double lambda = lambda_min + (double)lambda_idx * lambda_step;

        int p1 = p1_vals[pair_idx];
        int q1 = q1_vals[pair_idx];
        int p2 = p2_vals[pair_idx];
        int q2 = q2_vals[pair_idx];

        // Compute both path lengths via GL quadrature
        double L1 = compute_berger_path_length(p1, q1, lambda);
        double L2 = compute_berger_path_length(p2, q2, lambda);

        path_length = L1;  // store numerator path for diagnostics
        rho = lambda;      // store lambda in the rho field

        // Ratio: L1 / L2 (longer path mode / shorter path mode)
        if (L2 > 1.0e-15) {
            ratio = L1 / L2;
        }

        score = fabs(ratio - TARGET_RATIO);

        // Encode pair info: p = p1*1000 + q1, q = p2*1000 + q2
        my_p = p1 * 1000 + q1;
        my_q = p2 * 1000 + q2;
    }

    // Block-level reduction to find the best candidate in this block
    block_reduce_min_candidate(score, rho, my_p, my_q, path_length, ratio,
                               block_results);
}
