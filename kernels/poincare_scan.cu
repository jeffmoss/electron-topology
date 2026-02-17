// Poincare homology sphere S^3/2I and binary polyhedral quotient scan
//
// These quotient spaces use the same Berger sphere path-length formula
// as berger_sphere.cu, but with mode pairs pre-filtered on the CPU side
// by group-theoretic selection rules (icosahedral, dihedral, etc.).
//
// The kernel is identical to berger_scan -- the physics difference is
// entirely in which (p1,q1,p2,q2) pairs are uploaded.

#include "common.cuh"

// berger_integrand and compute_berger_path_length are in common.cuh

extern "C" __global__ void poincare_scan(
    int          num_lambda,    // number of lambda grid points
    double       lambda_min,    // smallest lambda value
    double       lambda_step,   // spacing between lambda values
    const int   *p1_vals,       // numerator p winding (pre-filtered by selection rule)
    const int   *q1_vals,       // numerator q winding
    const int   *p2_vals,       // denominator p winding
    const int   *q2_vals,       // denominator q winding
    int          num_pairs,     // number of (p1,q1,p2,q2) pairs
    CandidateResult *block_results  // output: one per block
) {
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    unsigned int total_work = (unsigned int)num_lambda * (unsigned int)num_pairs;

    double score       = 1.0e30;
    double rho         = 0.0;
    int    my_p        = 0;
    int    my_q        = 0;
    double path_length = 0.0;
    double ratio       = 0.0;

    if (gid < total_work) {
        int lambda_idx = gid / num_pairs;
        int pair_idx   = gid % num_pairs;

        double lambda = lambda_min + (double)lambda_idx * lambda_step;

        int p1 = p1_vals[pair_idx];
        int q1 = q1_vals[pair_idx];
        int p2 = p2_vals[pair_idx];
        int q2 = q2_vals[pair_idx];

        double L1 = compute_berger_path_length(p1, q1, lambda);
        double L2 = compute_berger_path_length(p2, q2, lambda);

        path_length = L1;
        rho = lambda;

        if (L2 > 1.0e-15) {
            ratio = L1 / L2;
        }

        score = fabs(ratio - TARGET_RATIO);

        my_p = p1 * 1000 + q1;
        my_q = p2 * 1000 + q2;
    }

    block_reduce_min_candidate(score, rho, my_p, my_q, path_length, ratio,
                               block_results);
}
