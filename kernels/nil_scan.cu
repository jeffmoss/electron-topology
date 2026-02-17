// Nil manifold M(a,b,k) abelian eigenvalue ratio scan
//
// On a Nil manifold with twist parameter tau = k/(ab), the abelian
// Laplacian eigenvalues are:
//
//   lambda_{m,n}(tau) = (2*pi*m*tau)^2 + (2*pi*n)^2
//
// This has the same structure as a flat torus with aspect ratio tau.
// The ratio is closed-form (no quadrature), so this kernel is pure
// arithmetic -- similar to lens_space.cu.
//
// Mode pairs are (m1,n1) / (m2,n2) where m,n are integer quantum numbers.

#include "common.cuh"

extern "C" __global__ void nil_scan(
    int          num_tau,       // number of tau grid points
    double       tau_min,       // smallest tau value
    double       tau_step,      // spacing between tau values
    const int   *m1_vals,       // numerator m quantum number
    const int   *n1_vals,       // numerator n quantum number
    const int   *m2_vals,       // denominator m quantum number
    const int   *n2_vals,       // denominator n quantum number
    int          num_pairs,     // number of (m1,n1,m2,n2) pairs
    CandidateResult *block_results  // output: one per block
) {
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    unsigned int total_work = (unsigned int)num_tau * (unsigned int)num_pairs;

    double score       = 1.0e30;
    double rho         = 0.0;   // stores tau in this field
    int    my_p        = 0;
    int    my_q        = 0;
    double path_length = 0.0;
    double ratio       = 0.0;

    if (gid < total_work) {
        int tau_idx  = gid / num_pairs;
        int pair_idx = gid % num_pairs;

        double tau = tau_min + (double)tau_idx * tau_step;

        int m1 = m1_vals[pair_idx];
        int n1 = n1_vals[pair_idx];
        int m2 = m2_vals[pair_idx];
        int n2 = n2_vals[pair_idx];

        // Abelian eigenvalues: E_{m,n}(tau) = (2*pi*m*tau)^2 + (2*pi*n)^2
        double m1f = (double)m1;
        double n1f = (double)n1;
        double m2f = (double)m2;
        double n2f = (double)n2;

        double E1 = (TWO_PI * m1f * tau) * (TWO_PI * m1f * tau)
                   + (TWO_PI * n1f) * (TWO_PI * n1f);
        double E2 = (TWO_PI * m2f * tau) * (TWO_PI * m2f * tau)
                   + (TWO_PI * n2f) * (TWO_PI * n2f);

        path_length = E1;  // store numerator eigenvalue

        if (E2 > 1.0e-15) {
            ratio = E1 / E2;
        }

        score = fabs(ratio - TARGET_RATIO);

        // Encode pair info: p = m1*1000 + n1, q = m2*1000 + n2
        my_p = m1 * 1000 + n1;
        my_q = m2 * 1000 + n2;
        rho  = tau;
    }

    block_reduce_min_candidate(score, rho, my_p, my_q, path_length, ratio,
                               block_results);
}
