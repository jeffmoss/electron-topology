// Williamson/van der Mark constrained scan kernel
//
// The electron is fixed as a (p_e, 2) double-loop torus knot.
// For each rho, we compute L_electron once, then test all muon candidate
// modes (p_mu, q_mu) and score L_muon / L_electron against TARGET_RATIO.
//
// Key difference from geodesic_scan: the denominator mode is always (p_e, 2)
// and is a kernel parameter, not part of the mode-pair arrays.

#include "common.cuh"

extern "C" __global__ void williamson_scan(
    int          num_rho,          // number of rho grid points
    double       rho_min,          // smallest rho value
    double       rho_step,         // spacing between rho values
    int          p_electron,       // electron poloidal winding (q_e=2 always)
    const int   *p_muon_vals,      // muon candidate poloidal windings
    const int   *q_muon_vals,      // muon candidate toroidal windings
    int          num_muon_modes,   // number of muon candidate modes
    CandidateResult *block_results // output: one per block
) {
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;
    unsigned int total_work = (unsigned int)num_rho * (unsigned int)num_muon_modes;

    // Default: worst possible score
    double score       = 1.0e30;
    double rho         = 0.0;
    int    my_p        = 0;
    int    my_q        = 0;
    double path_length = 0.0;
    double ratio       = 0.0;

    if (gid < total_work) {
        int rho_idx  = gid / num_muon_modes;
        int mode_idx = gid % num_muon_modes;

        rho = rho_min + (double)rho_idx * rho_step;

        // Electron path length: (p_electron, 2) double-loop
        double L_e = compute_path_length(p_electron, 2, rho);

        // Muon candidate mode
        int p_mu = p_muon_vals[mode_idx];
        int q_mu = q_muon_vals[mode_idx];
        double L_mu = compute_path_length(p_mu, q_mu, rho);

        path_length = L_mu;

        if (L_e > 1.0e-15) {
            ratio = L_mu / L_e;
        }

        score = fabs(ratio - TARGET_RATIO);

        // Encode: p = p_electron*1000 + 2, q = p_mu*1000 + q_mu
        my_p = p_electron * 1000 + 2;
        my_q = p_mu * 1000 + q_mu;
    }

    block_reduce_min_candidate(score, rho, my_p, my_q, path_length, ratio,
                               block_results);
}
