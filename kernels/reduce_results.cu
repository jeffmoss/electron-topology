// Global reduction kernel: collect the best candidates across all blocks.
//
// Two-pass design:
//   Pass 1 (reduce_best):  Reduce an array of CandidateResults down to one
//                           best-per-block, then the overall best.
//   Pass 2 (extract_top_n): Given the global best score as a threshold,
//                           extract the top N candidates sorted by score.
//
// This kernel operates on the CandidateResult arrays produced by the
// phase kernels (geodesic_scan, helmholtz_scan, cavity_scan).

#include "common.cuh"

// ---------------------------------------------------------------------------
// Pass 1: Reduce an array of CandidateResult to find the single global best.
//
// Input:  candidates[num_candidates]  (one per block from phase kernels)
// Output: output[0]                   (single global best)
//
// Launch with enough blocks to cover num_candidates, then reduce output
// on the host (or re-launch with num_candidates = grid_size).
// ---------------------------------------------------------------------------
extern "C" __global__ void reduce_best(
    const CandidateResult *candidates,
    int                    num_candidates,
    CandidateResult       *output          // one per block
) {
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;

    double score       = 1.0e30;
    double rho         = 0.0;
    int    p           = 0;
    int    q           = 0;
    double path_length = 0.0;
    double ratio       = 0.0;

    if (gid < (unsigned int)num_candidates) {
        score       = candidates[gid].score;
        rho         = candidates[gid].rho;
        p           = candidates[gid].p;
        q           = candidates[gid].q;
        path_length = candidates[gid].path_length;
        ratio       = candidates[gid].ratio;
    }

    block_reduce_min_candidate(score, rho, p, q, path_length, ratio, output);
}

// ---------------------------------------------------------------------------
// Pass 2: Extract the top N candidates (lowest score) from the full array.
//
// Each thread checks one candidate.  If its score is among the top N,
// it atomically claims a slot in the output array.
//
// This is approximate: in rare cases of ties, the output may have fewer
// than N entries or not the exact top N.  For our purposes (displaying
// a handful of best results), this is sufficient.
//
// counter[0] should be initialized to 0 before launch.
// ---------------------------------------------------------------------------
extern "C" __global__ void extract_top_n(
    const CandidateResult *candidates,
    int                    num_candidates,
    double                 score_threshold,  // only extract if score <= this
    CandidateResult       *output,           // output array of size max_n
    int                    max_n,            // maximum number to extract
    unsigned int          *counter            // atomic counter, init to 0
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (unsigned int)num_candidates) return;

    double s = candidates[gid].score;
    if (s <= score_threshold) {
        unsigned int slot = atomicAdd(counter, 1u);
        if (slot < (unsigned int)max_n) {
            output[slot] = candidates[gid];
        }
    }
}

// ---------------------------------------------------------------------------
// Single-block sort of extracted candidates by score (insertion sort).
//
// Call with a single block after extract_top_n.  Thread 0 does the sort.
// n_extracted = min(*counter, max_n) from the extract pass.
// ---------------------------------------------------------------------------
extern "C" __global__ void sort_top_n(
    CandidateResult *results,
    int              n_extracted
) {
    if (threadIdx.x != 0) return;

    // Simple insertion sort -- n_extracted is small (typically <= 64)
    for (int i = 1; i < n_extracted; i++) {
        CandidateResult key = results[i];
        int j = i - 1;
        while (j >= 0 && results[j].score > key.score) {
            results[j + 1] = results[j];
            j--;
        }
        results[j + 1] = key;
    }
}
