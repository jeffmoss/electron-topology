#ifndef COMMON_CUH
#define COMMON_CUH

// ---------------------------------------------------------------------------
// Mathematical constants
// ---------------------------------------------------------------------------
#define PI          3.14159265358979323846
#define TWO_PI      6.28318530717958647692
#define ALPHA       7.2973525693e-3         // fine structure constant 1/137.035999084
#define ALPHA_INV   137.035999084           // 1/alpha
#define TARGET_RATIO 206.7682843            // muon/electron mass ratio (CODATA)
#define ASPDEN_VALUE 206.7683078            // Aspden's theoretical prediction

// ---------------------------------------------------------------------------
// Candidate result structure -- one per block from reduction kernels
// ---------------------------------------------------------------------------
struct CandidateResult {
    double score;        // |ratio - TARGET_RATIO|, lower is better
    double rho;          // torus aspect-ratio parameter
    int    p;            // poloidal winding number
    int    q;            // toroidal winding number
    double path_length;  // computed geodesic path length
    double ratio;        // the actual ratio obtained
};

// ---------------------------------------------------------------------------
// 64-point Gauss-Legendre quadrature on [0, 2*pi]
//
// Nodes  t_i = pi*(x_i + 1)   where x_i are standard GL nodes on [-1,1]
// Weights w_i = pi * w_i(std)
// ---------------------------------------------------------------------------
__constant__ double GL_NODES[64] = {
    2.183275777449663821e-03,
    1.149786226222197956e-02,
    2.823232614920063999e-02,
    5.235070237734448101e-02,
    8.379624186315109968e-02,
    1.224944685677556711e-01,
    1.683536310608191588e-01,
    2.212649714165459136e-01,
    2.811029968130900891e-01,
    3.477257817276121488e-01,
    4.209753063357126734e-01,
    5.006778320903286783e-01,
    5.866443141806242378e-01,
    6.786708501159314233e-01,
    7.765391634639565721e-01,
    8.800171216380322514e-01,
    9.888592865256582431e-01,
    1.102807496662723707e+00,
    1.221591479578141515e+00,
    1.344929492859610720e+00,
    1.472528992421769489e+00,
    1.604087326392780621e+00,
    1.739292452974175562e+00,
    1.877823680571711673e+00,
    2.019352428441992142e+00,
    2.163543006050870687e+00,
    2.310053409295199423e+00,
    2.458536131699462501e+00,
    2.608638988663278813e+00,
    2.760005952804795992e+00,
    2.912277998418658420e+00,
    3.065093953045617159e+00,
    3.218091354133969073e+00,
    3.370907308760927812e+00,
    3.523179354374789796e+00,
    3.674546318516307419e+00,
    3.824649175480123731e+00,
    3.973131897884386809e+00,
    4.119642301128715545e+00,
    4.263832878737594534e+00,
    4.405361626607875003e+00,
    4.543892854205410892e+00,
    4.679097980786805167e+00,
    4.810656314757816965e+00,
    4.938255814319975734e+00,
    5.061593827601445383e+00,
    5.180377810516862525e+00,
    5.294326020653928211e+00,
    5.403168185541553648e+00,
    5.506646143715629549e+00,
    5.604514457063654476e+00,
    5.696540992998961883e+00,
    5.782507475089257554e+00,
    5.862210000843873559e+00,
    5.935459525451974194e+00,
    6.002082310366495754e+00,
    6.061920335763040235e+00,
    6.114831676118767767e+00,
    6.160690838611830422e+00,
    6.199389065316434966e+00,
    6.230834604802241827e+00,
    6.254952981030385217e+00,
    6.271687444917364296e+00,
    6.281002031402136865e+00
};

__constant__ double GL_WEIGHTS[64] = {
    5.602341614569299362e-03,
    1.302828922557574230e-02,
    2.043435737092602622e-02,
    2.779291567857825052e-02,
    3.508574488221406773e-02,
    4.229541236725909564e-02,
    4.940478181360866833e-02,
    5.639697840026274694e-02,
    6.325541250046021191e-02,
    6.996381443058452554e-02,
    7.650627148491954965e-02,
    8.286726507710293066e-02,
    8.903170729009736439e-02,
    9.498497654246614019e-02,
    1.007129522089097312e-01,
    1.062020480809781203e-01,
    1.114392445743920823e-01,
    1.164121195998436692e-01,
    1.211088780206999727e-01,
    1.255183796259826401e-01,
    1.296301655513103779e-01,
    1.334344830846052044e-01,
    1.369223087974038855e-01,
    1.400853699467260460e-01,
    1.429161640966170321e-01,
    1.454079769127552824e-01,
    1.475548980878676752e-01,
    1.493518353601522852e-01,
    1.507945265914351907e-01,
    1.518795498764020646e-01,
    1.526043316589172361e-01,
    1.529671528361698229e-01,
    1.529671528361698229e-01,
    1.526043316589172361e-01,
    1.518795498764020646e-01,
    1.507945265914351907e-01,
    1.493518353601522852e-01,
    1.475548980878676752e-01,
    1.454079769127552824e-01,
    1.429161640966170321e-01,
    1.400853699467260460e-01,
    1.369223087974038855e-01,
    1.334344830846052044e-01,
    1.296301655513103779e-01,
    1.255183796259826401e-01,
    1.211088780206999727e-01,
    1.164121195998436692e-01,
    1.114392445743920823e-01,
    1.062020480809781203e-01,
    1.007129522089097312e-01,
    9.498497654246614019e-02,
    8.903170729009736439e-02,
    8.286726507710293066e-02,
    7.650627148491954965e-02,
    6.996381443058452554e-02,
    6.325541250046021191e-02,
    5.639697840026274694e-02,
    4.940478181360866833e-02,
    4.229541236725909564e-02,
    3.508574488221406773e-02,
    2.779291567857825052e-02,
    2.043435737092602622e-02,
    1.302828922557574230e-02,
    5.602341614569299362e-03
};

// ---------------------------------------------------------------------------
// Device function: geodesic path-length integrand on a torus
//
// A torus parameterized by (theta, phi) with major radius R=1 and
// inverse aspect ratio rho (= r/R, where r is the minor radius).
//
// A (p,q) curve winds p times poloidally and q times toroidally:
//   theta(t) = p*t,  phi(t) = q*t,  t in [0, 2*pi]
//
// ds/dt = sqrt( rho^2 * p^2  +  (1 + rho*cos(p*t))^2 * q^2 )
//
// NOTE: The integrand uses the TORUS metric:
//   ds^2 = rho^2 * dtheta^2  +  (1 + rho*cos(theta))^2 * dphi^2
// ---------------------------------------------------------------------------
__device__ double path_length_integrand(double t, int p, int q, double rho) {
    double cos_pt = cos((double)p * t);
    double term_phi = (1.0 + rho * cos_pt) * (double)q;
    double term_theta = rho * (double)p;
    return sqrt(term_theta * term_theta + term_phi * term_phi);
}

// ---------------------------------------------------------------------------
// Device function: compute path length via 64-point GL quadrature
// ---------------------------------------------------------------------------
__device__ double compute_path_length(int p, int q, double rho) {
    double sum = 0.0;
    for (int i = 0; i < 64; i++) {
        sum += GL_WEIGHTS[i] * path_length_integrand(GL_NODES[i], p, q, rho);
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Warp-level reduction: find minimum score, carrying associated CandidateResult
//
// After this call, lane 0 of the warp holds the best (lowest-score) candidate.
// ---------------------------------------------------------------------------
__device__ void warp_reduce_min_candidate(
    double &score, double &rho, int &p, int &q,
    double &path_length, double &ratio
) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other_score  = __shfl_down_sync(0xFFFFFFFF, score, offset);
        double other_rho    = __shfl_down_sync(0xFFFFFFFF, rho, offset);
        int    other_p      = __shfl_down_sync(0xFFFFFFFF, p, offset);
        int    other_q      = __shfl_down_sync(0xFFFFFFFF, q, offset);
        double other_plen   = __shfl_down_sync(0xFFFFFFFF, path_length, offset);
        double other_ratio  = __shfl_down_sync(0xFFFFFFFF, ratio, offset);
        if (other_score < score) {
            score       = other_score;
            rho         = other_rho;
            p           = other_p;
            q           = other_q;
            path_length = other_plen;
            ratio       = other_ratio;
        }
    }
}

// ---------------------------------------------------------------------------
// Block-level reduction: collect warp-level winners into a single best per block.
//
// Writes the result into block_results[blockIdx.x].
//
// Shared memory arrays must be declared by the caller and passed in, or we can
// use a macro / inline approach.  Here we use a simple pattern: call after
// warp_reduce, with lane 0 writing to shared, then thread 0 reducing.
// ---------------------------------------------------------------------------
__device__ void block_reduce_min_candidate(
    double score, double rho, int p, int q,
    double path_length, double ratio,
    CandidateResult *block_results
) {
    unsigned int tid  = threadIdx.x;
    unsigned int bid  = blockIdx.x;
    unsigned int lane = tid & 31;
    unsigned int warp_id = tid >> 5;
    unsigned int block_size = blockDim.x;

    // Warp-level reduction first
    warp_reduce_min_candidate(score, rho, p, q, path_length, ratio);

    // Shared memory for inter-warp reduction (up to 8 warps = 256 threads)
    __shared__ double  ws_score[8];
    __shared__ double  ws_rho[8];
    __shared__ int     ws_p[8];
    __shared__ int     ws_q[8];
    __shared__ double  ws_plen[8];
    __shared__ double  ws_ratio[8];

    if (lane == 0) {
        ws_score[warp_id] = score;
        ws_rho[warp_id]   = rho;
        ws_p[warp_id]     = p;
        ws_q[warp_id]     = q;
        ws_plen[warp_id]  = path_length;
        ws_ratio[warp_id] = ratio;
    }
    __syncthreads();

    // Thread 0 performs final reduction across warps
    if (tid == 0) {
        unsigned int num_warps = (block_size + 31) >> 5;
        double best_score = ws_score[0];
        double best_rho   = ws_rho[0];
        int    best_p     = ws_p[0];
        int    best_q     = ws_q[0];
        double best_plen  = ws_plen[0];
        double best_ratio = ws_ratio[0];

        for (unsigned int w = 1; w < num_warps; w++) {
            if (ws_score[w] < best_score) {
                best_score = ws_score[w];
                best_rho   = ws_rho[w];
                best_p     = ws_p[w];
                best_q     = ws_q[w];
                best_plen  = ws_plen[w];
                best_ratio = ws_ratio[w];
            }
        }

        block_results[bid].score       = best_score;
        block_results[bid].rho         = best_rho;
        block_results[bid].p           = best_p;
        block_results[bid].q           = best_q;
        block_results[bid].path_length = best_plen;
        block_results[bid].ratio       = best_ratio;
    }
}

#endif // COMMON_CUH
