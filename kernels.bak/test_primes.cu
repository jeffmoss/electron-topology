// Segmented Sieve of Eratosthenes on GPU
//
// Each block sieves a segment of SEGMENT_SIZE odd numbers using shared memory.
// Phase 1: Each thread marks off multiples of assigned small primes.
// Phase 2: Each thread counts surviving primes in its slice of the bitmap.
//
// This is O(n log log n) vs O(n * sqrt(n)) for trial division — orders of
// magnitude faster for large ranges.

// Segment size per block: 2^16 = 65536 odd numbers = 131072 consecutive integers.
// Stored as a bit-per-odd-number bitmap = 8192 bytes = 8 KB shared memory.
// Well within the 48 KB shared memory limit per SM on Ada Lovelace.
#define SEGMENT_BITS 65536
#define SEGMENT_BYTES (SEGMENT_BITS / 8)  // 8192 bytes

// Small primes up to sqrt(max_n). For n ~ 4e12, sqrt ~ 2e6.
// We precompute these on the host and pass them in.
// Each block uses these to sieve its segment.

extern "C" __global__ void sieve_primes(
    unsigned long long seg_start,   // first odd number in the global range
    unsigned int num_segments,      // how many segments (= gridDim.x)
    const unsigned int *small_primes,  // array of small primes (3, 5, 7, 11, ...)
    unsigned int num_small_primes,
    unsigned int *block_counts,     // output: prime count per block
    unsigned long long *block_max   // output: largest prime per block
) {
    __shared__ unsigned char sieve[SEGMENT_BYTES];  // bit array: 0=prime, 1=composite

    unsigned int bid = blockIdx.x;
    if (bid >= num_segments) return;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    // This block sieves odd numbers in [lo, lo + 2*SEGMENT_BITS)
    // where lo = seg_start + bid * 2 * SEGMENT_BITS
    unsigned long long lo = seg_start + (unsigned long long)bid * 2ULL * SEGMENT_BITS;

    // Phase 0: Clear sieve (all assumed prime)
    // 8192 bytes / 256 threads = 32 bytes per thread
    for (unsigned int i = tid; i < SEGMENT_BYTES; i += block_size) {
        sieve[i] = 0;
    }
    __syncthreads();

    // Phase 1: Mark composites using small primes.
    // Each thread handles a subset of small primes.
    for (unsigned int pidx = tid; pidx < num_small_primes; pidx += block_size) {
        unsigned int p = small_primes[pidx];
        unsigned long long p2 = (unsigned long long)p * p;

        // Find first odd multiple of p >= lo
        unsigned long long first;
        if (p2 >= lo) {
            first = p2;
        } else {
            // first = lo + ((p - lo % p) % p)
            // but we need it to be odd
            unsigned long long r = lo % p;
            first = (r == 0) ? lo : lo + (p - r);
            // Make sure first is odd
            if (first % 2 == 0) first += p;
        }

        // Convert to bit index: bit = (first - lo) / 2
        // Step by p (in odd-number space: step by p in the number line = step by p in bit indices... no)
        // If p is odd, multiples alternate odd/even. Step by 2*p to stay on odd multiples.
        unsigned long long offset = first - lo;
        unsigned int bit = (unsigned int)(offset >> 1);  // /2 since we only store odds

        unsigned int step = p;  // step in bit indices = p (since each bit = 2 numbers, step of 2p numbers = p bits)

        for (; bit < SEGMENT_BITS; bit += step) {
            // Set bit (mark composite) — use atomicOr to avoid races between threads
            unsigned int byte_idx = bit >> 3;
            unsigned char mask = 1 << (bit & 7);
            atomicOr((unsigned int *)(sieve + (byte_idx & ~3u)), (unsigned int)mask << ((byte_idx & 3u) * 8));
        }
    }
    __syncthreads();

    // Phase 2: Count primes and find max in this segment.
    // Each thread scans a slice of the bitmap.
    unsigned int local_count = 0;
    unsigned long long local_max = 0;

    for (unsigned int bit = tid; bit < SEGMENT_BITS; bit += block_size) {
        unsigned int byte_idx = bit >> 3;
        unsigned int bit_in_byte = bit & 7;
        if ((sieve[byte_idx] & (1 << bit_in_byte)) == 0) {
            // This position is prime
            unsigned long long prime_val = lo + 2ULL * bit;
            local_count++;
            if (prime_val > local_max) local_max = prime_val;
        }
    }

    // Warp-level reduction for count
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, offset);
        unsigned long long other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        if (other > local_max) local_max = other;
    }

    // Lane 0 of each warp writes to shared memory for block reduction
    __shared__ unsigned int warp_counts[8];   // up to 256/32 = 8 warps
    __shared__ unsigned long long warp_max[8];

    unsigned int warp_id = tid >> 5;
    unsigned int lane = tid & 31;

    if (lane == 0) {
        warp_counts[warp_id] = local_count;
        warp_max[warp_id] = local_max;
    }
    __syncthreads();

    // Thread 0 does final reduction across warps
    if (tid == 0) {
        unsigned int total = 0;
        unsigned long long mx = 0;
        unsigned int num_warps = (block_size + 31) >> 5;
        for (unsigned int w = 0; w < num_warps; w++) {
            total += warp_counts[w];
            if (warp_max[w] > mx) mx = warp_max[w];
        }
        block_counts[bid] = total;
        block_max[bid] = mx;
    }
}
