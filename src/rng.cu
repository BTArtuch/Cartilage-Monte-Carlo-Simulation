#include "rng.cuh"

__global__ void init_rng_kernel(curandState* state, unsigned long seed, int num_photons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_photons) {
        // curand_init(seed, sequence_number, offset, state)
        curand_init(seed, idx, 0, &state[idx]);
    }
}