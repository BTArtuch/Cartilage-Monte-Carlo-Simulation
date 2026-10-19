#ifndef RNG_CUH
#define RNG_CUH

#include <curand_kernel.h>

// Kernel declaration to initialize cuRAND states
__global__ void init_rng_kernel(curandState* state, unsigned long seed, int num_photons);

#endif 