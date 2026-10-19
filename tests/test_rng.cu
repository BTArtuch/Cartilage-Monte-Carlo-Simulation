#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../include/rng.cuh"

TEST(RNGSuite, InitKernelLaunchSuccess) {
    int num_photons = 1000;
    curandState* d_states;
    
    ASSERT_EQ(cudaMalloc((void**)&d_states, num_photons * sizeof(curandState)), cudaSuccess);
    
    int threads_per_block = 256;
    int blocks = (num_photons + threads_per_block - 1) / threads_per_block;
    unsigned long seed = 9999ULL;
    
    // Launch kernel
    init_rng_kernel<<<blocks, threads_per_block>>>(d_states, seed, num_photons);
    cudaDeviceSynchronize();
    
    // Assert no launch or execution faults
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    
    cudaFree(d_states);
}