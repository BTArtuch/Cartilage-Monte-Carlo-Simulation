#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "physics.cuh"
#include "rng.cuh"

int main(int argc, char** argv) {
    // 1. Define the tissue optical properties on the host
    LayerProps h_layers[2];
    
    // Index 0: Horizontal "Treetop" Layer (Superficial Zone)
    h_layers[0] = {0.1f, 150.0f, 0.9f, 1.4f, 0.02f};
    
    // Index 1: Crossed Fiber Layer (Middle Zone)
    h_layers[1] = {0.1f, 200.0f, 0.8f, 1.4f, 0.08f};
    
    // Copy layer properties to GPU constant memory via wrapper function
    cudaError_t err = load_layers_to_constant(h_layers);
    if (err != cudaSuccess) {
        std::cerr << "Failed to load layers into constant memory: " << cudaGetErrorString(err) << "\n";
        return -1;
    }
    std::cout << "Successfully loaded structural layers into constant memory.\n";

    // Allocate device memory for the Photon array
    int num_photons = 100000;
    Photon* d_photons;
    err = cudaMalloc((void**)&d_photons, num_photons * sizeof(Photon));
    
    if (err != cudaSuccess) {
        std::cerr << "CUDA Malloc failed for photons: " << cudaGetErrorString(err) << "\n";
        return -1;
    }
    std::cout << "Successfully allocated memory for " << num_photons << " photons.\n";

    // Allocate device memory for cuRAND states
    curandState* d_states;
    err = cudaMalloc((void**)&d_states, num_photons * sizeof(curandState));
    
    if (err != cudaSuccess) {
        std::cerr << "CUDA Malloc failed for RNG states: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // Define block and grid dimensions for the kernel launch
    int threads_per_block = 256;
    int blocks = (num_photons + threads_per_block - 1) / threads_per_block;
    unsigned long seed = 1234ULL; // Fixed seed for testing reproducibility

    // Launch RNG initialization kernel
    init_rng_kernel<<<blocks, threads_per_block>>>(d_states, seed, num_photons);
    
    // Synchronize and check for kernel launch errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
        return -1;
    }
    std::cout << "Successfully initialized cuRAND states for all threads.\n";

    // Cleanup
    cudaFree(d_photons);
    cudaFree(d_states);
    
    return 0;
}