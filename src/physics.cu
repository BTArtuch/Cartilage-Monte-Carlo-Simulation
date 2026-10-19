#include "physics.cuh"
#include <math.h>

__host__ __device__ float calc_step_size(float mua, float mus, float xi) {
    // Avoid log(0) which returns -infinity
    if (xi <= 0.0f) xi = 1e-6f; 
    return -logf(xi) / (mua + mus);
}

__host__ __device__ void apply_roulette(float& weight, bool& active, float xi) {
    float threshold = 0.01f; // Typical weight threshold to trigger roulette
    float chance = 0.1f;     // 10% chance to survive
    
    if (weight < threshold) {
        if (xi < chance) {
            weight /= chance; // Conserve energy by boosting survivors
        } else {
            weight = 0.0f;
            active = false;   // Terminate photon
        }
    }
}

__constant__ LayerProps d_layers[2];

// Add this wrapper implementation
cudaError_t load_layers_to_constant(const LayerProps* h_layers) {
    return cudaMemcpyToSymbol(d_layers, h_layers, 2 * sizeof(LayerProps));
}