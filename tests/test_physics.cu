#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../include/physics.cuh"

TEST(PhysicsSuite, LoadConstantMemorySuccess) {
    LayerProps h_layers[2];
    
    // Test data
    h_layers[0] = {0.1f, 150.0f, 0.9f, 1.4f, 0.02f};
    h_layers[1] = {0.1f, 200.0f, 0.8f, 1.4f, 0.08f};
    
    // Execute wrapper
    cudaError_t err = load_layers_to_constant(h_layers);
    
    // Assert success
    EXPECT_EQ(err, cudaSuccess);
}

TEST(PhysicsSuite, PhotonStructSizeAndAlignment) {
    // 7 floats (28 bytes) + 2 bools (2 bytes) + 2 bytes compiler padding = 32 bytes
    // Ensures the struct aligns on a 4-byte boundary for optimized GPU memory access
    EXPECT_EQ(sizeof(Photon), 32); 
}

TEST(PhysicsSuite, CalculateMeanFreePath) {
    float mua = 0.1f;
    float mus = 99.9f;
    float xi = 0.367879f; // Approximately 1/e
    
    // Formula: -ln(xi) / (mua + mus)
    // -ln(1/e) / 100.0 = 1.0 / 100.0 = 0.01
    float step = calc_step_size(mua, mus, xi);
    
    // Use EXPECT_NEAR for floating point comparisons
    EXPECT_NEAR(step, 0.01f, 1e-5f);
}

TEST(PhysicsSuite, RussianRouletteTermination) {
    float weight = 0.005f; // Below typical threshold
    float xi_survive = 0.05f; // 5% chance to survive (typically 1 in 10 chance)
    float xi_die = 0.15f;     // Rolls above the survival probability
    
    // Test fatal roll
    float w_die = weight;
    bool active_die = true;
    apply_roulette(w_die, active_die, xi_die);
    EXPECT_FALSE(active_die);
    EXPECT_EQ(w_die, 0.0f);
    
    // Test survival roll (weight should be multiplied by 10)
    float w_survive = weight;
    bool active_survive = true;
    apply_roulette(w_survive, active_survive, xi_survive);
    EXPECT_TRUE(active_survive);
    EXPECT_NEAR(w_survive, 0.05f, 1e-5f);
}
