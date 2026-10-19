#ifndef PHYSICS_CUH
#define PHYSICS_CUH

// Represents an individual photon packet
struct Photon {
    float x, y, z;          // Spatial position
    float ux, uy, uz;       // Direction cosines (trajectory)
    float weight;           // Current energy weight
    bool active;            // True if photon is still propagating
    bool visited_middle;    // Custom flag to isolate middle layer data
};

// Represents the optical properties of a specific structural zone
struct LayerProps {
    float mua;              // Absorption coefficient
    float mus;              // Scattering coefficient
    float g;                // Anisotropy factor
    float n;                // Refractive index
    float z_bottom;         // Z-coordinate of the layer's lower boundary
};

// Declare constant memory for the Treetop and Middle layers
extern __constant__ LayerProps d_layers[2];

// Add this wrapper function declaration
cudaError_t load_layers_to_constant(const LayerProps* h_layers);

// Calculates the distance a photon travels before the next interaction
__host__ __device__ float calc_step_size(float mua, float mus, float xi);

// Applies Russian roulette to low-weight photons to conserve compute cycles
__host__ __device__ void apply_roulette(float& weight, bool& active, float xi);

#endif