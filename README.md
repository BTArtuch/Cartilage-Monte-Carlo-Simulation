# Cartilage Monte Carlo Simulation

A CUDA-accelerated Monte Carlo simulation designed to model the photon scattering properties of phantom cartilage tissue samples. This project models the structural anisotropy of articular cartilage by segmenting the tissue into distinct physical layers (e.g., the horizontal superficial "Treetop" zone and the crossed-fiber middle zone) to evaluate light transport, absorption, and spatial scattering.

## Architecture & Features
* **Multi-Layer Optical Modeling:** Simulates heterogeneous tissue with layer-specific optical properties ($\mu_a$, $\mu_s$, $g$, $n$).
* **CUDA Acceleration:** Maps individual photon lifecycle trajectories (random walks) to parallel GPU threads.
* **Hardware-Accelerated RNG:** Utilizes `cuRAND` to ensure statistically independent, high-performance pseudo-random walks.
* **Separable Device Code:** Employs Relocatable Device Code (RDC) for a modular architecture that separates host management from device physics.
* **Test-Driven Development:** Integrated with Google Test (GTest) for validating fundamental tissue optics math and `__device__` memory operations.

## Prerequisites
* **Operating System:** Ubuntu (configured for 26.04)
* **GPU:** Nvidia GTX 750 Ti (Target Architecture: `sm_50`)
* **CUDA Toolkit:** 12.4 (Note: Compiling for `sm_50` on 12.4 will generate a deprecation warning, which is suppressed via CMake).
* **Build System:** CMake 3.18+
* **Compiler:** C++17 compatible compiler (e.g., GCC 13)

