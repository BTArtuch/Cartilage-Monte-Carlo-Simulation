cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(CartilageMonteCarlo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Target the GTX 750 Ti (Maxwell architecture)
set(CMAKE_CUDA_ARCHITECTURES 50)

# Suppress the sm_50 deprecation warning from CUDA 12.4
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>")

# Include the headers directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Define the executable using all modular source files
add_executable(mc_sim 
    src/main.cu 
    src/kernels.cu 
    src/physics.cu 
    src/geometry.cu 
    src/rng.cu
)

# Enable Relocatable Device Code (RDC) for multi-file CUDA projects
set_target_properties(mc_sim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link the cuRAND library
target_link_libraries(mc_sim PRIVATE curand)

# --- Test-Driven Development Setup (GTest) ---
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Compile Physics Tests
add_executable(test_physics tests/test_physics.cu src/physics.cu)
set_target_properties(test_physics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_physics PRIVATE gtest_main gtest curand)
add_test(NAME PhysicsTest COMMAND test_physics)

# Compile RNG Tests
add_executable(test_rng tests/test_rng.cu src/rng.cu)
set_target_properties(test_rng PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_rng PRIVATE gtest_main gtest curand)
add_test(NAME RngTest COMMAND test_rng)