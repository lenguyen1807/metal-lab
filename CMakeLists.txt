cmake_minimum_required(VERSION 3.20)
project(gemm VERSION 1.0.0 LANGUAGES CXX OBJCXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(metal)
find_package(MLX REQUIRED CONFIG)

add_library(gemm_metal SHARED
    gemm/benchmark.cpp
    gemm/kernel.cpp
    gemm/metal_mgr.cpp
    gemm/mlx_gemm.cpp
    gemm/mps_gemm.mm
)
target_compile_features(gemm_metal PUBLIC cxx_std_20)
target_include_directories(gemm_metal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gemm_metal PUBLIC METAL_CPP "-framework MetalPerformanceShaders"
    PRIVATE mlx)
target_compile_options(gemm_metal PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)
target_compile_definitions(gemm_metal PUBLIC
    KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernels/"
    OUTPUTS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/outputs/"
)

add_executable(gemm main.cpp)
target_link_libraries(gemm PRIVATE gemm_metal)
