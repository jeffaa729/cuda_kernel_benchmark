cmake_minimum_required(VERSION 3.24)

project(cuda_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

function(apply_cuda_benchmark_options target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
endfunction()

add_library(hpc_cuda STATIC
    src/vector_add.cu
    src/transpose.cu
    src/reduction.cu
    src/gemm.cu
)
target_include_directories(hpc_cuda PUBLIC "${PROJECT_SOURCE_DIR}/include")
apply_cuda_benchmark_options(hpc_cuda)

function(add_cuda_benchmark target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(${target} PRIVATE hpc_cuda)
    apply_cuda_benchmark_options(${target})
endfunction()

add_cuda_benchmark(cuda_benchmarks
    benchmarks/main.cu
    benchmarks/vector_add/vector_add_benchmark.cu
    benchmarks/transpose/transpose_benchmark.cu
    benchmarks/reduction/reduction_benchmark.cu
    benchmarks/gemm/gemm_benchmark.cu
)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME vector_add_smoke
             COMMAND cuda_benchmarks vector_add 1048576)
    add_test(NAME transpose_smoke
             COMMAND cuda_benchmarks transpose 128)
    add_test(NAME reduction_smoke
             COMMAND cuda_benchmarks reduction 1048576)
    add_test(NAME gemm_smoke
             COMMAND cuda_benchmarks gemm 64)
endif()
