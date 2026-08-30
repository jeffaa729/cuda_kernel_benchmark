cmake_minimum_required(VERSION 3.24)

project(cuda_benchmark LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

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
    src/softmax.cu
    src/conv2d.cu
    src/rmsnorm.cu
    src/swiglu.cu
    src/rope.cu
    src/embedding.cu
    src/adamw.cu
    src/global_norm.cu
    src/cross_entropy.cu
    src/attention.cu
)
target_include_directories(hpc_cuda PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(hpc_cuda PUBLIC CUDA::cublas)
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
    benchmarks/softmax/softmax_benchmark.cu
    benchmarks/conv2d/conv2d_benchmark.cu
    benchmarks/rmsnorm/rmsnorm_benchmark.cu
    benchmarks/swiglu/swiglu_benchmark.cu
    benchmarks/rope/rope_benchmark.cu
    benchmarks/embedding/embedding_benchmark.cu
    benchmarks/adamw/adamw_benchmark.cu
    benchmarks/global_norm/global_norm_benchmark.cu
    benchmarks/cross_entropy/cross_entropy_benchmark.cu
    benchmarks/softmax/causal_softmax_benchmark.cu
    benchmarks/attention/attention_benchmark.cu
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
    add_test(NAME softmax_smoke
             COMMAND cuda_benchmarks softmax 128 256)
    add_test(NAME conv2d_smoke
             COMMAND cuda_benchmarks conv2d 2 3 8 8 4)
    add_test(NAME rmsnorm_smoke COMMAND cuda_benchmarks rmsnorm 8 512)
    add_test(NAME rmsnorm_short COMMAND cuda_benchmarks rmsnorm 7 12)
    add_test(NAME swiglu_smoke COMMAND cuda_benchmarks swiglu 1027)
    add_test(NAME rope_smoke COMMAND cuda_benchmarks rope 2 17 3 64 32)
    add_test(NAME rope_full COMMAND cuda_benchmarks rope 1 3 2 8 8)
    add_test(NAME embedding_smoke COMMAND cuda_benchmarks embedding 67 64 31)
    add_test(NAME adamw_smoke COMMAND cuda_benchmarks adamw 4100)
    add_test(NAME global_norm_smoke COMMAND cuda_benchmarks global_norm 4100)
    add_test(NAME global_norm_small COMMAND cuda_benchmarks global_norm 4)
    add_test(NAME cross_entropy_smoke COMMAND cuda_benchmarks cross_entropy 7 257)
    add_test(NAME causal_softmax_smoke COMMAND cuda_benchmarks causal_softmax 2 3 65)
    add_test(NAME causal_softmax_single COMMAND cuda_benchmarks causal_softmax 1 1 1)
    add_test(NAME attention_smoke COMMAND cuda_benchmarks attention 2 17 3 32)
    add_test(NAME vector_add_tail COMMAND cuda_benchmarks vector_add 1027)
    add_test(NAME softmax_tail COMMAND cuda_benchmarks softmax 7 259)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME profile_script
                 COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/test_profile.py)
    endif()
    get_property(benchmark_tests DIRECTORY PROPERTY TESTS)
    set_tests_properties(${benchmark_tests} PROPERTIES RESOURCE_LOCK gpu ENVIRONMENT "CUDA_BENCH_DEVICE_INFO=0")
endif()
