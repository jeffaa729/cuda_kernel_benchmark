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

function(add_cuda_benchmark target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
endfunction()

add_cuda_benchmark(cuda_benchmarks
    benchmarks/main.cu
    benchmarks/vector_add/vector_add.cu
    benchmarks/transpose/transpose.cu
)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME vector_add_smoke
             COMMAND cuda_benchmarks vector_add 1048576 2)
    add_test(NAME transpose_smoke
             COMMAND cuda_benchmarks transpose 128 256)
endif()
