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

function(add_cuda_benchmark target source)
    add_executable(${target} ${source})
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

add_cuda_benchmark(vector_add_benchmark benchmarks/vector_add/main.cu)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME vector_add_smoke
             COMMAND vector_add_benchmark 1048576 2)
endif()
