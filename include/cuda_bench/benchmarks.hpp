#pragma once

#include <cstddef>

namespace cuda_bench {

int vector_add_benchmark(std::size_t size);
int transpose_benchmark(std::size_t n);
int reduction_benchmark(std::size_t size);
int gemm_benchmark(std::size_t n);

}  // namespace cuda_bench
