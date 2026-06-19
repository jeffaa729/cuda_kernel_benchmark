#pragma once

#include <cstddef>

namespace cuda_bench {

int vector_add_benchmark(std::size_t size, std::size_t iterations);
int transpose_benchmark(std::size_t n);

}  // namespace cuda_bench
