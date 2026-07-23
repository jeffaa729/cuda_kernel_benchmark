#pragma once

#include <cstddef>

namespace cuda_bench {

int vector_add_benchmark(std::size_t size);
int transpose_benchmark(std::size_t n);
int reduction_benchmark(std::size_t size);
int gemm_benchmark(std::size_t n);
int softmax_benchmark(std::size_t rows, std::size_t cols);
int conv2d_benchmark(std::size_t batch_size, std::size_t c_in,
                     std::size_t height, std::size_t width,
                     std::size_t c_out);

}  // namespace cuda_bench
