#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cuda_bench {

int vector_add_benchmark(std::size_t size);
int transpose_benchmark(std::size_t n);
int reduction_benchmark(std::size_t size);
int gemm_benchmark(std::size_t n,
                   const std::vector<std::string>& algorithms = {});
int softmax_benchmark(std::size_t rows, std::size_t cols);
int conv2d_benchmark(std::size_t batch_size, std::size_t c_in,
                     std::size_t height, std::size_t width,
                     std::size_t c_out);

int rmsnorm_benchmark(std::size_t rows, std::size_t hidden);
int swiglu_benchmark(std::size_t elements);
int rope_benchmark(std::size_t batch, std::size_t sequence, std::size_t heads, std::size_t head_size, std::size_t rotary_size);
int embedding_benchmark(std::size_t tokens, std::size_t hidden, std::size_t vocabulary);
int adamw_benchmark(std::size_t elements);
int global_norm_benchmark(std::size_t elements);
int cross_entropy_benchmark(std::size_t rows, std::size_t vocabulary);
int causal_softmax_benchmark(std::size_t batch, std::size_t heads, std::size_t sequence);
int attention_benchmark(std::size_t batch, std::size_t sequence, std::size_t heads, std::size_t head_size);

}  // namespace cuda_bench
