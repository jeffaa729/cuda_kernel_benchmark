#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace hpc {

enum class SoftmaxAlgo {
    Naive,
    SharedMemory,
    WarpShuffleRegCache,
    BlockReduce,
};

const char* to_string(SoftmaxAlgo algo);

void softmax(const float* input, float* output, std::size_t rows,
             std::size_t cols, SoftmaxAlgo algo = SoftmaxAlgo::Naive);

// Scaled causal softmax over [batch, heads, sequence, sequence] scores.
void causal_softmax_forward(
    float* probabilities,
    const float* logits,
    int batch_size,
    int heads,
    int sequence_length,
    float scale,
    cudaStream_t stream = nullptr);

// Accumulates the scaled softmax gradient into logits_gradient.
void causal_softmax_backward(
    float* logits_gradient,
    const float* probabilities_gradient,
    const float* probabilities,
    int batch_size,
    int heads,
    int sequence_length,
    float scale,
    cudaStream_t stream = nullptr);

}  // namespace hpc
