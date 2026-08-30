#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace hpc {

enum class VectorAddAlgo {
    Naive,
    Vectorized,
};

const char* to_string(VectorAddAlgo algo);

void vector_add(const float* a, const float* b, float* c, std::size_t size,
                VectorAddAlgo algo = VectorAddAlgo::Naive);

// Backward buffers contain a multiple of four FP32 elements.
// Adds output_gradient into both input-gradient buffers.
void vector_add_backward(
    float* input_gradient,
    float* branch_gradient,
    const float* output_gradient,
    int elements,
    cudaStream_t stream = nullptr);

}  // namespace hpc
