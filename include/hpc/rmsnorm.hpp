#pragma once

#include <cuda_runtime.h>

namespace hpc {

void rmsnorm_forward(
    float* output,
    float* inverse_rms,
    const float* input,
    const float* weight,
    int rows,
    int hidden_size,
    float epsilon,
    cudaStream_t stream = nullptr);

// Accumulates into input_gradient and weight_gradient; callers must initialize
// those buffers according to the surrounding computation graph.
void rmsnorm_backward(
    float* input_gradient,
    float* weight_gradient,
    const float* output_gradient,
    const float* input,
    const float* weight,
    const float* inverse_rms,
    int rows,
    int hidden_size,
    cudaStream_t stream = nullptr);

}  // namespace hpc
