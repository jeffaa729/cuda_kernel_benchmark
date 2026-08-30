#pragma once

#include <cuda_runtime.h>

namespace hpc {

void swiglu_forward(
    float* output,
    const float* gate,
    const float* up,
    int elements,
    cudaStream_t stream = nullptr);

void swiglu_backward(
    float* gate_gradient,
    float* up_gradient,
    const float* output_gradient,
    const float* gate,
    const float* up,
    int elements,
    cudaStream_t stream = nullptr);

}  // namespace hpc
