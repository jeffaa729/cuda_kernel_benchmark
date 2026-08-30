#include <hpc/vector_add.hpp>

#include <cuda_runtime.h>
#include <cuda_bench/cuda_utils.cuh>

namespace {

__global__ void vector_add_naive_kernel(const float* a, const float* b, float* c,
                                        std::size_t size) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

void launch_vector_add_naive(const float* a, const float* b, float* c,
                             std::size_t size) {
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((size + threads_per_block - 1) /
                                        threads_per_block);
    vector_add_naive_kernel<<<blocks, threads_per_block>>>(a, b, c, size);
}

}  // namespace

// Adds a transformer branch to its residual stream with vectorized FP32 memory operations.
// The backward kernel accumulates the shared upstream gradient into both graph branches.


namespace hpc {
namespace {

constexpr int kThreads = 256;

__global__ void vector_add_vectorized_kernel(
    float4* output,
    const float4* input,
    const float4* branch,
    int vectors) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= vectors) {
        return;
    }

    const float4 left = input[index];
    const float4 right = branch[index];
    output[index] = make_float4(
        left.x + right.x,
        left.y + right.y,
        left.z + right.z,
        left.w + right.w);
}

__global__ void vector_add_backward_kernel(
    float4* input_gradient,
    float4* branch_gradient,
    const float4* output_gradient,
    int vectors) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= vectors) {
        return;
    }

    const float4 gradient = output_gradient[index];
    float4 input_value = input_gradient[index];
    float4 branch_value = branch_gradient[index];
    input_value.x += gradient.x;
    input_value.y += gradient.y;
    input_value.z += gradient.z;
    input_value.w += gradient.w;
    branch_value.x += gradient.x;
    branch_value.y += gradient.y;
    branch_value.z += gradient.z;
    branch_value.w += gradient.w;
    input_gradient[index] = input_value;
    branch_gradient[index] = branch_value;
}

}  // namespace

static void vector_add_vectorized(
    float* output,
    const float* input,
    const float* branch,
    int elements,
    cudaStream_t stream) {
    const int vectors = elements / 4;
    const int blocks = (vectors + kThreads - 1) / kThreads;
    vector_add_vectorized_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<float4*>(output),
        reinterpret_cast<const float4*>(input),
        reinterpret_cast<const float4*>(branch),
        vectors);
    CUDA_CHECK(cudaGetLastError());
}

void vector_add_backward(
    float* input_gradient,
    float* branch_gradient,
    const float* output_gradient,
    int elements,
    cudaStream_t stream) {
    const int vectors = elements / 4;
    const int blocks = (vectors + kThreads - 1) / kThreads;
    vector_add_backward_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<float4*>(input_gradient),
        reinterpret_cast<float4*>(branch_gradient),
        reinterpret_cast<const float4*>(output_gradient),
        vectors);
    CUDA_CHECK(cudaGetLastError());
}



const char* to_string(VectorAddAlgo algo) {
    switch (algo) {
        case VectorAddAlgo::Vectorized:
            return "vectorized";
        case VectorAddAlgo::Naive:
            return "naive";
    }
    return "unknown";
}

void vector_add(const float* a, const float* b, float* c, std::size_t size,
                VectorAddAlgo algo) {
    switch (algo) {
        case VectorAddAlgo::Vectorized: {
            // Keep vectorized loads for the aligned prefix and handle any tail.
            const std::size_t aligned = size - size % 4;
            if (aligned) vector_add_vectorized(c, a, b, static_cast<int>(aligned), nullptr);
            if (aligned != size) launch_vector_add_naive(
                a + aligned, b + aligned, c + aligned, size - aligned);
            return;
        }
        case VectorAddAlgo::Naive:
            launch_vector_add_naive(a, b, c, size);
            return;
    }
}

}  // namespace hpc
