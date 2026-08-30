#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace hpc {

enum class GemmAlgo {
    Naive,
    Tiled,
    Tiled_v2,
    Tiled_v3,
    Tiled_v4,
    Tiled_v5,
    TensorCore,
    CublasTensorCore,
    Cublas,
};

const char* to_string(GemmAlgo algo);

void gemm(const float* a, const float* b, float* c, int n,
          GemmAlgo algo = GemmAlgo::Cublas);

// Computes a strided batch of logical matrix products. Transposed operands are
// stored as [K, M] for the left matrix or [N, K] for the right matrix.
void gemm_strided_batched(
    float* output,
    const float* left,
    const float* right,
    int M,
    int N,
    int K,
    int batch_count,
    int left_batch_stride,
    int right_batch_stride,
    int output_batch_stride,
    bool transpose_left,
    bool transpose_right,
    bool accumulate,
    cudaStream_t stream = nullptr);

}  // namespace hpc
