#pragma once

#include <cstddef>

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

}  // namespace hpc
