#pragma once

#include <cstddef>

namespace hpc {

enum class GemmAlgo {
    Naive,
    Tiled,
    Register,
};

const char* to_string(GemmAlgo algo);

void gemm(const float* a, const float* b, float* c, int n,
          GemmAlgo algo = GemmAlgo::Naive);

}  // namespace hpc
