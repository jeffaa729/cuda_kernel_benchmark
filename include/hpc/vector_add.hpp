#pragma once

#include <cstddef>

namespace hpc {

enum class VectorAddAlgo {
    Naive,
};

const char* to_string(VectorAddAlgo algo);

void vector_add(const float* a, const float* b, float* c, std::size_t size,
                VectorAddAlgo algo = VectorAddAlgo::Naive);

}  // namespace hpc
