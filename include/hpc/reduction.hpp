#pragma once

#include <cstddef>

namespace hpc {

enum class ReductionAlgo {
    Interleave,
};

const char* to_string(ReductionAlgo algo);

void reduction(const float* input, float* output, std::size_t size,
               ReductionAlgo algo = ReductionAlgo::Interleave);

}  // namespace hpc
