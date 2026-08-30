#pragma once

#include <cuda_bench/cuda_utils.cuh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda_bench {

inline std::vector<float> random_values(std::size_t size, unsigned seed = 42) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> values(size);
    std::generate(values.begin(), values.end(), [&] { return distribution(generator); });
    return values;
}

inline void require_multiple(std::size_t value, int multiple, const char* name) {
    if (!value || value > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        value % multiple != 0) {
        throw std::invalid_argument(std::string(name) + " must be a positive multiple of " +
                                    std::to_string(multiple) + " that fits in int");
    }
}

inline bool validate_result(const DeviceBuffer<float>& device,
                            const std::vector<float>& expected,
                            const char* name, float tolerance = 2.0e-5F) {
    std::vector<float> actual(expected.size());
    device.copy_to_host(actual.data());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
            std::abs(actual[i] - expected[i]) >
                tolerance * std::max(1.0F, std::abs(expected[i]))) {
            std::cerr << name << " mismatch at " << i << ": "
                      << actual[i] << " versus " << expected[i] << '\n';
            return false;
        }
    }
    return true;
}

}  // namespace cuda_bench
