// Validates the migrated forward/backward operation before Nsight reports its kernels.
// CPU work and host copies are outside the GPU kernel timings.
#include <cuda_bench/benchmarks.hpp>
#include <cuda_bench/validation.hpp>
#include <hpc/global_norm.hpp>
#include <cmath>
#include <vector>

namespace {
namespace reference {


float global_norm_cpu(const float* gradients, int elements) {
    double sum = 0.0;
    for (int index = 0; index < elements; ++index) {
        const double gradient = gradients[index];
        sum += gradient * gradient;
    }
    return static_cast<float>(std::sqrt(sum));
}

void clip_gradients_cpu(
    float* gradients,
    int elements,
    float norm,
    float max_norm) {
    const float scale = std::min(1.0F, max_norm / norm);
    for (int index = 0; index < elements; ++index) {
        gradients[index] *= scale;
    }
}


}  // namespace reference
}  // namespace

namespace cuda_bench {

int global_norm_benchmark(std::size_t size) {
    require_multiple(size, 4, "elements");
    auto gradient = random_values(size);
    const float norm = reference::global_norm_cpu(gradient.data(), size);
    std::vector<float> expected_norm{norm};
    DeviceBuffer<float> g(size), n(1), workspace(hpc::global_norm_workspace_elements(size));
    g.copy_from_host(gradient.data());
    hpc::global_norm(n.data(), g.data(), workspace.data(), size);
    reference::clip_gradients_cpu(gradient.data(), size, norm, 1.0F);
    hpc::clip_gradients(g.data(), n.data(), size, 1.0F);
    CUDA_CHECK(cudaDeviceSynchronize());
    bool valid = validate_result(n, expected_norm, "global norm");
    valid &= validate_result(g, gradient, "clipped gradients");
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace cuda_bench
