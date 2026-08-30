// Validates the migrated forward/backward operation before Nsight reports its kernels.
// CPU work and host copies are outside the GPU kernel timings.
#include <cuda_bench/benchmarks.hpp>
#include <cuda_bench/validation.hpp>
#include <hpc/swiglu.hpp>
#include <cmath>
#include <vector>

namespace {
namespace reference {

namespace {

float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

}  // namespace

void swiglu_forward_cpu(
    float* output,
    const float* gate,
    const float* up,
    int elements) {
    for (int index = 0; index < elements; ++index) {
        const float gate_value = gate[index];
        output[index] = gate_value * sigmoid(gate_value) * up[index];
    }
}

void swiglu_backward_cpu(
    float* gate_gradient,
    float* up_gradient,
    const float* output_gradient,
    const float* gate,
    const float* up,
    int elements) {
    for (int index = 0; index < elements; ++index) {
        const float gate_value = gate[index];
        const float sigmoid_value = sigmoid(gate_value);
        const float silu_value = gate_value * sigmoid_value;
        const float silu_gradient =
            sigmoid_value * (1.0F + gate_value * (1.0F - sigmoid_value));

        gate_gradient[index] = output_gradient[index] * up[index] * silu_gradient;
        up_gradient[index] = output_gradient[index] * silu_value;
    }
}


}  // namespace reference
}  // namespace

namespace cuda_bench {

int swiglu_benchmark(std::size_t size) {
    require_multiple(size, 1, "elements");
    auto gate = random_values(size), up = random_values(size, 43), gradient = random_values(size, 44);
    std::vector<float> output(size), dg(size), du(size);
    DeviceBuffer<float> g(size), u(size), dy(size), y(size), gg(size), gu(size);
    g.copy_from_host(gate.data()); u.copy_from_host(up.data()); dy.copy_from_host(gradient.data());
    reference::swiglu_forward_cpu(output.data(), gate.data(), up.data(), size);
    reference::swiglu_backward_cpu(dg.data(), du.data(), gradient.data(), gate.data(), up.data(), size);
    hpc::swiglu_forward(y.data(), g.data(), u.data(), size);
    hpc::swiglu_backward(gg.data(), gu.data(), dy.data(), g.data(), u.data(), size);
    CUDA_CHECK(cudaDeviceSynchronize());
    bool valid = validate_result(y, output, "swiglu output");
    valid &= validate_result(gg, dg, "swiglu gate gradient");
    valid &= validate_result(gu, du, "swiglu up gradient");
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace cuda_bench
