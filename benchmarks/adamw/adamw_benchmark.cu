// Validates the migrated forward/backward operation before Nsight reports its kernels.
// CPU work and host copies are outside the GPU kernel timings.
#include <cuda_bench/benchmarks.hpp>
#include <cuda_bench/validation.hpp>
#include <hpc/adamw.hpp>
#include <cmath>
#include <vector>

namespace {
namespace reference {


void adamw_step_cpu(
    float* parameters,
    float* first_moment,
    float* second_moment,
    const float* gradients,
    int elements,
    int step,
    const hpc::AdamWConfig& config) {
    const float first_correction =
        1.0F / (1.0F - std::pow(config.beta1, step));
    const float second_correction =
        1.0F / (1.0F - std::pow(config.beta2, step));

    for (int index = 0; index < elements; ++index) {
        const float gradient = gradients[index];
        const float first =
            config.beta1 * first_moment[index] +
            (1.0F - config.beta1) * gradient;
        const float second =
            config.beta2 * second_moment[index] +
            (1.0F - config.beta2) * gradient * gradient;
        float parameter = parameters[index];
        parameter -=
            config.learning_rate * config.weight_decay * parameter;
        parameter -= config.learning_rate *
            (first * first_correction) /
            (std::sqrt(second * second_correction) + config.epsilon);

        parameters[index] = parameter;
        first_moment[index] = first;
        second_moment[index] = second;
    }
}


}  // namespace reference
}  // namespace

namespace cuda_bench {

int adamw_benchmark(std::size_t size) {
    require_multiple(size, 4, "elements");
    const hpc::AdamWConfig config{3.0e-4F, 0.9F, 0.95F, 1.0e-8F, 0.1F};
    auto parameters = random_values(size), gradient = random_values(size, 43);
    gradient[0] = 0.0F;  // Check decoupled decay without an adaptive update.
    std::vector<float> first(size, 0.0F), second(size, 0.0F);
    DeviceBuffer<float> p(size), g(size), m(size), v(size);
    p.copy_from_host(parameters.data()); g.copy_from_host(gradient.data());
    m.copy_from_host(first.data()); v.copy_from_host(second.data());
    for (int step = 1; step <= 3; ++step) {
        reference::adamw_step_cpu(parameters.data(), first.data(), second.data(), gradient.data(),
                                  size, step, config);
        hpc::adamw_step(p.data(), m.data(), v.data(), g.data(), size, step, config);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    bool valid = validate_result(p, parameters, "adamw parameters", 2.0e-6F);
    valid &= validate_result(m, first, "adamw first moment", 2.0e-6F);
    valid &= validate_result(v, second, "adamw second moment", 2.0e-6F);
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace cuda_bench
