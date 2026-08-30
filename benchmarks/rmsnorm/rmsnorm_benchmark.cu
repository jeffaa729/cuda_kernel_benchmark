// Validates the migrated forward/backward operation before Nsight reports its kernels.
// CPU work and host copies are outside the GPU kernel timings.
#include <cuda_bench/benchmarks.hpp>
#include <cuda_bench/validation.hpp>
#include <hpc/rmsnorm.hpp>
#include <cmath>
#include <vector>

namespace {
namespace reference {


void rmsnorm_forward_cpu(
    float* output,
    float* inverse_rms,
    const float* input,
    const float* weight,
    int rows,
    int hidden_size,
    float epsilon) {
    for (int row = 0; row < rows; ++row) {
        const int offset = row * hidden_size;
        float sum_of_squares = 0.0F;
        for (int column = 0; column < hidden_size; ++column) {
            const float value = input[offset + column];
            sum_of_squares += value * value;
        }

        const float scale = 1.0F / std::sqrt(sum_of_squares / hidden_size + epsilon);
        inverse_rms[row] = scale;
        for (int column = 0; column < hidden_size; ++column) {
            output[offset + column] = input[offset + column] * scale * weight[column];
        }
    }
}

void rmsnorm_backward_cpu(
    float* input_gradient,
    float* weight_gradient,
    const float* output_gradient,
    const float* input,
    const float* weight,
    const float* inverse_rms,
    int rows,
    int hidden_size) {
    for (int row = 0; row < rows; ++row) {
        const int offset = row * hidden_size;
        const float scale = inverse_rms[row];
        float projection = 0.0F;

        for (int column = 0; column < hidden_size; ++column) {
            projection += output_gradient[offset + column] * weight[column] * input[offset + column];
        }

        const float correction = projection * scale * scale * scale / hidden_size;
        for (int column = 0; column < hidden_size; ++column) {
            const int index = offset + column;
            const float upstream_times_weight = output_gradient[index] * weight[column];
            input_gradient[index] += scale * upstream_times_weight - input[index] * correction;
            weight_gradient[column] += output_gradient[index] * input[index] * scale;
        }
    }
}


}  // namespace reference
}  // namespace

namespace cuda_bench {

int rmsnorm_benchmark(std::size_t rows, std::size_t hidden) {
    require_multiple(hidden, 4, "hidden");
    const std::size_t size = rows * hidden;
    auto input = random_values(size), weight = random_values(hidden, 43);
    auto gradient = random_values(size, 44);
    std::vector<float> output(size), inverse(rows), dx(size, 0.25F), dw(hidden, 0.25F);
    DeviceBuffer<float> x(size), w(hidden), dy(size), y(size), inv(rows), gx(size), gw(hidden);
    x.copy_from_host(input.data()); w.copy_from_host(weight.data()); dy.copy_from_host(gradient.data());
    gx.copy_from_host(dx.data()); gw.copy_from_host(dw.data());
    reference::rmsnorm_forward_cpu(output.data(), inverse.data(), input.data(), weight.data(),
                                  rows, hidden, 1.0e-6F);
    reference::rmsnorm_backward_cpu(dx.data(), dw.data(), gradient.data(), input.data(),
                                   weight.data(), inverse.data(), rows, hidden);
    hpc::rmsnorm_forward(y.data(), inv.data(), x.data(), w.data(), rows, hidden, 1.0e-6F);
    hpc::rmsnorm_backward(gx.data(), gw.data(), dy.data(), x.data(), w.data(), inv.data(), rows, hidden);
    CUDA_CHECK(cudaDeviceSynchronize());
    bool valid = validate_result(y, output, "rmsnorm output");
    valid &= validate_result(inv, inverse, "rmsnorm inverse RMS");
    valid &= validate_result(gx, dx, "rmsnorm input gradient", 3.0e-5F);
    valid &= validate_result(gw, dw, "rmsnorm weight gradient", 3.0e-4F);
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace cuda_bench
