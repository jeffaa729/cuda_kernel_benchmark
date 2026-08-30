// Validates the migrated forward/backward operation before Nsight reports its kernels.
// CPU work and host copies are outside the GPU kernel timings.
#include <cuda_bench/benchmarks.hpp>
#include <cuda_bench/validation.hpp>
#include <hpc/softmax.hpp>
#include <cmath>
#include <vector>

namespace {
namespace reference {


void causal_softmax_forward_cpu(
    float* probabilities,
    const float* logits,
    int batch_size,
    int heads,
    int sequence_length,
    float scale) {
    const int matrix_size = sequence_length * sequence_length;
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < heads; ++head) {
            const int matrix_offset = (batch * heads + head) * matrix_size;
            for (int query = 0; query < sequence_length; ++query) {
                const int row_offset = matrix_offset + query * sequence_length;
                float maximum = scale * logits[row_offset];
                for (int key = 1; key <= query; ++key) {
                    maximum = std::max(maximum, scale * logits[row_offset + key]);
                }

                float denominator = 0.0F;
                for (int key = 0; key <= query; ++key) {
                    const float exponential =
                        std::exp(scale * logits[row_offset + key] - maximum);
                    probabilities[row_offset + key] = exponential;
                    denominator += exponential;
                }

                for (int key = 0; key <= query; ++key) {
                    probabilities[row_offset + key] /= denominator;
                }
                for (int key = query + 1; key < sequence_length; ++key) {
                    probabilities[row_offset + key] = 0.0F;
                }
            }
        }
    }
}

void causal_softmax_backward_cpu(
    float* logits_gradient,
    const float* probabilities_gradient,
    const float* probabilities,
    int batch_size,
    int heads,
    int sequence_length,
    float scale) {
    const int matrix_size = sequence_length * sequence_length;
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < heads; ++head) {
            const int matrix_offset = (batch * heads + head) * matrix_size;
            for (int query = 0; query < sequence_length; ++query) {
                const int row_offset = matrix_offset + query * sequence_length;
                float projection = 0.0F;
                for (int key = 0; key <= query; ++key) {
                    projection += probabilities_gradient[row_offset + key] *
                                  probabilities[row_offset + key];
                }

                for (int key = 0; key <= query; ++key) {
                    const int index = row_offset + key;
                    logits_gradient[index] +=
                        scale * probabilities[index] *
                        (probabilities_gradient[index] - projection);
                }
            }
        }
    }
}


}  // namespace reference
}  // namespace

namespace cuda_bench {

int causal_softmax_benchmark(std::size_t batch, std::size_t heads, std::size_t sequence) {
    const std::size_t size = batch * heads * sequence * sequence;
    const float scale = 0.125F;
    auto logits = random_values(size), gradient = random_values(size, 43);
    for (auto& value : logits) value *= 16.0F;
    std::vector<float> output(size), dx(size, 0.25F);
    DeviceBuffer<float> x(size), dy(size), y(size), gx(size);
    x.copy_from_host(logits.data()); dy.copy_from_host(gradient.data()); gx.copy_from_host(dx.data());
    reference::causal_softmax_forward_cpu(output.data(), logits.data(), batch, heads, sequence, scale);
    reference::causal_softmax_backward_cpu(dx.data(), gradient.data(), output.data(), batch, heads, sequence, scale);
    hpc::causal_softmax_forward(y.data(), x.data(), batch, heads, sequence, scale);
    hpc::causal_softmax_backward(gx.data(), dy.data(), y.data(), batch, heads, sequence, scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    bool valid = validate_result(y, output, "causal softmax output");
    valid &= validate_result(gx, dx, "causal softmax gradient");
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace cuda_bench
