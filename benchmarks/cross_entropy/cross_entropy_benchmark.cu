// Validates the migrated forward/backward operation before Nsight reports its kernels.
// CPU work and host copies are outside the GPU kernel timings.
#include <cuda_bench/benchmarks.hpp>
#include <cuda_bench/validation.hpp>
#include <hpc/cross_entropy.hpp>
#include <cmath>
#include <vector>

namespace {
namespace reference {


void cross_entropy_forward_cpu(
    float* mean_loss,
    float* logsumexp,
    const float* logits,
    const int* targets,
    int rows,
    int vocabulary_size) {
    float loss = 0.0F;
    for (int row = 0; row < rows; ++row) {
        const int offset = row * vocabulary_size;
        float maximum = logits[offset];
        for (int column = 1; column < vocabulary_size; ++column) {
            maximum = std::max(maximum, logits[offset + column]);
        }

        float sum = 0.0F;
        for (int column = 0; column < vocabulary_size; ++column) {
            sum += std::exp(logits[offset + column] - maximum);
        }
        const float row_logsumexp = maximum + std::log(sum);
        logsumexp[row] = row_logsumexp;
        loss += row_logsumexp - logits[offset + targets[row]];
    }
    *mean_loss = loss / rows;
}

void cross_entropy_backward_cpu(
    float* logits_gradient,
    const float* logits,
    const float* logsumexp,
    const int* targets,
    int rows,
    int vocabulary_size) {
    const float scale = 1.0F / rows;
    for (int row = 0; row < rows; ++row) {
        const int offset = row * vocabulary_size;
        for (int column = 0; column < vocabulary_size; ++column) {
            const float probability =
                std::exp(logits[offset + column] - logsumexp[row]);
            logits_gradient[offset + column] = scale *
                (probability - static_cast<float>(column == targets[row]));
        }
    }
}


}  // namespace reference
}  // namespace

namespace cuda_bench {

int cross_entropy_benchmark(std::size_t rows, std::size_t vocabulary) {
    const std::size_t size = rows * vocabulary;
    auto logits = random_values(size);
    for (auto& value : logits) value *= 12.0F;
    std::vector<int> targets(rows);
    for (std::size_t i = 0; i < rows; ++i) targets[i] = (i * 17) % vocabulary;
    std::vector<float> loss(1), lse(rows), dx(size);
    DeviceBuffer<float> x(size), y(1), saved(rows), gx(size);
    DeviceBuffer<int> labels(rows);
    x.copy_from_host(logits.data()); labels.copy_from_host(targets.data());
    reference::cross_entropy_forward_cpu(loss.data(), lse.data(), logits.data(), targets.data(),
                                         rows, vocabulary);
    reference::cross_entropy_backward_cpu(dx.data(), logits.data(), lse.data(), targets.data(),
                                          rows, vocabulary);
    hpc::cross_entropy_forward(y.data(), saved.data(), x.data(), labels.data(), rows, vocabulary);
    hpc::cross_entropy_backward(gx.data(), x.data(), saved.data(), labels.data(), rows, vocabulary);
    CUDA_CHECK(cudaDeviceSynchronize());
    bool valid = validate_result(y, loss, "cross entropy mean loss");
    valid &= validate_result(saved, lse, "cross entropy LSE");
    valid &= validate_result(gx, dx, "cross entropy logits gradient");
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace cuda_bench
