// Validates the migrated forward/backward operation before Nsight reports its kernels.
// CPU work and host copies are outside the GPU kernel timings.
#include <cuda_bench/benchmarks.hpp>
#include <cuda_bench/validation.hpp>
#include <hpc/embedding.hpp>
#include <cmath>
#include <vector>

namespace {
namespace reference {


void embedding_forward_cpu(
    float* output,
    const int* token_ids,
    const float* weight,
    int token_count,
    int hidden_size) {
    for (int token = 0; token < token_count; ++token) {
        const int source = token_ids[token] * hidden_size;
        const int destination = token * hidden_size;
        for (int column = 0; column < hidden_size; ++column) {
            output[destination + column] = weight[source + column];
        }
    }
}

void embedding_backward_cpu(
    float* weight_gradient,
    const float* output_gradient,
    const int* token_ids,
    int token_count,
    int hidden_size) {
    for (int token = 0; token < token_count; ++token) {
        const int destination = token_ids[token] * hidden_size;
        const int source = token * hidden_size;
        for (int column = 0; column < hidden_size; ++column) {
            weight_gradient[destination + column] +=
                output_gradient[source + column];
        }
    }
}


}  // namespace reference
}  // namespace

namespace cuda_bench {

int embedding_benchmark(std::size_t tokens, std::size_t hidden, std::size_t vocabulary) {
    require_multiple(hidden, 4, "hidden");
    const std::size_t size = tokens * hidden, weights = vocabulary * hidden;
    std::vector<int> ids(tokens);
    // Repeated IDs deliberately exercise atomic accumulation.
    for (std::size_t i = 0; i < tokens; ++i) ids[i] = (i * 17) % std::min(vocabulary, std::size_t{31});
    auto weight = random_values(weights), gradient = random_values(size, 43);
    std::vector<float> output(size), dw(weights, 0.25F);
    DeviceBuffer<int> token_ids(tokens);
    DeviceBuffer<float> w(weights), dy(size), y(size), gw(weights);
    token_ids.copy_from_host(ids.data()); w.copy_from_host(weight.data());
    dy.copy_from_host(gradient.data()); gw.copy_from_host(dw.data());
    reference::embedding_forward_cpu(output.data(), ids.data(), weight.data(), tokens, hidden);
    reference::embedding_backward_cpu(dw.data(), gradient.data(), ids.data(), tokens, hidden);
    hpc::embedding_forward(y.data(), token_ids.data(), w.data(), tokens, hidden);
    hpc::embedding_backward(gw.data(), dy.data(), token_ids.data(), tokens, hidden);
    CUDA_CHECK(cudaDeviceSynchronize());
    bool valid = validate_result(y, output, "embedding output");
    valid &= validate_result(gw, dw, "embedding weight gradient", 3.0e-4F);
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace cuda_bench
