// Validates the migrated forward/backward operation before Nsight reports its kernels.
// CPU work and host copies are outside the GPU kernel timings.
#include <cuda_bench/benchmarks.hpp>
#include <cuda_bench/validation.hpp>
#include <hpc/rope.hpp>
#include <cmath>
#include <vector>

namespace {
namespace reference {


void rope_forward_cpu(
    float* output,
    const float* input,
    const float* cosine,
    const float* sine,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    int rotary_size) {
    const int rotary_pairs = rotary_size / 2;
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int position = 0; position < sequence_length; ++position) {
            for (int head = 0; head < heads; ++head) {
                const int head_offset =
                    ((batch * sequence_length + position) * heads + head) * head_size;
                const int frequency_offset = position * rotary_pairs;

                for (int pair = 0; pair < rotary_pairs; ++pair) {
                    const int column = pair * 2;
                    const float first = input[head_offset + column];
                    const float second = input[head_offset + column + 1];
                    const float cosine_value = cosine[frequency_offset + pair];
                    const float sine_value = sine[frequency_offset + pair];
                    output[head_offset + column] =
                        first * cosine_value - second * sine_value;
                    output[head_offset + column + 1] =
                        first * sine_value + second * cosine_value;
                }

                for (int column = rotary_size; column < head_size; ++column) {
                    output[head_offset + column] = input[head_offset + column];
                }
            }
        }
    }
}

void rope_backward_cpu(
    float* input_gradient,
    const float* output_gradient,
    const float* cosine,
    const float* sine,
    int batch_size,
    int sequence_length,
    int heads,
    int head_size,
    int rotary_size) {
    const int rotary_pairs = rotary_size / 2;
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int position = 0; position < sequence_length; ++position) {
            for (int head = 0; head < heads; ++head) {
                const int head_offset =
                    ((batch * sequence_length + position) * heads + head) * head_size;
                const int frequency_offset = position * rotary_pairs;

                for (int pair = 0; pair < rotary_pairs; ++pair) {
                    const int column = pair * 2;
                    const float first_gradient = output_gradient[head_offset + column];
                    const float second_gradient = output_gradient[head_offset + column + 1];
                    const float cosine_value = cosine[frequency_offset + pair];
                    const float sine_value = sine[frequency_offset + pair];
                    input_gradient[head_offset + column] +=
                        first_gradient * cosine_value + second_gradient * sine_value;
                    input_gradient[head_offset + column + 1] +=
                        -first_gradient * sine_value + second_gradient * cosine_value;
                }

                for (int column = rotary_size; column < head_size; ++column) {
                    input_gradient[head_offset + column] += output_gradient[head_offset + column];
                }
            }
        }
    }
}


}  // namespace reference
}  // namespace

namespace cuda_bench {

int rope_benchmark(std::size_t batch, std::size_t sequence, std::size_t heads,
                   std::size_t hidden, std::size_t rotary) {
    require_multiple(hidden, 2, "head size");
    require_multiple(rotary, 2, "rotary size");
    if (rotary > hidden) throw std::invalid_argument("rotary size exceeds head size");
    const std::size_t size = batch * sequence * heads * hidden;
    const std::size_t frequencies = sequence * rotary / 2;
    auto input = random_values(size), gradient = random_values(size, 43);
    std::vector<float> cosine(frequencies), sine(frequencies), output(size), dx(size, 0.25F);
    for (std::size_t t = 0; t < sequence; ++t) {
        for (std::size_t j = 0; j < rotary / 2; ++j) {
            const float angle = t * std::pow(10000.0F, -2.0F * j / rotary);
            cosine[t * rotary / 2 + j] = std::cos(angle);
            sine[t * rotary / 2 + j] = std::sin(angle);
        }
    }
    DeviceBuffer<float> x(size), dy(size), c(frequencies), s(frequencies), y(size), gx(size);
    x.copy_from_host(input.data()); dy.copy_from_host(gradient.data());
    c.copy_from_host(cosine.data()); s.copy_from_host(sine.data()); gx.copy_from_host(dx.data());
    reference::rope_forward_cpu(output.data(), input.data(), cosine.data(), sine.data(),
                                batch, sequence, heads, hidden, rotary);
    reference::rope_backward_cpu(dx.data(), gradient.data(), cosine.data(), sine.data(),
                                 batch, sequence, heads, hidden, rotary);
    hpc::rope_forward(y.data(), x.data(), c.data(), s.data(), batch, sequence, heads, hidden, rotary);
    hpc::rope_backward(gx.data(), dy.data(), c.data(), s.data(), batch, sequence, heads, hidden, rotary);
    CUDA_CHECK(cudaDeviceSynchronize());
    bool valid = validate_result(y, output, "rope output");
    valid &= validate_result(gx, dx, "rope input gradient");
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace cuda_bench
