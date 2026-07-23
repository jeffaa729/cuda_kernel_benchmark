#include <cuda_bench/benchmark.hpp>
#include <cuda_bench/benchmarks.hpp>
#include <cuda_bench/cuda_utils.cuh>
#include <hpc/conv2d.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

CUDA_BENCH_NOINLINE void conv2d_cpu(const float* input, const float* weight,
                                    const float* bias, float* output,
                                    std::size_t batch_size, std::size_t c_in,
                                    std::size_t height, std::size_t width,
                                    std::size_t c_out,
                                    std::size_t kernel_h,
                                    std::size_t kernel_w,
                                    std::size_t height_out,
                                    std::size_t width_out,
                                    std::size_t stride,
                                    std::size_t padding) {
    for (std::size_t n = 0; n < batch_size; ++n) {
        for (std::size_t co = 0; co < c_out; ++co) {
            for (std::size_t oh = 0; oh < height_out; ++oh) {
                for (std::size_t ow = 0; ow < width_out; ++ow) {
                    float sum = bias[co];

                    for (std::size_t ci = 0; ci < c_in; ++ci) {
                        for (std::size_t kh = 0; kh < kernel_h; ++kh) {
                            const int ih = static_cast<int>(oh * stride + kh) -
                                           static_cast<int>(padding);
                            if (ih < 0 || ih >= static_cast<int>(height)) {
                                continue;
                            }

                            for (std::size_t kw = 0; kw < kernel_w; ++kw) {
                                const int iw =
                                    static_cast<int>(ow * stride + kw) -
                                    static_cast<int>(padding);
                                if (iw < 0 || iw >= static_cast<int>(width)) {
                                    continue;
                                }

                                const std::size_t input_idx =
                                    ((n * c_in + ci) * height +
                                     static_cast<std::size_t>(ih)) *
                                        width +
                                    static_cast<std::size_t>(iw);
                                const std::size_t weight_idx =
                                    ((co * c_in + ci) * kernel_h + kh) *
                                        kernel_w +
                                    kw;
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }

                    const std::size_t output_idx =
                        ((n * c_out + co) * height_out + oh) * width_out + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

bool validate_conv2d(const float* cpu_result, const float* gpu_result,
                     std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        const float tolerance =
            1.0e-4f * std::max(1.0f, std::abs(cpu_result[i]));
        if (std::abs(cpu_result[i] - gpu_result[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

bool run_and_validate_conv2d(hpc::Conv2DAlgo algo,
                             cuda_bench::DeviceBuffer<float>& device_input,
                             cuda_bench::DeviceBuffer<float>& device_weight,
                             cuda_bench::DeviceBuffer<float>& device_bias,
                             cuda_bench::DeviceBuffer<float>& device_output,
                             float* gpu_result, const float* cpu_result,
                             std::size_t batch_size, std::size_t c_in,
                             std::size_t height, std::size_t width,
                             std::size_t c_out, std::size_t kernel_h,
                             std::size_t kernel_w, std::size_t height_out,
                             std::size_t width_out, std::size_t stride,
                             std::size_t padding) {
    hpc::conv2d(device_input.data(), device_weight.data(), device_bias.data(),
                device_output.data(), static_cast<int>(batch_size),
                static_cast<int>(c_in), static_cast<int>(height),
                static_cast<int>(width), static_cast<int>(c_out),
                static_cast<int>(kernel_h), static_cast<int>(kernel_w),
                static_cast<int>(height_out), static_cast<int>(width_out),
                static_cast<int>(stride), static_cast<int>(padding), algo);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    device_output.copy_to_host(gpu_result);
    return validate_conv2d(cpu_result, gpu_result,
                           batch_size * c_out * height_out * width_out);
}

}  // namespace

namespace cuda_bench {

int conv2d_benchmark(std::size_t batch_size, std::size_t c_in,
                     std::size_t height, std::size_t width,
                     std::size_t c_out) {
    constexpr std::size_t kernel_h = 3;
    constexpr std::size_t kernel_w = 3;
    constexpr std::size_t stride = 1;
    constexpr std::size_t padding = 1;

    const std::size_t height_out =
        (height + 2 * padding - kernel_h) / stride + 1;
    const std::size_t width_out =
        (width + 2 * padding - kernel_w) / stride + 1;
    const std::size_t input_size = batch_size * c_in * height * width;
    const std::size_t weight_size = c_out * c_in * kernel_h * kernel_w;
    const std::size_t output_size =
        batch_size * c_out * height_out * width_out;

    std::vector<float> input(input_size);
    std::vector<float> weight(weight_size);
    std::vector<float> bias(c_out);
    std::vector<float> cpu_result(output_size);
    std::vector<float> gpu_result(output_size);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::generate(input.begin(), input.end(),
                  [&] { return distribution(generator); });
    std::generate(weight.begin(), weight.end(),
                  [&] { return distribution(generator); });
    std::generate(bias.begin(), bias.end(),
                  [&] { return distribution(generator); });

    conv2d_cpu(input.data(), weight.data(), bias.data(), cpu_result.data(),
               batch_size, c_in, height, width, c_out, kernel_h, kernel_w,
               height_out, width_out, stride, padding);

    cuda_bench::DeviceBuffer<float> device_input(input_size);
    cuda_bench::DeviceBuffer<float> device_weight(weight_size);
    cuda_bench::DeviceBuffer<float> device_bias(c_out);
    cuda_bench::DeviceBuffer<float> device_output(output_size);

    device_input.copy_from_host(input.data());
    device_weight.copy_from_host(weight.data());
    device_bias.copy_from_host(bias.data());

    const bool naive_valid = run_and_validate_conv2d(
        hpc::Conv2DAlgo::Naive, device_input, device_weight, device_bias,
        device_output, gpu_result.data(), cpu_result.data(), batch_size, c_in,
        height, width, c_out, kernel_h, kernel_w, height_out, width_out, stride,
        padding);

    const bool tiled_valid = run_and_validate_conv2d(
        hpc::Conv2DAlgo::Tiled, device_input, device_weight, device_bias,
        device_output, gpu_result.data(), cpu_result.data(), batch_size, c_in,
        height, width, c_out, kernel_h, kernel_w, height_out, width_out, stride,
        padding);

    return naive_valid && tiled_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace cuda_bench
