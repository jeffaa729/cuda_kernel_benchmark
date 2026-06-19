#include <cuda_bench/benchmark.hpp>
#include <cuda_bench/benchmarks.hpp>
#include <cuda_bench/cuda_utils.cuh>
#include <hpc/reduction.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

CUDA_BENCH_NOINLINE void reduction_cpu(const float* input, float* output,
                                       std::size_t size) {
    constexpr std::size_t block_size = 1024;
    const std::size_t blocks = size / block_size;

    for (std::size_t block = 0; block < blocks; ++block) {
        std::array<float, block_size> data{};
        for (std::size_t i = 0; i < block_size; ++i) {
            data[i] = input[block * block_size + i];
        }

        for (std::size_t stride = 1; stride < block_size; stride *= 2) {
            for (std::size_t tid = 0; tid < block_size; tid += 2 * stride) {
                data[tid] += data[tid + stride];
            }
        }

        output[block] = data[0];
    }
}

}  // namespace

namespace cuda_bench {

int reduction_benchmark(std::size_t size) {
    constexpr std::size_t block_size = 1024;
    constexpr hpc::ReductionAlgo algo = hpc::ReductionAlgo::Interleave;

    if (size % block_size != 0) {
        return EXIT_FAILURE;
    }

    const std::size_t output_size = size / block_size;
    std::vector<float> input(size);
    std::vector<float> cpu_result(output_size);
    std::vector<float> gpu_result(output_size);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::generate(input.begin(), input.end(),
                  [&] { return distribution(generator); });

    reduction_cpu(input.data(), cpu_result.data(), size);

    cuda_bench::DeviceBuffer<float> device_input(size);
    cuda_bench::DeviceBuffer<float> device_output(output_size);
    device_input.copy_from_host(input.data());

    hpc::reduction(device_input.data(), device_output.data(), size, algo);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    device_output.copy_to_host(gpu_result.data());
    for (std::size_t i = 0; i < output_size; ++i) {
        if (cpu_result[i] != gpu_result[i]) {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

}  // namespace cuda_bench
