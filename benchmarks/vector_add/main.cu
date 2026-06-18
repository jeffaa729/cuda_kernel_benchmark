#include <cuda_bench/benchmark.hpp>
#include <cuda_bench/cuda_utils.cuh>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
//kernel
__global__ void vector_add_kernel(const float* a, const float* b, float* c,
                                  std::size_t size) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

CUDA_BENCH_NOINLINE void vector_add_cpu(const float* a, const float* b, float* c,
                                        std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

double bandwidth_gbs(std::size_t size, double milliseconds) {
    const double bytes = 3.0 * static_cast<double>(size) * sizeof(float);
    return bytes / (milliseconds * 1.0e6);
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t size =
        argc > 1 ? cuda_bench::parse_positive_size(argv[1], "vector size")
                 : 1ULL << 24;
    const std::size_t iterations =
        argc > 2 ? cuda_bench::parse_positive_size(argv[2], "iterations") : 20;
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>((size + threads_per_block - 1) /
                                        threads_per_block);

    std::vector<float> a(size);
    std::vector<float> b(size);
    std::vector<float> cpu_result(size);
    std::vector<float> gpu_result(size);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::generate(a.begin(), a.end(), [&] { return distribution(generator); });
    std::generate(b.begin(), b.end(), [&] { return distribution(generator); });

    vector_add_cpu(a.data(), b.data(), cpu_result.data(), size);
    const double cpu_ms = cuda_bench::measure_cpu_ms(iterations, [&] {
        vector_add_cpu(a.data(), b.data(), cpu_result.data(), size);
    });

    cuda_bench::DeviceBuffer<float> device_a(size);
    cuda_bench::DeviceBuffer<float> device_b(size);
    cuda_bench::DeviceBuffer<float> device_c(size);
    device_a.copy_from_host(a.data());
    device_b.copy_from_host(b.data());

    vector_add_kernel<<<blocks, threads_per_block>>>(
        device_a.data(), device_b.data(), device_c.data(), size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cuda_bench::CudaEvent kernel_start;
    cuda_bench::CudaEvent kernel_stop;
    kernel_start.record();
    for (std::size_t i = 0; i < iterations; ++i) {
        vector_add_kernel<<<blocks, threads_per_block>>>(
            device_a.data(), device_b.data(), device_c.data(), size);
    }
    kernel_stop.record();
    kernel_stop.synchronize();
    const double kernel_ms =
        cuda_bench::elapsed_ms(kernel_start, kernel_stop) /
        static_cast<double>(iterations);

    const double gpu_total_ms = cuda_bench::measure_cpu_ms(iterations, [&] {
        device_a.copy_from_host(a.data());
        device_b.copy_from_host(b.data());
        vector_add_kernel<<<blocks, threads_per_block>>>(
            device_a.data(), device_b.data(), device_c.data(), size);
        CUDA_CHECK(cudaGetLastError());
        device_c.copy_to_host(gpu_result.data());
    });

    device_c.copy_to_host(gpu_result.data());
    double max_error = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        max_error = std::max(
            max_error,
            static_cast<double>(std::abs(cpu_result[i] - gpu_result[i])));
    }

    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    const double vector_mib = size * sizeof(float) / (1024.0 * 1024.0);
    std::cout << "Benchmark: vector_add\n"
              << "Device: " << device.name << '\n'
              << "Elements: " << size << " (" << std::fixed
              << std::setprecision(1) << vector_mib << " MiB/vector)\n"
              << "Iterations: " << iterations << "\n\n"
              << std::left << std::setw(24) << "Method" << std::right
              << std::setw(12) << "Time (ms)" << std::setw(21)
              << "Throughput" << '\n'
              << std::string(57, '-') << '\n';
    cuda_bench::print_metric_row("CPU scalar", cpu_ms,
                                 bandwidth_gbs(size, cpu_ms), "GB/s");
    cuda_bench::print_metric_row("GPU kernel only", kernel_ms,
                                 bandwidth_gbs(size, kernel_ms), "GB/s");
    cuda_bench::print_time_row("GPU incl. transfers", gpu_total_ms);
    std::cout << "\nMax absolute error: " << std::scientific << max_error << '\n';

    return max_error <= 1.0e-6 ? EXIT_SUCCESS : EXIT_FAILURE;
}
