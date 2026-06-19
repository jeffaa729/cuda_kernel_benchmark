#include <cuda_bench/benchmark.hpp>
#include <cuda_bench/benchmarks.hpp>
#include <cuda_bench/cuda_utils.cuh>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <sstream>
#include <vector>

namespace {

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

namespace cuda_bench {

int vector_add_benchmark(std::size_t size, std::size_t iterations) {
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
    std::ostringstream problem_size;
    problem_size << size << " elements, " << std::fixed << std::setprecision(1)
                 << vector_mib << " MiB/vector, device " << device.name;
    cuda_bench::print_benchmark_header("vector_add", problem_size.str(),
                                       iterations);
    cuda_bench::print_metric_row("CPU scalar", cpu_ms,
                                 bandwidth_gbs(size, cpu_ms), "GB/s");
    cuda_bench::print_metric_row("GPU kernel only", kernel_ms,
                                 bandwidth_gbs(size, kernel_ms), "GB/s");
    cuda_bench::print_validation_error(max_error);

    return max_error <= 1.0e-6 ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace cuda_bench
