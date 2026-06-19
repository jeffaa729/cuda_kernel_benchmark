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

// transpose : B = A^T
CUDA_BENCH_NOINLINE void transpose_cpu(const float* a, float* b, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        for (std::size_t j = 0; j < size; ++j) {
            b[j * size + i] = a[i * size + j];
        }
    }
}

//Naive kernel
// coalesced : LOAD : for a[y * n + x] : for 1 warp, y is equal , x get [0,1,....31] for 32 thread
//             so 32 * 4 = 128 bytes -> 1 DRAM laod, so it is coalesced
// Strided :  WRITE : for b[x*b + y] : x [0...31] , y same -> 32 threads wrtie into x[0...31][y] 
//            every address far from Size N * 4 bytes
__global__ void transpose_naive(const float* a, float* b, std::size_t size) {
    const std::size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    const std::size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < size && y < size) {
        b[x * size + y] = a[y * size + x];
    }
}

}  // namespace

double bandwidth_gbs(std::size_t n, double milliseconds) {
    const double elements = static_cast<double>(n) * static_cast<double>(n);
    const double bytes = 2.0 * elements * sizeof(float);
    return bytes / (milliseconds * 1.0e6);
}

namespace cuda_bench {

int transpose_benchmark(std::size_t n) {
    const std::size_t iterations = 10;
    constexpr int tile_dim = 16;
    const dim3 threads(tile_dim, tile_dim);
    const dim3 blocks(static_cast<unsigned int>((n + tile_dim - 1) / tile_dim),
                      static_cast<unsigned int>((n + tile_dim - 1) / tile_dim));

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    std::vector<float> a(n * n);
    std::vector<float> cpu_result(n * n);
    std::vector<float> gpu_result(n * n);

    for (float& value : a) {
        value = distribution(generator);
    }
    const double cpu_ms = cuda_bench::measure_cpu_ms(iterations, [&] {
        transpose_cpu(a.data(), cpu_result.data(), n);
    });

    cuda_bench::DeviceBuffer<float> device_a(n * n);
    cuda_bench::DeviceBuffer<float> device_b(n * n);
    device_a.copy_from_host(a.data());

    transpose_naive<<<blocks, threads>>>(device_a.data(), device_b.data(), n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cuda_bench::CudaEvent kernel_start;
    cuda_bench::CudaEvent kernel_stop;
    kernel_start.record();
    for (std::size_t i = 0; i < iterations; ++i) {
        transpose_naive<<<blocks, threads>>>(device_a.data(), device_b.data(), n);
    }
    kernel_stop.record();
    kernel_stop.synchronize();
    const double naive_ms =
        cuda_bench::elapsed_ms(kernel_start, kernel_stop) /
        static_cast<double>(iterations);

    device_b.copy_to_host(gpu_result.data());
    double max_error = 0.0;
    for (std::size_t i = 0; i < n * n; ++i) {
        max_error = std::max(
            max_error,
            static_cast<double>(std::abs(cpu_result[i] - gpu_result[i])));
    }

    const double matrix_mib =
        static_cast<double>(n) * static_cast<double>(n) * sizeof(float) /
        (1024.0 * 1024.0);
    std::ostringstream problem_size;
    problem_size << n << " x " << n << ", " << std::fixed
                 << std::setprecision(1) << matrix_mib << " MiB/matrix";
    cuda_bench::print_benchmark_header("transpose", problem_size.str(),
                                       iterations);
    cuda_bench::print_metric_row("CPU scalar", cpu_ms,
                                 bandwidth_gbs(n, cpu_ms), "GB/s");
    cuda_bench::print_metric_row("GPU naive", naive_ms,
                                 bandwidth_gbs(n, naive_ms), "GB/s");
    cuda_bench::print_validation_error(max_error);
    return max_error <= 1.0e-6 ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace cuda_bench
