#include <cuda_bench/benchmark.hpp>
#include <cuda_bench/benchmarks.hpp>
#include <cuda_bench/cuda_utils.cuh>
#include <cuda_bench/validation.hpp>
#include <hpc/vector_add.hpp>

#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

CUDA_BENCH_NOINLINE void vector_add_cpu(const float* a, const float* b, float* c,
                                        std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

}  // namespace

namespace cuda_bench {

int vector_add_benchmark(std::size_t size) {
    constexpr hpc::VectorAddAlgo algo = hpc::VectorAddAlgo::Naive;

    std::vector<float> a(size);
    std::vector<float> b(size);
    std::vector<float> cpu_result(size);
    std::vector<float> gpu_result(size);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::generate(a.begin(), a.end(), [&] { return distribution(generator); });
    std::generate(b.begin(), b.end(), [&] { return distribution(generator); });
    vector_add_cpu(a.data(), b.data(), cpu_result.data(), size);

    cuda_bench::DeviceBuffer<float> device_a(size);
    cuda_bench::DeviceBuffer<float> device_b(size);
    cuda_bench::DeviceBuffer<float> device_c(size);
    device_a.copy_from_host(a.data());
    device_b.copy_from_host(b.data());

    hpc::vector_add(device_a.data(), device_b.data(), device_c.data(), size,
                    algo);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    device_c.copy_to_host(gpu_result.data());
    bool valid = true;
    for (std::size_t i = 0; i < size; ++i) {
        if (cpu_result[i] != gpu_result[i]) {
            valid = false;
            break;
        }
    }

    hpc::vector_add(device_a.data(), device_b.data(), device_c.data(), size,
                    hpc::VectorAddAlgo::Vectorized);
    CUDA_CHECK(cudaDeviceSynchronize());
    valid &= validate_result(device_c, cpu_result, "vectorized add", 0.0F);

    // Residual backward adds the same upstream gradient to both branches.
    // Only the aligned prefix belongs to this vectorized backward contract.
    const std::size_t aligned = size - size % 4;
    std::vector<float> dx(size, 0.25F), db(size, -0.5F);
    DeviceBuffer<float> device_dx(size), device_db(size);
    device_dx.copy_from_host(dx.data()); device_db.copy_from_host(db.data());
    if (aligned) {
        hpc::vector_add_backward(device_dx.data(), device_db.data(), device_c.data(), aligned);
    }
    for (std::size_t i = 0; i < aligned; ++i) {
        dx[i] += cpu_result[i];
        db[i] += cpu_result[i];
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    valid &= validate_result(device_dx, dx, "vector add left gradient", 0.0F);
    valid &= validate_result(device_db, db, "vector add right gradient", 0.0F);
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace cuda_bench
