#include <cuda_bench/benchmark.hpp>
#include <cuda_bench/benchmarks.hpp>
#include <cuda_bench/cuda_utils.cuh>
#include <hpc/gemm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

bool is_known_algorithm(const std::string& algorithm) {
    return algorithm == "all" || algorithm == "naive" ||
           algorithm == "tiled" || algorithm == "tiled_v2" ||
           algorithm == "tiled_v3" || algorithm == "cublas";
}

std::unordered_set<std::string> selected_algorithms(
    const std::vector<std::string>& algorithms) {
    std::unordered_set<std::string> selected;
    for (const std::string& algorithm : algorithms) {
        if (!is_known_algorithm(algorithm)) {
            std::cerr << "Unknown GEMM algorithm: " << algorithm << '\n';
            std::exit(EXIT_FAILURE);
        }
        if (algorithm == "all") {
            selected.clear();
            return selected;
        }
        selected.insert(algorithm);
    }

    if (!selected.empty()) {
        selected.insert("cublas");
    }
    return selected;
}

bool should_run(const std::unordered_set<std::string>& selected,
                hpc::GemmAlgo algo) {
    return selected.empty() || selected.count(hpc::to_string(algo)) > 0;
}

}  // namespace

namespace cuda_bench {

namespace {

bool run_and_validate_gemm(hpc::GemmAlgo algo,
                           cuda_bench::DeviceBuffer<float>& device_a,
                           cuda_bench::DeviceBuffer<float>& device_b,
                           cuda_bench::DeviceBuffer<float>& device_c,
                           float* gpu_result, const float* reference_result,
                           std::size_t size, std::size_t n) {
    hpc::gemm(device_a.data(), device_b.data(), device_c.data(),
              static_cast<int>(n), algo);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    device_c.copy_to_host(gpu_result);
    for (std::size_t i = 0; i < size; ++i) {
        const float tolerance =
            1.0e-3f * std::max(1.0f, std::abs(reference_result[i]));
        if (std::abs(reference_result[i] - gpu_result[i]) > tolerance) {
            std::cerr << "GEMM " << hpc::to_string(algo)
                      << " validation failed at index " << i
                      << ": reference=" << reference_result[i]
                      << ", GPU=" << gpu_result[i]
                      << ", tolerance=" << tolerance << '\n';
            return false;
        }
    }

    return true;
}

}  // namespace

int gemm_benchmark(std::size_t n,
                   const std::vector<std::string>& algorithms) {
    const std::unordered_set<std::string> selected =
        selected_algorithms(algorithms);

    const std::size_t size = n * n;

    std::vector<float> a(size);
    std::vector<float> b(size);
    std::vector<float> reference_result(size);
    std::vector<float> gpu_result(size);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::generate(a.begin(), a.end(), [&] { return distribution(generator); });
    std::generate(b.begin(), b.end(), [&] { return distribution(generator); });

    cuda_bench::DeviceBuffer<float> device_a(size);
    cuda_bench::DeviceBuffer<float> device_b(size);
    cuda_bench::DeviceBuffer<float> device_c(size);
    cuda_bench::DeviceBuffer<float> device_reference(size);
    device_a.copy_from_host(a.data());
    device_b.copy_from_host(b.data());

    hpc::gemm(device_a.data(), device_b.data(), device_reference.data(),
              static_cast<int>(n), hpc::GemmAlgo::Cublas);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    device_reference.copy_to_host(reference_result.data());

    bool valid = true;
    if (should_run(selected, hpc::GemmAlgo::Naive)) {
        valid &= run_and_validate_gemm(
            hpc::GemmAlgo::Naive, device_a, device_b, device_c,
            gpu_result.data(), reference_result.data(), size, n);
    }
    if (should_run(selected, hpc::GemmAlgo::Tiled)) {
        valid &= run_and_validate_gemm(
            hpc::GemmAlgo::Tiled, device_a, device_b, device_c,
            gpu_result.data(), reference_result.data(), size, n);
    }
    if (should_run(selected, hpc::GemmAlgo::Tiled_v2)) {
        valid &= run_and_validate_gemm(
            hpc::GemmAlgo::Tiled_v2, device_a, device_b, device_c,
            gpu_result.data(), reference_result.data(), size, n);
    }
    if (should_run(selected, hpc::GemmAlgo::Tiled_v3)) {
        valid &= run_and_validate_gemm(
            hpc::GemmAlgo::Tiled_v3, device_a, device_b, device_c,
            gpu_result.data(), reference_result.data(), size, n);
    }
    if (should_run(selected, hpc::GemmAlgo::Cublas)) {
        device_reference.copy_to_host(gpu_result.data());
    }

    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace cuda_bench
