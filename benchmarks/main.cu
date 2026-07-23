#include <cuda_bench/benchmark.hpp>
#include <cuda_bench/benchmarks.hpp>
#include <cuda_bench/cuda_utils.cuh>

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

int fp32_cores_per_sm(int major, int minor) {
    switch (major) {
        case 9:
            return 128;
        case 8:
            return minor == 0 ? 64 : 128;
        case 7:
            return 64;
        case 6:
            return minor == 0 ? 64 : 128;
        case 5:
            return 128;
        case 3:
            return 192;
        case 2:
            return minor == 1 ? 48 : 32;
        default:
            return 0;
    }
}

bool device_info_enabled() {
    const char* value = std::getenv("CUDA_BENCH_DEVICE_INFO");
    return value == nullptr || std::string(value) != "0";
}

void print_device_peak_reference() {
    if (!device_info_enabled()) {
        return;
    }

    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        std::cerr << "CUDA device query failed: " << cudaGetErrorString(status)
                  << '\n';
        return;
    }

    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        std::cerr << "CUDA device properties query failed: "
                  << cudaGetErrorString(status) << '\n';
        return;
    }

    int clock_rate_khz = 0;
    int memory_clock_rate_khz = 0;
    int memory_bus_width_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate,
                                      device));
    CUDA_CHECK(cudaDeviceGetAttribute(&memory_clock_rate_khz,
                                      cudaDevAttrMemoryClockRate, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&memory_bus_width_bits,
                                      cudaDevAttrGlobalMemoryBusWidth,
                                      device));

    const int cores_per_sm =
        fp32_cores_per_sm(properties.major, properties.minor);
    const double core_clock_ghz = clock_rate_khz / 1.0e6;
    const double memory_clock_ghz = memory_clock_rate_khz / 1.0e6;
    const double estimated_fp32_tflops =
        cores_per_sm == 0
            ? 0.0
            : properties.multiProcessorCount * cores_per_sm *
                  core_clock_ghz * 2.0 / 1000.0;
    const double estimated_memory_gbs =
        2.0 * memory_clock_rate_khz * (memory_bus_width_bits / 8.0) /
        1.0e6;

    std::cout << std::fixed << std::setprecision(2)
              << "GPU: " << properties.name << " (SM "
              << properties.major << '.' << properties.minor << ")\n"
              << "Estimated peak reference: ";

    if (estimated_fp32_tflops > 0.0) {
        std::cout << estimated_fp32_tflops << " TFLOP/s FP32, ";
    } else {
        std::cout << "FP32 peak unavailable for this SM version, ";
    }

    std::cout << estimated_memory_gbs << " GB/s memory bandwidth"
              << " (" << properties.multiProcessorCount << " SMs, "
              << core_clock_ghz << " GHz core, "
              << memory_clock_ghz << " GHz memory, "
              << memory_bus_width_bits << "-bit bus)\n\n";
}

void print_usage(const char* program) {
    std::cout << "Usage:\n"
              << "  " << program << " all\n"
              << "  " << program << " vector_add [elements]\n"
              << "  " << program << " transpose [n]\n"
              << "  " << program << " reduction [elements]\n"
              << "  " << program
              << " gemm [n] [naive|tiled|tiled_v2|tiled_v3|cublas]...\n"
              << "  " << program << " softmax [rows] [cols]\n"
              << "  " << program
              << " conv2d [batch] [c_in] [height] [width] [c_out]\n";
}

bool is_positive_size_text(const char* text) {
    if (text == nullptr || *text == '\0') {
        return false;
    }

    for (const char* current = text; *current != '\0'; ++current) {
        if (*current < '0' || *current > '9') {
            return false;
        }
    }

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string benchmark = argc > 1 ? argv[1] : "all";

    print_device_peak_reference();

    if (benchmark == "all") {
        int status = EXIT_SUCCESS;
        status |= cuda_bench::vector_add_benchmark(1ULL << 24);
        std::cout << '\n';
        status |= cuda_bench::transpose_benchmark(4096);
        std::cout << '\n';
        status |= cuda_bench::reduction_benchmark(1ULL << 24);
        std::cout << '\n';
        status |= cuda_bench::gemm_benchmark(512);
        std::cout << '\n';
        status |= cuda_bench::softmax_benchmark(4096, 1024);
        std::cout << '\n';
        status |= cuda_bench::conv2d_benchmark(64, 16, 16, 16, 32);
        return status == EXIT_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    if (benchmark == "vector_add") {
        const std::size_t size =
            argc > 2 ? cuda_bench::parse_positive_size(argv[2], "vector size")
                     : 1ULL << 24;
        return cuda_bench::vector_add_benchmark(size);
    }

    if (benchmark == "transpose") {
        const std::size_t n =
            argc > 2 ? cuda_bench::parse_positive_size(argv[2], "n") : 4096;
        return cuda_bench::transpose_benchmark(n);
    }

    if (benchmark == "reduction") {
        const std::size_t size =
            argc > 2 ? cuda_bench::parse_positive_size(argv[2], "size")
                     : 1ULL << 24;
        return cuda_bench::reduction_benchmark(size);
    }

    if (benchmark == "gemm") {
        std::size_t n = 512;
        int algo_start = 2;
        if (argc > 2 && is_positive_size_text(argv[2])) {
            n = cuda_bench::parse_positive_size(argv[2], "n");
            algo_start = 3;
        }

        std::vector<std::string> algorithms;
        for (int i = algo_start; i < argc; ++i) {
            algorithms.emplace_back(argv[i]);
        }
        return cuda_bench::gemm_benchmark(n, algorithms);
    }

    if (benchmark == "softmax") {
        const std::size_t rows =
            argc > 2 ? cuda_bench::parse_positive_size(argv[2], "rows") : 4096;
        const std::size_t cols =
            argc > 3 ? cuda_bench::parse_positive_size(argv[3], "cols") : 1024;
        return cuda_bench::softmax_benchmark(rows, cols);
    }

    if (benchmark == "conv2d") {
        const std::size_t batch =
            argc > 2 ? cuda_bench::parse_positive_size(argv[2], "batch") : 64;
        const std::size_t c_in =
            argc > 3 ? cuda_bench::parse_positive_size(argv[3], "c_in") : 16;
        const std::size_t height =
            argc > 4 ? cuda_bench::parse_positive_size(argv[4], "height") : 16;
        const std::size_t width =
            argc > 5 ? cuda_bench::parse_positive_size(argv[5], "width") : 16;
        const std::size_t c_out =
            argc > 6 ? cuda_bench::parse_positive_size(argv[6], "c_out") : 32;
        return cuda_bench::conv2d_benchmark(batch, c_in, height, width, c_out);
    }

    std::cerr << "Unknown benchmark: " << benchmark << "\n\n";
    print_usage(argv[0]);
    return EXIT_FAILURE;
}
