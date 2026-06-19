#include <cuda_bench/benchmark.hpp>
#include <cuda_bench/benchmarks.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void print_usage(const char* program) {
    std::cout << "Usage:\n"
              << "  " << program << " all\n"
              << "  " << program << " vector_add [elements] [iterations]\n"
              << "  " << program << " transpose [n]\n";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string benchmark = argc > 1 ? argv[1] : "all";

    if (benchmark == "all") {
        int status = EXIT_SUCCESS;
        status |= cuda_bench::vector_add_benchmark(1ULL << 24, 20);
        std::cout << '\n';
        status |= cuda_bench::transpose_benchmark(4096);
        return status == EXIT_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    if (benchmark == "vector_add") {
        const std::size_t size =
            argc > 2 ? cuda_bench::parse_positive_size(argv[2], "vector size")
                     : 1ULL << 24;
        const std::size_t iterations =
            argc > 3 ? cuda_bench::parse_positive_size(argv[3], "iterations")
                     : 20;
        return cuda_bench::vector_add_benchmark(size, iterations);
    }

    if (benchmark == "transpose") {
        const std::size_t n =
            argc > 2 ? cuda_bench::parse_positive_size(argv[2], "n") : 4096;
        return cuda_bench::transpose_benchmark(n);
    }

    std::cerr << "Unknown benchmark: " << benchmark << "\n\n";
    print_usage(argv[0]);
    return EXIT_FAILURE;
}

