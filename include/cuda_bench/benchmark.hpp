#pragma once

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(_MSC_VER)
#define CUDA_BENCH_NOINLINE __declspec(noinline)
#else
#define CUDA_BENCH_NOINLINE __attribute__((noinline))
#endif

namespace cuda_bench {

inline std::size_t parse_positive_size(const char* text, const char* name) {
    try {
        const unsigned long long value = std::stoull(text);
        if (value == 0 || value > std::numeric_limits<std::size_t>::max()) {
            throw std::out_of_range("invalid range");
        }
        return static_cast<std::size_t>(value);
    } catch (const std::exception&) {
        std::cerr << "Invalid " << name << ": " << text << '\n';
        std::exit(EXIT_FAILURE);
    }
}

template <typename Function>
double measure_cpu_ms(std::size_t iterations, Function&& function) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        function();
    }
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           static_cast<double>(iterations);
}

inline void print_metric_row(const std::string& method, double milliseconds,
                             double metric, const std::string& unit) {
    std::cout << std::left << std::setw(24) << method << std::right
              << std::setw(12) << std::fixed << std::setprecision(3)
              << milliseconds << std::setw(16) << metric << ' ' << unit << '\n';
}

inline void print_time_row(const std::string& method, double milliseconds) {
    std::cout << std::left << std::setw(24) << method << std::right
              << std::setw(12) << std::fixed << std::setprecision(3)
              << milliseconds << std::setw(21) << "n/a" << '\n';
}

}  // namespace cuda_bench
