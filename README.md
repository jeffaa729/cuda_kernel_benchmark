# CUDA Benchmark Suite

A growing suite for comparing reference CPU implementations with CUDA kernels.
The first benchmark is vector addition; future operations can have independent
dependencies, command-line options, correctness checks, and performance metrics.

## Layout

```text
benchmarks/
  vector_add/main.cu       Operation-specific CPU/GPU code and driver
include/cuda_bench/
  benchmark.hpp            Shared CLI, CPU timing, and reporting helpers
  cuda_utils.cuh           CUDA error checks, device buffers, and events
CMakeLists.txt              Benchmark targets and smoke tests
Makefile                    Convenience wrapper around CMake
```

Add new operations under `benchmarks/<name>/`. Keep kernel implementations and
their reference code local until multiple benchmarks genuinely share them.
Register each executable with `add_cuda_benchmark` in `CMakeLists.txt`.

## Build

Run in WSL with the CUDA 13.3 compiler first on `PATH`:

```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Or use the wrapper:

```bash
make
```

## Run

```bash
./build/bin/vector_add_benchmark
./build/bin/vector_add_benchmark 16777216 20
```

The vector-add benchmark reports CPU time, GPU kernel-only time, GPU end-to-end
time including transfers, effective bandwidth, and maximum validation error.

Run the small correctness smoke tests with:

```bash
ctest --test-dir build --output-on-failure
```

## Growth Guidelines

- Use one executable per operation rather than one large mode-switching binary.
- Measure kernel-only and end-to-end costs separately.
- Keep a straightforward CPU implementation for correctness and comparison.
- Report bandwidth for memory-bound kernels and FLOP/s for compute-bound kernels.
- Add optional libraries such as cuBLAS, cuDNN, or CUTLASS only to targets that
  need them.
- Store scripts and machine-readable result files separately from benchmark code
  when automated sweeps are introduced.
