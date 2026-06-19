# CUDA Benchmark Suite

A growing suite for comparing reference CPU implementations with CUDA kernels.
The first benchmark is vector addition; future operations can have independent
dependencies, command-line options, correctness checks, and performance metrics.

## Layout

```text
benchmarks/
  main.cu                  Dispatches benchmark functions from one executable
  vector_add/vector_add.cu Vector-add benchmark function
  transpose/transpose.cu   Transpose benchmark function
include/cuda_bench/
  benchmarks.hpp           Benchmark function declarations
  benchmark.hpp            Shared CLI, CPU timing, and reporting helpers
  cuda_utils.cuh           CUDA error checks, device buffers, and events
CMakeLists.txt              Benchmark targets and smoke tests
Makefile                    Convenience wrapper around CMake
```

Add new operations under `benchmarks/<name>/`. Keep kernel implementations and
their reference code local until multiple benchmarks genuinely share them.
Declare the benchmark function in `include/cuda_bench/benchmarks.hpp`, call it
from `benchmarks/main.cu`, and add the `.cu` file to the `cuda_benchmarks`
target in `CMakeLists.txt`.

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
make run
make vector_add
make vector_add ARGS="16777216 20"
make transpose ARGS="1024 2048"
make run BENCH=vector_add ARGS="16777216 20"
```

The vector-add benchmark reports CPU time, GPU kernel time, effective bandwidth,
and maximum validation error.

Run the small correctness smoke tests with:

```bash
ctest --test-dir build --output-on-failure
```

## Growth Guidelines

- Use one benchmark function per operation and dispatch them from `benchmarks/main.cu`.
- Measure kernel execution time; host/device copies are setup and validation work.
- Keep a straightforward CPU implementation for correctness and comparison.
- Report bandwidth for memory-bound kernels and FLOP/s for compute-bound kernels.
- Add optional libraries such as cuBLAS, cuDNN, or CUTLASS only to targets that
  need them.
- Store scripts and machine-readable result files separately from benchmark code
  when automated sweeps are introduced.
