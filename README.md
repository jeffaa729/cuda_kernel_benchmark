# CUDA HPC Library And Benchmark Suite

A small CUDA high-performance computing library with benchmarks that compare CPU
references against library kernels. The library currently exposes vector add and
transpose operations.

## Layout

```text
include/hpc/
  hpc.hpp                  Convenience include for public library APIs
  vector_add.hpp           hpc::vector_add and VectorAddAlgo
  transpose.hpp            hpc::transpose and TransposeAlgo
  reduction.hpp            hpc::reduction and ReductionAlgo
  gemm.hpp                 hpc::gemm and GemmAlgo
src/
  vector_add.cu            Vector-add CUDA implementations
  transpose.cu             Transpose CUDA implementations
  reduction.cu             Reduction CUDA implementations
  gemm.cu                  GEMM CUDA implementations
benchmarks/
  main.cu                  Dispatches benchmark functions from one executable
  vector_add/vector_add_benchmark.cu Benchmarks hpc::vector_add
  transpose/transpose_benchmark.cu   Benchmarks hpc::transpose
  reduction/reduction_benchmark.cu   Benchmarks hpc::reduction
  gemm/gemm_benchmark.cu             Benchmarks hpc::gemm
include/cuda_bench/
  benchmarks.hpp           Benchmark function declarations
  benchmark.hpp            Shared CLI helpers
  cuda_utils.cuh           CUDA error checks and device buffers
CMakeLists.txt              hpc_cuda library, benchmark target, and smoke tests
Makefile                    Convenience wrapper around CMake
```

Add new operations by creating a public API under `include/hpc/`, implementing
CUDA kernels under `src/`, and then adding a benchmark under `benchmarks/<name>/`.
Benchmarks should call the `hpc::` API instead of defining kernels directly.

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
make vector_add ARGS="16777216"
make transpose ARGS="4096"
make reduction ARGS="16777216"
make gemm ARGS="512"
make run BENCH=vector_add ARGS="16777216"
```

The benchmark executable does not print timings. It launches kernels repeatedly
and validates results against CPU references, returning a nonzero exit code on
failure. Use Nsight Compute for profiling:

```bash
ncu ./build/bin/cuda_benchmarks transpose 4096
ncu ./build/bin/cuda_benchmarks vector_add 16777216
ncu ./build/bin/cuda_benchmarks reduction 16777216
ncu ./build/bin/cuda_benchmarks gemm 512
```

Run the small correctness smoke tests with:

```bash
ctest --test-dir build --output-on-failure
```

## Growth Guidelines

- Keep public operation APIs under `hpc::`, with algorithms selected by enums.
- Use one benchmark function per operation and dispatch them from `benchmarks/main.cu`.
- Use Nsight Compute for timing, roofline, and memory-workload metrics.
- Keep a straightforward CPU implementation for correctness and comparison.
- Keep benchmark output minimal so profiler output is easy to read.
- Add optional libraries such as cuBLAS, cuDNN, or CUTLASS only to targets that
  need them.
- Store scripts and machine-readable result files separately from benchmark code
  when automated sweeps are introduced.
