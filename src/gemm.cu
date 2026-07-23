#include <hpc/gemm.hpp>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace {
// gemm : D = A * B + C , where A, B, C, D are all N x N matrices stored in row-major order
// Constants for tiled kernel
constexpr int TS = 16;
constexpr int BM = 64; // rows of C per block
constexpr int BN = 64; // cols of C per block
constexpr int BK = 8; // K depth per tile
constexpr int TM = 8; // rows of C per thread
constexpr int THREADS = BM * BN / TM;

// Constants for tiled_v3 kernel
constexpr int BM_V3 = 128; // rows of C per block
constexpr int BN_V3 = 128; // cols of C per block
constexpr int BK_V3 = 16; // K depth per tile
constexpr int TM_V3 = 8; // rows of C per thread
constexpr int TN_V3 = 8; // cols of C per thread
constexpr int THREADS_V3 = BM_V3 * BN_V3 / (TM_V3 * TN_V3);

void cublas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS call failed");
    }
}

// Naive kernel: one thread computes one D[row, col].
// naive coalesced memory access : fast thread index for indexing the row of matrices that are stored in row-major order
__global__ void gemm_naive_kernel(const float* a, const float* b, float* c,
                                  int N) {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < N && col < N) {
        float res = 0.0f;
        for (int k = 0; k < N; k++) {
            res += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = res;
    }
}

void launch_gemm_naive(const float* a, const float* b, float* c, int N) {
    constexpr int tile_dim = 16;
    const dim3 threads(tile_dim, tile_dim);
    const dim3 blocks((N + tile_dim - 1) / tile_dim,
                      (N + tile_dim - 1) / tile_dim);
    gemm_naive_kernel<<<blocks, threads>>>(a, b, c, N);
}

// Tiled kernel: each thread computes one D[row, col] using shared memory.
__global__ void gemm_tiled_kernel(const float* a, const float* b, float* c,
                                  int N) {
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TS + ty;
    const int col = blockIdx.x * TS + tx;
    float acc = 0.0f;

    for (int t = 0; t < (N + TS - 1) / TS; t++) {
        const int tiled_col = t * TS + tx;
        const int tiled_row = t * TS + ty;
        As[ty][tx] = (row < N && tiled_col < N) ? a[row * N + tiled_col] : 0.0f;
        Bs[ty][tx] = (tiled_row < N && col < N) ? b[tiled_row * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TS; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        c[row * N + col] = acc;
    }
}

void launch_gemm_tiled(const float* a, const float* b, float* c, int N) {
    const dim3 threads(TS, TS);
    const dim3 blocks((N + TS - 1) / TS, (N + TS - 1) / TS);
    gemm_tiled_kernel<<<blocks, threads>>>(a, b, c, N);
}

// Tiled kernel + 1D thread tiling :  
__global__ void gemm_tiled_kernel_v2(const float* a, const float* b, float* c, int N) {
    const int tid = threadIdx.x;
    const int local_col = tid % BN; // each thread own one col inside the block tile
    const int global_col = blockIdx.x * BN + local_col; // global col index of C
    // each thread also own 8 rows of C, so we need to compute the global row index for each of the 8 rows 
    const int local_row_base = (tid / BN) * TM; // base local row index for this thread
    /*
    so the thread tid computes :
    C[block_row + local_row_base + 0, global_col]
    C[block_row + local_row_base + 1, global_col]
    C[block_row + local_row_base + 2, global_col]
    C[block_row + local_row_base + 3, global_col]
    ... 
    */
    // instead of 1 accumulate, 
    float acc[TM] = {0.0f}; // accumulate 8 rows of C for this thread
    // shared memory for A and B tiles
    // each K phase load 8 columns of A and 8 rows of B into shared memory, then compute 8 rows of C for each thread
    __shared__ float As[BM][BK]; // shared memory for A tile, 64 rows of A per block, 8 cols of A per tile
    __shared__ float Bs[BK][BN]; // shared memory for B tile, 8 rows of B per tile, 64 cols of B per block

    // warp the load /compute
    for (int tile_k = 0; tile_k < (N + BK - 1) / BK; tile_k++) {
        //Load A tile, Since BM*BK = 64 * 8 = 512, and THREADS = 512, each thread load one element of A tile
        const int a_local_row = tid / BK; // local row index of A tile for this thread
        const int a_local_col = tid % BK; // local col index of A tile for this thread

        const int a_global_row = blockIdx.y * BM + a_local_row; // global row index of A
        const int a_global_col = tile_k * BK + a_local_col; // global col index of A
        As[a_local_row][a_local_col] = (a_global_row < N && a_global_col < N) ? a[a_global_row * N + a_global_col] : 0.0f;
        //Load B tile, Since BK*BN = 8 * 64 = 512, and THREADS = 512, each thread load one element of B tile
        const int b_local_row = tid / BN; // local row index of B tile for this thread
        const int b_local_col = tid % BN; // local col index of B tile for this thread
        const int b_global_row = tile_k * BK + b_local_row; // global row index of B
        const int b_global_col = blockIdx.x * BN + b_local_col; // global col index of B
        Bs[b_local_row][b_local_col] = (b_global_row < N && b_global_col < N) ? b[b_global_row * N + b_global_col] : 0.0f;
        __syncthreads();

        // Compute using register reuse, For each k inside the tile
        for (int k = 0; k< BK; k++) {
            float b_reg = Bs[k][local_col]; // load B[k, col] into register, here load once then reuse for 8 multiply add
            for (int i = 0; i < TM; i++) {
                float a_reg = As[local_row_base + i][k]; // load A[row, k] into register
                acc[i] += a_reg * b_reg; // accumulate
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; i++) {
        const int global_row = blockIdx.y * BM + local_row_base + i;
        if (global_row < N && global_col < N) {
            c[global_row * N + global_col] = acc[i];
        }
    }
}

void launch_gemm_tiled_v2(const float* a, const float* b, float* c, int N) {
    const dim3 threads(THREADS);
    const dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);
    gemm_tiled_kernel_v2<<<blocks, threads>>>(a, b, c, N);
}

// Tiled kernel + 2D thread tiling :
__global__ void gemm_tiled_kernel_v3(const float* a, const float* b, float* c, int N) {
    static_assert(BM_V3 % TM_V3 == 0);
    static_assert(BN_V3 % TN_V3 == 0);
    static_assert(BM_V3 * BK_V3 % THREADS_V3 == 0);
    static_assert(BK_V3 * BN_V3 % THREADS_V3 == 0);

    const int tid = threadIdx.x;

    // Each thread owns one 8x8 tile inside the 128x128 block tile of C.
    const int thread_tile_col = tid % (BN_V3 / TN_V3);
    const int thread_tile_row = tid / (BN_V3 / TN_V3);
    const int local_row_base = thread_tile_row * TM_V3;
    const int local_col_base = thread_tile_col * TN_V3;
    const int global_row_base = blockIdx.y * BM_V3 + local_row_base;
    const int global_col_base = blockIdx.x * BN_V3 + local_col_base;

    /*
    so the thread tid computes one small 8x8 matrix:
    C[global_row_base + 0 : global_row_base + 7,
      global_col_base + 0 : global_col_base + 7]
    */
    float acc[TM_V3][TN_V3] = {0.0f}; // accumulate 64 values of C for this thread
    float a_reg[TM_V3]; // cache 8 values of A from shared memory into registers
    float b_reg[TN_V3]; // cache 8 values of B from shared memory into registers

    // shared memory for A and B tiles
    // each K phase loads A[128x16] and B[16x128], then computes a 128x128 C tile
    __shared__ float As[BM_V3][BK_V3];
    __shared__ float Bs[BK_V3][BN_V3];

    // wrap the load / compute
    for (int tile_k = 0; tile_k < (N + BK_V3 - 1) / BK_V3; tile_k++) {
        // Load A tile. Since BM_V3*BK_V3 = 2048 and THREADS_V3 = 256,
        // each thread loads multiple A elements with a stride loop.
        for (int idx = tid; idx < BM_V3 * BK_V3; idx += THREADS_V3) {
            const int a_local_row = idx / BK_V3;
            const int a_local_col = idx % BK_V3;
            const int a_global_row = blockIdx.y * BM_V3 + a_local_row;
            const int a_global_col = tile_k * BK_V3 + a_local_col;

            As[a_local_row][a_local_col] =
                (a_global_row < N && a_global_col < N)
                    ? a[a_global_row * N + a_global_col]
                    : 0.0f;
        }

        // Load B tile. Since BK_V3*BN_V3 = 2048 and THREADS_V3 = 256,
        // each thread also loads multiple B elements with a stride loop.
        for (int idx = tid; idx < BK_V3 * BN_V3; idx += THREADS_V3) {
            const int b_local_row = idx / BN_V3;
            const int b_local_col = idx % BN_V3;
            const int b_global_row = tile_k * BK_V3 + b_local_row;
            const int b_global_col = blockIdx.x * BN_V3 + b_local_col;

            Bs[b_local_row][b_local_col] =
                (b_global_row < N && b_global_col < N)
                    ? b[b_global_row * N + b_global_col]
                    : 0.0f;
        }
        __syncthreads();

        // Compute using register reuse. For each k inside the tile, cache
        // 8 A values and 8 B values, then compute an 8x8 outer product.
        for (int k = 0; k < BK_V3; k++) {
            for (int i = 0; i < TM_V3; i++) {
                a_reg[i] = As[local_row_base + i][k];
            }
            for (int j = 0; j < TN_V3; j++) {
                b_reg[j] = Bs[k][local_col_base + j];
            }

            for (int i = 0; i < TM_V3; i++) {
                for (int j = 0; j < TN_V3; j++) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    // Write the 8x8 thread tile back to global memory.
    for (int i = 0; i < TM_V3; i++) {
        const int global_row = global_row_base + i;
        if (global_row >= N) {
            continue;
        }
        for (int j = 0; j < TN_V3; j++) {
            const int global_col = global_col_base + j;
            if (global_col < N) {
                c[global_row * N + global_col] = acc[i][j];
            }
        }
    }
}

void launch_gemm_tiled_v3(const float* a, const float* b, float* c, int N) {
    const dim3 threads(THREADS_V3);
    const dim3 blocks((N + BN_V3 - 1) / BN_V3,
                      (N + BM_V3 - 1) / BM_V3);
    gemm_tiled_kernel_v3<<<blocks, threads>>>(a, b, c, N);
}

void launch_gemm_cublas(const float* a, const float* b, float* c, int N) {
    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle));
    cublas_check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublas_check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                             &alpha, b, N, a, N, &beta, c, N));
    cublas_check(cublasDestroy(handle));
}

}  // namespace

namespace hpc {

const char* to_string(GemmAlgo algo) {
    switch (algo) {
        case GemmAlgo::Naive:
            return "naive";
        case GemmAlgo::Tiled:
            return "tiled";
        case GemmAlgo::Tiled_v2:
            return "tiled_v2";
        case GemmAlgo::Tiled_v3:
            return "tiled_v3";
        case GemmAlgo::Cublas:
            return "cublas";
    }
    return "unknown";
}

void gemm(const float* a, const float* b, float* c, int N, GemmAlgo algo) {
    switch (algo) {
        case GemmAlgo::Naive:
            launch_gemm_naive(a, b, c, N);
            return;
        case GemmAlgo::Tiled:
            launch_gemm_tiled(a, b, c, N);
            return;
        case GemmAlgo::Tiled_v2:
            launch_gemm_tiled_v2(a, b, c, N);
            return;
        case GemmAlgo::Tiled_v3:
            launch_gemm_tiled_v3(a, b, c, N);
            return;
        case GemmAlgo::Cublas:
            launch_gemm_cublas(a, b, c, N);
            return;
    }
}

}  // namespace hpc
