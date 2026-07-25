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

// Constants for tiled_v4 kernel
constexpr int BM_V4 = 128; // rows of C per block
constexpr int BN_V4 = 128; // cols of C per block
constexpr int BK_V4 = 16;  // K tile depth
constexpr int TM_V4 = 8;   // rows per thread
constexpr int TN_V4 = 8;   // cols per thread
constexpr int THREADS_V4 = BM_V4 * BN_V4 / (TM_V4 * TN_V4);
constexpr int VEC = 4;     // float4 = 4 floats, vectorized load/store

// constants for tiled_v5 kernel
constexpr int BM_V5 = 128; // rows of C per block
constexpr int BN_V5 = 128; // cols of C per block
constexpr int BK_V5 = 16;  // K tile depth
constexpr int WM_V5 = 64;  // rows of C per warp
constexpr int WN_V5 = 32;  // cols of C per warp

constexpr int WARPS_M_V5 = BM_V5 / WM_V5; // 2
constexpr int WARPS_N_V5 = BN_V5 / WN_V5; // 4
constexpr int WARPS_V5 = WARPS_M_V5 * WARPS_N_V5; // 8
constexpr int THREADS_V5 = WARPS_V5 * 32; // 256

constexpr int TM_V5 = 8; // rows per thread
constexpr int TN_V5 = 8; // cols per thread

constexpr int WARP_THREADS_M_V5 = 8;
constexpr int WARP_THREADS_N_V5 = 4;

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

// Tiled kernel + 2D thread tiling + vectorized memory access :
__global__ void gemm_tiled_kernel_v4(const float* a, const float* b, float* c,
                                     int N) {
    static_assert(BK_V4 % VEC == 0);
    static_assert(BN_V4 % VEC == 0);
    static_assert(TN_V4 % VEC == 0);
    static_assert(BM_V4 % TM_V4 == 0);
    static_assert(BN_V4 % TN_V4 == 0);
    static_assert(BM_V4 * (BK_V4 / VEC) % THREADS_V4 == 0);
    static_assert(BK_V4 * (BN_V4 / VEC) % THREADS_V4 == 0);

    // A is stored transposed in shared memory. The global A tile is
    // A[128][16], but shared memory stores it as AsT[16][128].
    // This makes each thread read its 8 A values from a contiguous row.
    __shared__ float AsT[BK_V4][BM_V4];
    __shared__ float Bs[BK_V4][BN_V4];

    const int tid = threadIdx.x;
    const int thread_tile_col = tid % (BN_V4 / TN_V4);
    const int thread_tile_row = tid / (BN_V4 / TN_V4);
    const int local_row_base = thread_tile_row * TM_V4;
    const int local_col_base = thread_tile_col * TN_V4;
    const int global_row_base = blockIdx.y * BM_V4 + local_row_base;
    const int global_col_base = blockIdx.x * BN_V4 + local_col_base;

    // Each thread computes an 8x8 tile of C:
    // C[global_row_base + 0 : global_row_base + 7,
    //   global_col_base + 0 : global_col_base + 7]
    float acc[TM_V4][TN_V4] = {0.0f};
    float a_reg[TM_V4];
    float b_reg[TN_V4];

    // Wrap the load / compute over K tiles.
    for (int tile_k = 0; tile_k < (N + BK_V4 - 1) / BK_V4; tile_k++) {
        // Load A tile. A[128][16] has 2048 floats. With float4, this is
        // 512 vector loads, so 256 threads each load two vectors.
        for (int idx = tid; idx < BM_V4 * (BK_V4 / VEC); idx += THREADS_V4) {
            const int a_local_row = idx / (BK_V4 / VEC);
            const int a_local_col_vec = idx % (BK_V4 / VEC);
            const int a_local_col = a_local_col_vec * VEC;
            const int a_global_row = blockIdx.y * BM_V4 + a_local_row;
            const int a_global_col = tile_k * BK_V4 + a_local_col;
            const int a_global_idx = a_global_row * N + a_global_col;

            float vals[VEC] = {};
            if (a_global_row < N && a_global_col + VEC - 1 < N &&
                a_global_idx % VEC == 0) {
                const float4 v =
                    *reinterpret_cast<const float4*>(&a[a_global_idx]);
                vals[0] = v.x;
                vals[1] = v.y;
                vals[2] = v.z;
                vals[3] = v.w;
            } else {
                for (int x = 0; x < VEC; x++) {
                    if (a_global_row < N && a_global_col + x < N) {
                        vals[x] = a[a_global_idx + x];
                    }
                }
            }

            // Store A transposed: AsT[k][row].
            for (int x = 0; x < VEC; x++) {
                AsT[a_local_col + x][a_local_row] = vals[x];
            }
        }

        // Load B tile. B[16][128] also has 2048 floats, so we use the same
        // float4 loading pattern and keep B in normal row-major shared layout.
        for (int idx = tid; idx < BK_V4 * (BN_V4 / VEC); idx += THREADS_V4) {
            const int b_local_row = idx / (BN_V4 / VEC);
            const int b_local_col_vec = idx % (BN_V4 / VEC);
            const int b_local_col = b_local_col_vec * VEC;
            const int b_global_row = tile_k * BK_V4 + b_local_row;
            const int b_global_col = blockIdx.x * BN_V4 + b_local_col;
            const int b_global_idx = b_global_row * N + b_global_col;

            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (b_global_row < N && b_global_col + VEC - 1 < N &&
                b_global_idx % VEC == 0) {
                v = *reinterpret_cast<const float4*>(&b[b_global_idx]);
            } else {
                float vals[VEC] = {};
                for (int x = 0; x < VEC; x++) {
                    if (b_global_row < N && b_global_col + x < N) {
                        vals[x] = b[b_global_idx + x];
                    }
                }
                v = make_float4(vals[0], vals[1], vals[2], vals[3]);
            }

            *reinterpret_cast<float4*>(&Bs[b_local_row][b_local_col]) = v;
        }
        __syncthreads();

        // Compute using register reuse. For each k, load 8 A values and
        // 8 B values into registers, then do the 8x8 outer product.
        for (int k = 0; k < BK_V4; k++) {
            for (int i = 0; i < TM_V4; i++) {
                a_reg[i] = AsT[k][local_row_base + i];
            }

            for (int jv = 0; jv < TN_V4 / VEC; jv++) {
                const float4 v = *reinterpret_cast<const float4*>(
                    &Bs[k][local_col_base + jv * VEC]);
                b_reg[jv * VEC + 0] = v.x;
                b_reg[jv * VEC + 1] = v.y;
                b_reg[jv * VEC + 2] = v.z;
                b_reg[jv * VEC + 3] = v.w;
            }

            for (int i = 0; i < TM_V4; i++) {
                for (int j = 0; j < TN_V4; j++) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    // Vectorized write of the 8x8 thread tile back to C.
    for (int i = 0; i < TM_V4; i++) {
        const int global_row = global_row_base + i;
        if (global_row >= N) {
            continue;
        }

        for (int jv = 0; jv < TN_V4 / VEC; jv++) {
            const int global_col = global_col_base + jv * VEC;
            const int global_idx = global_row * N + global_col;

            if (global_col + VEC - 1 < N && global_idx % VEC == 0) {
                const float4 v =
                    make_float4(acc[i][jv * VEC + 0], acc[i][jv * VEC + 1],
                                acc[i][jv * VEC + 2], acc[i][jv * VEC + 3]);
                *reinterpret_cast<float4*>(&c[global_idx]) = v;
            } else {
                for (int x = 0; x < VEC; x++) {
                    if (global_col + x < N) {
                        c[global_idx + x] = acc[i][jv * VEC + x];
                    }
                }
            }
        }
    }
}

void launch_gemm_tiled_v4(const float* a, const float* b, float* c, int N) {
    const dim3 threads(THREADS_V4);
    const dim3 blocks((N + BN_V4 - 1) / BN_V4,
                      (N + BM_V4 - 1) / BM_V4);
    gemm_tiled_kernel_v4<<<blocks, threads>>>(a, b, c, N);
}

// Tiled kernel + 2D warp tiling + 2D thread tiling + vectorized memory access.
__global__ void gemm_tiled_kernel_v5(const float* a, const float* b, float* c,
                                     int N) {
    static_assert(BK_V5 % VEC == 0);
    static_assert(BN_V5 % VEC == 0);
    static_assert(TN_V5 % VEC == 0);
    static_assert(BM_V5 % WM_V5 == 0);
    static_assert(BN_V5 % WN_V5 == 0);
    static_assert(WARP_THREADS_M_V5 * WARP_THREADS_N_V5 == 32);
    static_assert(WM_V5 == WARP_THREADS_M_V5 * TM_V5);
    static_assert(WN_V5 == WARP_THREADS_N_V5 * TN_V5);
    static_assert(BM_V5 * (BK_V5 / VEC) % THREADS_V5 == 0);
    static_assert(BK_V5 * (BN_V5 / VEC) % THREADS_V5 == 0);

    // Cache one block tile of A and B in shared memory.
    // A is transposed from A[128][16] to AsT[16][128], so each thread can
    // read its 8 A values as AsT[k][row + i].
    __shared__ float AsT[BK_V5][BM_V5];
    __shared__ float Bs[BK_V5][BN_V5];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Step 1: choose which 64x32 warp tile this warp owns inside the
    // 128x128 block tile. There are 2 warp rows and 4 warp columns.
    const int warp_col = warp_id % WARPS_N_V5;
    const int warp_row = warp_id / WARPS_N_V5;

    // Step 2: arrange the 32 lanes as an 8x4 grid inside the warp tile.
    // Each lane owns one 8x8 thread tile, so the warp covers 64x32.
    const int lane_row = lane_id / WARP_THREADS_N_V5;
    const int lane_col = lane_id % WARP_THREADS_N_V5;

    // Step 3: convert warp/lane coordinates into this thread's local and
    // global C tile coordinates.
    const int warp_tile_row_base = warp_row * WM_V5;
    const int warp_tile_col_base = warp_col * WN_V5;
    const int local_row_base = warp_tile_row_base + lane_row * TM_V5;
    const int local_col_base = warp_tile_col_base + lane_col * TN_V5;
    const int global_row_base = blockIdx.y * BM_V5 + local_row_base;
    const int global_col_base = blockIdx.x * BN_V5 + local_col_base;

    // Step 4: keep this thread's 8x8 output tile in registers until all
    // K tiles are accumulated.
    float acc[TM_V5][TN_V5] = {0.0f};
    float a_reg[TM_V5];
    float b_reg[TN_V5];

    // Step 5: walk over K in 16-wide chunks. Each chunk computes
    // C_block += A_block[128x16] * B_block[16x128].
    for (int tile_k = 0; tile_k < (N + BK_V5 - 1) / BK_V5; tile_k++) {
        // Load A tile with float4. Each block needs 128*16 floats, or
        // 512 vector loads, so 256 threads each load two vectors.
        for (int idx = tid; idx < BM_V5 * (BK_V5 / VEC); idx += THREADS_V5) {
            const int a_local_row = idx / (BK_V5 / VEC);
            const int a_local_col_vec = idx % (BK_V5 / VEC);
            const int a_local_col = a_local_col_vec * VEC;
            const int a_global_row = blockIdx.y * BM_V5 + a_local_row;
            const int a_global_col = tile_k * BK_V5 + a_local_col;
            const int a_global_idx = a_global_row * N + a_global_col;

            float vals[VEC] = {};
            if (a_global_row < N && a_global_col + VEC - 1 < N &&
                a_global_idx % VEC == 0) {
                const float4 v =
                    *reinterpret_cast<const float4*>(&a[a_global_idx]);
                vals[0] = v.x;
                vals[1] = v.y;
                vals[2] = v.z;
                vals[3] = v.w;
            } else {
                for (int x = 0; x < VEC; x++) {
                    if (a_global_row < N && a_global_col + x < N) {
                        vals[x] = a[a_global_idx + x];
                    }
                }
            }

            // Store A transposed: global A[row][k] becomes shared AsT[k][row].
            for (int x = 0; x < VEC; x++) {
                AsT[a_local_col + x][a_local_row] = vals[x];
            }
        }

        // Load B tile with float4 and keep it in normal row-major layout.
        for (int idx = tid; idx < BK_V5 * (BN_V5 / VEC); idx += THREADS_V5) {
            const int b_local_row = idx / (BN_V5 / VEC);
            const int b_local_col_vec = idx % (BN_V5 / VEC);
            const int b_local_col = b_local_col_vec * VEC;
            const int b_global_row = tile_k * BK_V5 + b_local_row;
            const int b_global_col = blockIdx.x * BN_V5 + b_local_col;
            const int b_global_idx = b_global_row * N + b_global_col;

            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (b_global_row < N && b_global_col + VEC - 1 < N &&
                b_global_idx % VEC == 0) {
                v = *reinterpret_cast<const float4*>(&b[b_global_idx]);
            } else {
                float vals[VEC] = {};
                for (int x = 0; x < VEC; x++) {
                    if (b_global_row < N && b_global_col + x < N) {
                        vals[x] = b[b_global_idx + x];
                    }
                }
                v = make_float4(vals[0], vals[1], vals[2], vals[3]);
            }
            *reinterpret_cast<float4*>(&Bs[b_local_row][b_local_col]) = v;
        }
        __syncthreads();

        // Step 6: for each k inside the shared tile, load one 8-value A
        // fragment and one 8-value B fragment, then accumulate their outer
        // product into this thread's 8x8 register tile.
        for (int k = 0; k < BK_V5; k++) {
            for (int i = 0; i < TM_V5; i++) {
                a_reg[i] = AsT[k][local_row_base + i];
            }
            for (int jv = 0; jv < TN_V5 / VEC; jv++) {
                const float4 v = *reinterpret_cast<const float4*>(
                    &Bs[k][local_col_base + jv * VEC]);
                b_reg[jv * VEC + 0] = v.x;
                b_reg[jv * VEC + 1] = v.y;
                b_reg[jv * VEC + 2] = v.z;
                b_reg[jv * VEC + 3] = v.w;
            }
            for (int i = 0; i < TM_V5; i++) {
                for (int j = 0; j < TN_V5; j++) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    // Step 7: write the completed 8x8 thread tile back to global memory.
    for (int i = 0; i < TM_V5; i++) {
        const int global_row = global_row_base + i;
        if (global_row >= N) {
            continue;
        }
        for (int jv = 0; jv < TN_V5 / VEC; jv++) {
            const int global_col = global_col_base + jv * VEC;
            const int global_idx = global_row * N + global_col;

            if (global_col + VEC - 1 < N && global_idx % VEC == 0) {
                const float4 v =
                    make_float4(acc[i][jv * VEC + 0], acc[i][jv * VEC + 1],
                                acc[i][jv * VEC + 2], acc[i][jv * VEC + 3]);
                *reinterpret_cast<float4*>(&c[global_idx]) = v;
            } else {
                for (int x = 0; x < VEC; x++) {
                    if (global_col + x < N) {
                        c[global_idx + x] = acc[i][jv * VEC + x];
                    }
                }
            }
        }
    }
}

void launch_gemm_tiled_v5(const float* a, const float* b, float* c, int N) {
    const dim3 threads(THREADS_V5);
    const dim3 blocks((N + BN_V5 - 1) / BN_V5,
                      (N + BM_V5 - 1) / BM_V5);
    gemm_tiled_kernel_v5<<<blocks, threads>>>(a, b, c, N);
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
        case GemmAlgo::Tiled_v4:
            return "tiled_v4";
        case GemmAlgo::Tiled_v5:
            return "tiled_v5";
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
        case GemmAlgo::Tiled_v4:
            launch_gemm_tiled_v4(a, b, c, N);
            return;
        case GemmAlgo::Tiled_v5:
            launch_gemm_tiled_v5(a, b, c, N);
            return;
        case GemmAlgo::Cublas:
            launch_gemm_cublas(a, b, c, N);
            return;
    }
}

}  // namespace hpc
