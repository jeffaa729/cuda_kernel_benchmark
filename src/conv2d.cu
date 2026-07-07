#include <hpc/conv2d.hpp>

#include <cuda_runtime.h>

namespace {

constexpr int kTile = 16;
constexpr int kSharedTile = kTile + 2;

__global__ void conv2d_naive_kernel(const float* input, const float* weight,
                                    const float* bias, float* output,
                                    int batch_size, int c_in, int height,
                                    int width, int c_out, int kernel_h,
                                    int kernel_w, int height_out,
                                    int width_out, int stride, int padding) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * c_out * height_out * width_out;
    if (idx >= total) {
        return;
    }

    const int ow = idx % width_out;
    const int oh = (idx / width_out) % height_out;
    const int co = (idx / (height_out * width_out)) % c_out;
    const int n = idx / (c_out * height_out * width_out);

    float sum = bias[co];
    for (int ci = 0; ci < c_in; ++ci) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int ih = oh * stride - padding + kh;
                const int iw = ow * stride - padding + kw;
                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    const int input_idx =
                        ((n * c_in + ci) * height + ih) * width + iw;
                    const int weight_idx =
                        ((co * c_in + ci) * kernel_h + kh) * kernel_w + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    output[idx] = sum;
}

void launch_conv2d_naive(const float* input, const float* weight,
                         const float* bias, float* output, int batch_size,
                         int c_in, int height, int width, int c_out,
                         int kernel_h, int kernel_w, int height_out,
                         int width_out, int stride, int padding) {
    const int total = batch_size * c_out * height_out * width_out;
    constexpr int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    conv2d_naive_kernel<<<num_blocks, block_size>>>(
        input, weight, bias, output, batch_size, c_in, height, width, c_out,
        kernel_h, kernel_w, height_out, width_out, stride, padding);
}

__global__ void conv2d_3x3_tiled_kernel(const float* input,
                                        const float* weight,
                                        const float* bias, float* output,
                                        int batch_size, int c_in, int height,
                                        int width, int c_out, int height_out,
                                        int width_out) {
    const int ow = blockIdx.x * kTile + threadIdx.x;
    const int oh = blockIdx.y * kTile + threadIdx.y;
    const int co = blockIdx.z % c_out;
    const int n = blockIdx.z / c_out;

    if (n >= batch_size) {
        return;
    }

    __shared__ float input_tile[kSharedTile][kSharedTile];

    float sum = bias[co];
    const int thread_linear = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y;
    constexpr int shared_elements = kSharedTile * kSharedTile;

    for (int ci = 0; ci < c_in; ++ci) {
        for (int tile_idx = thread_linear; tile_idx < shared_elements;
             tile_idx += threads_per_block) {
            const int tile_y = tile_idx / kSharedTile;
            const int tile_x = tile_idx % kSharedTile;
            const int ih = blockIdx.y * kTile + tile_y - 1;
            const int iw = blockIdx.x * kTile + tile_x - 1;

            float value = 0.0f;
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                const int input_idx =
                    ((n * c_in + ci) * height + ih) * width + iw;
                value = input[input_idx];
            }
            input_tile[tile_y][tile_x] = value;
        }
        __syncthreads();

        if (oh < height_out && ow < width_out) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    const int weight_idx = ((co * c_in + ci) * 3 + kh) * 3 + kw;
                    sum += input_tile[threadIdx.y + kh][threadIdx.x + kw] *
                           weight[weight_idx];
                }
            }
        }
        __syncthreads();
    }

    if (oh < height_out && ow < width_out) {
        const int output_idx =
            ((n * c_out + co) * height_out + oh) * width_out + ow;
        output[output_idx] = sum;
    }
}

void launch_conv2d_3x3_tiled(const float* input, const float* weight,
                             const float* bias, float* output, int batch_size,
                             int c_in, int height, int width, int c_out,
                             int height_out, int width_out) {
    const dim3 threads(kTile, kTile);
    const dim3 blocks((width_out + kTile - 1) / kTile,
                      (height_out + kTile - 1) / kTile,
                      batch_size * c_out);
    conv2d_3x3_tiled_kernel<<<blocks, threads>>>(
        input, weight, bias, output, batch_size, c_in, height, width, c_out,
        height_out, width_out);
}

}  // namespace

namespace hpc {

const char* to_string(Conv2DAlgo algo) {
    switch (algo) {
        case Conv2DAlgo::Naive:
            return "naive";
        case Conv2DAlgo::Tiled:
            return "tiled";
    }
    return "unknown";
}

void conv2d(const float* input, const float* weight, const float* bias,
            float* output, int batch_size, int c_in, int height, int width,
            int c_out, int kernel_h, int kernel_w, int height_out,
            int width_out, int stride, int padding, Conv2DAlgo algo) {
    switch (algo) {
        case Conv2DAlgo::Naive:
            launch_conv2d_naive(input, weight, bias, output, batch_size, c_in,
                                height, width, c_out, kernel_h, kernel_w,
                                height_out, width_out, stride, padding);
            return;
        case Conv2DAlgo::Tiled:
            if (kernel_h == 3 && kernel_w == 3 && stride == 1 &&
                padding == 1) {
                launch_conv2d_3x3_tiled(input, weight, bias, output, batch_size,
                                        c_in, height, width, c_out, height_out,
                                        width_out);
                return;
            }
            launch_conv2d_naive(input, weight, bias, output, batch_size, c_in,
                                height, width, c_out, kernel_h, kernel_w,
                                height_out, width_out, stride, padding);
            return;
    }
}

}  // namespace hpc
