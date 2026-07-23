#pragma once

#include <cstddef>

namespace hpc {

enum class Conv2DAlgo {
    Naive,
    Tiled,
};

const char* to_string(Conv2DAlgo algo);

void conv2d(const float* input, const float* weight, const float* bias,
            float* output, int batch_size, int c_in, int height, int width,
            int c_out, int kernel_h, int kernel_w, int height_out,
            int width_out, int stride, int padding,
            Conv2DAlgo algo = Conv2DAlgo::Tiled);

}  // namespace hpc
