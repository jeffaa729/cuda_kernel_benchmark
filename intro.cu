#include <stdio.h>
#include <cuda/cmath>
//kernel definition
__global__ void vectorAdd(float* A, float* B, float* C, int vectorLength) {
    int workIdx = blockIdx.x * blockDim.x + threadIdx.x
    if (workIdx < vectorLength) {
        C[workIdx] = A[workIdx] + B[workIdx]
    }
}

int main() {
    // Kernel invocation
    // first : grid dimension, second : thread block dimension. 
    //dim3 grid(16,16);
    //dim3 block(8,8); 
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
}