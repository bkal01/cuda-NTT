#pragma once

__global__ void naive_ntt(int8_t *A, int N) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= N || y >= N) {
        return;
    }
    
    
}