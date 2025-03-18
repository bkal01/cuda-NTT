#include "kernels.cuh"
#include "runner.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
 
void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
 
void CudaDeviceInfo() {
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, deviceId);

    printf("Device ID: %d\n\
        Name: %s\n\
        Compute Capability: %d.%d\n\
        memoryBusWidth: %d\n\
        maxThreadsPerBlock: %d\n\
        maxThreadsPerMultiProcessor: %d\n\
        maxRegsPerBlock: %d\n\
        maxRegsPerMultiProcessor: %d\n\
        totalGlobalMem: %zuMB\n\
        sharedMemPerBlock: %zuKB\n\
        sharedMemPerMultiprocessor: %zuKB\n\
        totalConstMem: %zuKB\n\
        multiProcessorCount: %d\n\
        Warp Size: %d\n",
            deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
            props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
            props.regsPerBlock, props.regsPerMultiprocessor,
            props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
            props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
            props.multiProcessorCount, props.warpSize);
};

void generate_primitive_roots(int N, int8_t *roots) {
    int8_t root = 3;
    for (int i = 0; i < N; i++) {
        roots[i] = root;
        root = (root * 3) % N;
    }
}
 
void runNTTNaive(cublasHandle_t handle, int N, int8_t *A) {
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);
    naive_ntt<<<gridDim, blockDim>>>(A, N);
}
 
void run_kernel(int kernel_num, int N, int8_t *A, cublasHandle_t handle) {
    switch (kernel_num) {
    case 0:
        runNTTNaive(handle, N, A);
        break;
    default:
        throw std::invalid_argument("Unknown kernel number");
    }
}