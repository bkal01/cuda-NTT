#pragma once

#include <cublas_v2.h>

#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

void cudaCheck(cudaError_t error, const char *file, int line);

void CudaDeviceInfo();

int div_ceil(int numerator, int denominator);

void runNTTNaive(cublasHandle_t handle, int N, int8_t *A);

void run_kernel(int kernel_num, int N, int8_t *A, cublasHandle_t handle);
