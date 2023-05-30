#pragma once

#ifdef CUDA_ENABLE
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define BLOCK_SIZE 32

namespace matrix_operations {
    void cuda_dense_backward(const double *A, const double *B, double *C_gpu, int rowsA, int colsA, int colsB, double batches);
    void cuda_dense_forward(double *A, double *B, double *C_gpu, int rowsA, int colsA, int colsB, double* biases);
}